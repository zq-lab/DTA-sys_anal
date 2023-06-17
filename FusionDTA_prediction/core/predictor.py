import pandas as pd
import torch
import torch.utils.data
from src.utils import collate, AminoAcid
from src.models.DAT import DAT3
from src.utils import Smiles
import pickle
import numpy as np
import copy
from datetime import datetime
from numpy.core import records

# 日志
class Log:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'w') as f:
            f.write('')
    def write(self, log):
        with open(self.path, 'a') as f:
            f.write(log)
            f.write('\n')

class DrugTargetDataset(torch.utils.data.Dataset):
    def __init__(self, X0, z, pid, is_target_pretrain=True, is_drug_pretrain=False, self_link=True):
        self.X0 = X0 #graph
        self.pid = pid
        self.smilebet = Smiles()
        smiles = copy.deepcopy(self.X0)
        smiles = [x.encode('utf-8').upper() for x in smiles]
        smiles = [torch.from_numpy(self.smilebet.encode(x)).long() for x in smiles]
        self.X2 = smiles #smiles
        self.is_target_pretrain = is_target_pretrain
        self.is_drug_pretrain = is_drug_pretrain
        self.output_place_holder = torch.tensor(0.)
        self.z = z

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        prot = torch.from_numpy(self.z[self.pid[i]]).squeeze()
        return [prot, self.X2[i], self.output_place_holder]



use_cuda = torch.cuda.is_available()

embedding_dim = 1280
rnn_dim = 128
hidden_dim = 256
graph_dim = 256

n_heads = 8
dropout = 0.3
alpha = 0.2
is_pretrain = True
Alphabet = AminoAcid()

#model
model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)

model.load_state_dict(torch.load('pretrained_models/DAT_best_davis.pkl')['model'], strict=False)

if use_cuda:
    model.cuda()

model.eval()

def get_predict_fn(prot_repr_path):
    with open(prot_repr_path, 'rb') as f:
        z = pickle.load(f)
    
    def predict(df, batch_size=270):

        test_drug = list(df['compound_iso_smiles'])
        lotus_id = list(df['np_id'])
        pid = list(df['pid'])

        dataset_test = DrugTargetDataset(test_drug, z, pid, is_target_pretrain=is_pretrain, self_link=False)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         collate_fn=collate
                                                        )
        
        result = []
        
        with torch.no_grad():
            for protein, smiles, _ in dataloader_test:
                if use_cuda:
                    protein = [p.cuda() for p in protein]
                    smiles = [s.cuda() for s in smiles]

                _, out = model(protein, smiles)

                out = out.cpu()

                result.extend(out.tolist())
                          
        return lotus_id, pid, result
    return predict

# 数据保存格式
dt = np.dtype(
        [
            ('lotus_id', '<i4'),
            ('pid', '<i4'),
            ('affinity', '<f4')
        ]
    )




class BatchGenerator:
    '''
    Users are invited to inherit this class and override the read_np
    and read_prot methods to customize the data reading process
    according to their own data input format. 
    
    The read_np method requires the return of two lists, the first 
    list is the id of the compound, which can be defined according to
    the user's subsequent data analysis process, and the second list
    is a list of SMILES strings without configuration information. 
    
    The data type and numbering rules can be defined by the user
    according to the subsequent data analysis process. The size of
    the saved batch is specified by the np_batch_size and
    prot_batch_size parameters. The batch size is np_batch_size *
    prot_batch_size.
    
    Note that this class only generates the saved batches and does
    not specify the batch size for model prediction. The batch size
    used by the model at prediction time (i.e., batch_size) is
    specified by the predict_batch_size parameter of the batch_predict
    function. 
    
    The BatchGeneratorFromCsv class used in the project is defined
    by inheriting from this class:

    ```
    class BatchGeneratorFromCsv(BatchGenerator):
        def read_np(self, path):
            df_np = pd.read_csv(path)
            np_id = df_np['lotus_id'].to_list()
            np_smiles = df_np['smiles'].to_list()
            return np_id, np_smiles

        def read_prot(self, path):
            df_protein = pd.read_csv(path)
            return df_protein['id'].to_list()
    ```
    Objects of this class can be passed into the batch_generator 
    function as the value of the loader parameter:
    ```
    # np_path is the path to the file where the drug ids and smiles
    # are stored in this project.
    # prot_path is the path to the file that holds the protein id 
    # in this project.
    # pkl_path is the path to the file where the protein pickle is 
    # stored in this project.
    # result_path is the path where the results are stored in this
    # project.
    # log_path is the path where the logs are stored in this project
    
    batgen = BatchGeneratorFromCsv(np_path, prot_path, 10, 311)
    batch_predict(batgen, pkl_path, result_path, log_path)
    ```
    '''
    def __init__(self, np_path:str, prot_path:str, np_batch_size:int, prot_batch_size=None):
        '''
        np_path: path to the file where drug ids and smiles are stored.
        prot_path: path to the file where the protein id is stored.
        np_batch_size: the batch size of the drug.
        prot_batch_size: the batch size of the protein.
        '''
        self.np_id_list, self.np_smiles_list = self.read_np(np_path)
        assert len(self.np_id_list) == len(self.np_smiles_list), 'length of np_id and np_smiles must be the same.'
        self.prot_list = self.read_prot(prot_path)
        
        self.np_batch_size = np_batch_size
        self.prot_batch_size = prot_batch_size
        
    def read_np(self, path):
        raise NotImplementedError("Method 'read_np' not implemented. Use help(BatchGenerator) to get more infomation.")
    
    def read_prot(self, path):
        raise NotImplementedError("Method 'read_prot' not implemented. Use help(BatchGenerator) to get more infomation.")
    
    def __iter__(self):
        # 如果未指定prot_batch_size
        if self.prot_batch_size is None:
            def iterator():
                np_batch = 0
                while np_batch < len(self.np_id_list):
                    # 构造一个以'np_id'、'compound_iso_smiles'与所有蛋白质id为列名的DataFrame
                    df = pd.DataFrame(columns=['np_id', 'compound_iso_smiles', *self.prot_list])
                    # 向'np_id'与'compound_iso_smiles'列填入化合物id与无构型smiles
                    df['np_id'] = self.np_id_list[np_batch * self.np_batch_size: (np_batch + 1) * self.np_batch_size]
                    df['compound_iso_smiles'] = self.np_smiles_list[np_batch * self.np_batch_size: (np_batch + 1) * self.np_batch_size]
                    # 使用pd.melt生成化合物与蛋白质的一一组合
                    df = pd.melt(df, id_vars=['np_id', 'compound_iso_smiles'], value_vars=self.prot_list)
                    df.rename(columns={'variable': 'pid'}, inplace=True)
                    
                    np_batch += self.np_batch_size
                    
                    yield df
        # 如果指定了prot_batch_size
        else:
            def iterator():
                np_batch = 0
                while np_batch < len(self.np_id_list):
                    prot_batch = 0
                    while prot_batch < len(self.prot_list):
                        
                        prot_list_part = self.prot_list[prot_batch * self.prot_batch_size: (prot_batch + 1) * self.prot_batch_size]
                        
                        df = pd.DataFrame(columns=['np_id', 'compound_iso_smiles', *prot_list_part])

                        df['np_id'] = self.np_id_list[np_batch * self.np_batch_size: (np_batch + 1) * self.np_batch_size]
                        df['compound_iso_smiles'] = self.np_smiles_list[np_batch * self.np_batch_size: (np_batch + 1) * self.np_batch_size]

                        df = pd.melt(df, id_vars=['np_id', 'compound_iso_smiles'], value_vars=prot_list_part)
                        df.rename(columns={'variable': 'pid'}, inplace=True)
                        
                        prot_batch += self.prot_batch_size
                        
                        yield df
                    
                    np_batch += self.np_batch_size
                    
        return iterator()

class BatchGeneratorFromCsv(BatchGenerator):
    def read_np(self, path):
        df_np = pd.read_csv(path)
        np_id = df_np['lotus_id'].to_list()
        np_smiles = df_np['smiles'].to_list()
        return np_id, np_smiles

    def read_prot(self, path):
        df_protein = pd.read_csv(path)
        return df_protein['id'].to_list()
    
    
    
    
def batch_predict(batch_generator, prot_repr_path, result_file_path, predict_batch_size=270, log_path=None):
    '''
    Used to batch predict DTA and save it on-the-fly.
    
    batch_generator: object of BatchGenerator subclass, use
    help(BatchGenerator ator) for more information.
    
    prot_repr_path: A pickle file generated using the
    generate_protein_representation function to store path of the pickle
    file generated using the generate_protein_representation function.
    
    result_file_path: The path where the results will be saved. The results
    will be saved as a binary file that will be appended after The results
    will be saved as a binary file, which will be appended after each batch
    calculation.
    
    predict_batch_size: the size of the batch when predicting. Please adjust
    according to the memory.
    
    log_path: None or the path to save the log file. If None, no log file will
    be generated, if not If None, no log file will be generated, if not, log
    file will be generated.
    '''
    predict = get_predict_fn(prot_repr_path)
    
    if log_path:
        log = Log(log_path)
    batch = 0
    
    try:
        for df in batch_generator:
            if log_path:
                log.write(f'[{datetime.now()}] Start predict batch {batch}.')

            lotus_id, pid, result = predict(df, batch_size = predict_batch_size)

            if log_path:
                log.write(f'[{datetime.now()}] Saving Batch {batch}.')
            
            # 将一批次的结果追加写入bin文件中
            with open(result_file_path, 'ab') as f:
                f.write(records.fromarrays([lotus_id, pid, [str(r)[:8] for r in result]], dtype=dt).tobytes())

            if log_path:
                log.write(f'[{datetime.now()}] Batch {batch} saved.')

            batch += 1
            
    except Exception as e:
        if log_path:
            log.write(f'[{datetime.now()}] Err at {batch}.')
            log.write(str(e))
        raise e
