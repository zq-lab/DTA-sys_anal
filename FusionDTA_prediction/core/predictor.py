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
    请用户根据自己的数据输入格式，继承该类并重写read_np与read_prot方
    法以自定义数据读取过程。
    其中read_np方法要求返回两个列表，第一个列表为化合物的id，该id可根
    据用户的后续数据分析流程自行定义数据类型与编号规则；第二个列表为包
    含无构型信息的SMILES字符串的列表。
    read_prot方法要求返回一个包含蛋白质id的列表，该id可根据用户的后续
    数据分析流程自行定义数据类型与编号规则。
    
    保存的批次大小由np_batch_size与prot_batch_size参数指定。每次生成
    的批次大小为np_batch_size * prot_batch_size。
    
    注意本类仅生成保存的批次，不指定模型预测时的批次大小。预测时模型使
    用的批次大小(即batch_size)由batch_predict函数的predict_batch_s
    ize参数指定。
    
    项目中使用的BatchGeneratorFromCsv类便是通过继承该类定义的：
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
    该类的对象可传入batch_generator函数作为loader参数的值：
    ```
    # np_path为本项目中存放药物id与smiles的文件的路径
    # prot_path为本项目中存放蛋白质id的文件的路径
    # pkl_path为本项目中存放蛋白质表示pickle的文件的路径
    # result_path为本项目中保存结果的路径
    # log_path为本项目中保存日志的路径
    batgen = BatchGeneratorFromCsv(np_path, prot_path, 10, 311)
    batch_predict(batgen, pkl_path, result_path, log_path)
    ```
    '''
    def __init__(self, np_path:str, prot_path:str, np_batch_size:int, prot_batch_size=None):
        '''
        np_path: 存放药物id与smiles的文件的路径
        prot_path: 存放蛋白质id的文件的路径
        np_batch_size: 药物的批次大小
        prot_batch_size: 蛋白质的批次大小
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
    用于批量预测DTA并即时保存。
    
    batch_generator: BatchGenerator子类的对象, 使用help(BatchGener
        ator)获取更多信息。
    prot_repr_path: 使用generate_protein_representation函数生成的存
        放蛋白质表示的pickle文件的路径。
    result_file_path: 结果保存的路径。结果将被保存为一个二进制文件，在
        每个batch计算结束后会追加写入该文件。
    predict_batch_size: 预测时的批次大小。请根据显存调整。
    log_path: None或日志文件的保存路径。如果为None则不生成日志文件，否
        则生成日志文件。
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
