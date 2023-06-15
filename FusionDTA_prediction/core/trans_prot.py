import esm
import pickle
import torch
import datetime
import pandas as pd

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

class ProteinDataLoader:
    '''
    请用户根据自己的数据输入格式，继承该类并重写read_data_file方法以
    自定义数据读取过程。该方法要求返回两个列表。第一个列表为为蛋白质id，
    该id可根据用户的后续数据分析流程自行定义数据类型与编号规则；第二个
    列表为包含蛋白质序列字符串的列表。
    项目中使用的ProteinCsvLoader类便是通过继承该类定义的：
    ```
    import pandas as pd
    class ProteinCsvLoader(ProteinDataLoader):
        def read_data_file(self, path:str):
            protein_df = pd.read_csv(path)
            return protein_df['id'], protein_df['seq']
    ```
    该类的对象可传入generate_protein_representation函数作为loader参
    数的值：
    ```
    # path为要读取的csv文件路径
    loader = ProteinCsvLoader(path)
    generate_protein_representation(loader, 'task.pickle')
    ```
    亦可作为迭代器使用：
    ```
    # path为要读取的csv文件路径
    for pid, token in ProteinCsvLoader(path):
        print(pid, token)
        break
    ```
    '''
    def __init__(self, path: str):
        self.data = self.read_data_file(path)
        self.batch_converter = alphabet.get_batch_converter()
        
    def read_data_file(self, path: str):
        raise NotImplementedError("Method 'read_data_file' not implemented. Use help(ProteinDataLoader) to get more infomation.")
    
    def __iter__(self):
        def iterator():
            for pid, prot in zip(*self.data):
                *_, token = self.batch_converter([(pid, prot[:1022])])
                yield pid, token
        return iterator()

    
class ProteinCsvLoader(ProteinDataLoader):
    def read_data_file(self, path:str):
        protein_df = pd.read_csv(path)
        return protein_df['id'], protein_df['seq']

    
def generate_protein_representation(data: ProteinDataLoader, save_path: str):
    '''
    该函数用于调用esm-1b生成蛋白质表示。
    data: ProteinDataLoader的子类，指定读取的文件与读取方式。
    save_path: 结果的保存路径，结果将被保存为pickle文件。
    '''
    
    data_dict = {}
    n = 0
    
    print(f'[{datetime.datetime.now()}] Started.')
    
    for pid, token in data:
        with torch.no_grad():
            results = model(token, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].numpy()
        data_dict[pid] = token_representations
        
        if n % 50 == 0 and n > 0:
            print(f'[{datetime.datetime.now()}] {n} proteins transformed.')
        n += 1
    
    print(f'[{datetime.datetime.now()}] Saving to {save_path} ...')
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'[{datetime.datetime.now()}] Finished.')
    
    return data_dict