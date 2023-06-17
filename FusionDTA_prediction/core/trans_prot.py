import esm
import pickle
import torch
import datetime
import pandas as pd

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

class ProteinDataLoader:
    '''
    Users are invited to inherit this class and override the read_data_file method 
    according to their data input format to customize the data reading process. 
    
    This method requires two lists to be returned. The first list is a protein id. 
    The id can be defined according to the user's subsequent data analysis process;
    the second The second list is a list of protein sequence strings.
    
    The ProteinCsvLoader class used in the project is defined by inheriting from
    this class:
    ```
    import pandas as pd
    class ProteinCsvLoader(ProteinDataLoader).
        def read_data_file(self, path:str).
            protein_df = pd.read_csv(path)
            return protein_df['id'], protein_df['seq']
    ```
    The object of this class can be passed into the generate_protein_representation
    function as the loader parameter values of:
    ```
    # path is the path to the csv file to read
    loader = ProteinCsvLoader(path)
    generate_protein_representation(loader, 'task.pickle')
    ```
    Can also be used as an iterator:
    ```
    # path is the path to the csv file to read
    for pid, token in ProteinCsvLoader(path).
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
    This function is used to call esm-1b to generate the protein representation.
    data: subclass of ProteinDataLoader, specifies the file to be read and the reading method.
    save_path: path to save the results, the results will be saved as a pickle file.
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