import numpy as np
import pandas as pd

# read fusionDTA predication result (bin)
def read_fusionDTA(path):
    dt = np.dtype(
        [
            ('lotus_id', '<i4'),
            ('pid', '<i4'),
            ('affinity', '<f4')
        ]
    )
    record = np.fromfile(path, dt)
    return pd.DataFrame(record)

# read TTD uniprot txt file
def read_TTD_table(file,skiprows=22,sep='\t'):
    data = pd.read_table(file, sep=sep, skiprows=skiprows, names=['ttd_id','key','value'])
    data.dropna(how='all',inplace=True)
    
    data = data.pivot(index='ttd_id',columns='key',values='value')

    data['name'] = data['TARGNAME'].str.split('(',expand=True)[0]
    data['name'] = data['name'].str.strip()

    data['abbr_name'] = data['TARGNAME'].str.extract(r'[(](.*?)[)]')
    data['abbr_name'] = data['abbr_name'].str.strip()

    return data