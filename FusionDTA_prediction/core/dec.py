import numpy as np
import pandas as pd

def read_result_bin(path):
    dt = np.dtype(
        [
            ('lotus_id', '<i4'),
            ('pid', '<i4'),
            ('affinity', '<f4')
        ]
    )
    record = np.fromfile(path, dt)
    return pd.DataFrame(record)