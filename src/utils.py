import pandas as pd

def load_and_clean_data(data_path='data/census-bureau.data', columns_path='data/census-bureau.columns'):
    '''loads census data, strips whitespace, and maps column arrays'''
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f]
        
    df = pd.read_csv(data_path, header=None, names=columns)
    
    df_clean = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    df_clean = df_clean.replace('?', 'NA')
    
    return df_clean