import pandas as pd
def preprocess(filepath):
    data = pd.read_csv(filepath, sep='\t+', engine='python',
                       names=['question', 'passage', 'relation'])
    print(data.columns)
    print("Data Types {0}",data.dtypes)