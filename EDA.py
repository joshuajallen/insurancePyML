import pandas as pd

data = pd.read_csv('C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data\\1 YE 2017 simulation file incl total loss.csv')
data.to_parquet('C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data\\insurance_dataset.gzip', compression = 'gzip')

