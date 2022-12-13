# Feature selection for regression

# Check the distribution of all variables
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Useful settings
#%matplotlib inline
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

data = pd.read_parquet('C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data\\insurance_dataset.gzip')

# %%
#data = data.drop(columns=['Trial', 'Timestep'])
#X = data.drop(columns=['Total_Loss'])
#y = data[['Total_Loss']]
df = X.copy()
#df = df.iloc[:, : 5]
for i in df.columns:
    plt.figure()
    plt.hist(df[i], bins = 1000)
# %%
print(df.size)
dfm = df.melt(var_name='columns')
g = sns.FacetGrid(dfm, col='columns')
g = (g.map(sns.distplot, 'value'))

# %% 
X = X.iloc[:5,:]
X.to_csv("Example.csv")
# %%
data.plot(subplots=True)

plt.tight_layout()
plt.show()


# potential features to remove are those which are not Gaussian distributed.
# %%
unique_counts = data.nunique()
unique_counts[unique_counts < 10000]
#unique_counts.to_csv("UniqueCountsPerColumn.csv")
# %%
X = data.drop(columns=['Trial', 'Timestep', 'Total_Loss'])
y = data[['Total_Loss']]
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)