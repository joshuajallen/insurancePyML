# Creating a synthetic version of the dataset

#%%
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
X = data.drop(columns=['Trial', 'Timestep', 'Total_Loss'])
y = data[['Total_Loss']]
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %% 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
X_train.to_numpy()


# %%
from copulas.multivariate import GaussianMultivariate

def create_synthetic(X, y):
    """
    This function combines X and y into a single dataset D, models it
    using a Gaussian copula, and generates a synthetic dataset S. It
    returns the new, synthetic versions of X and y.
    """
    dataset = np.concatenate([X, np.expand_dims(y, 1)], axis=1)

    model = GaussianMultivariate()
    model.fit(dataset)

    synthetic = model.sample(len(dataset))

    X = synthetic.values[:, :-1]
    y = synthetic.values[:, -1]

    return X, y

X_synthetic, y_synthetic = create_synthetic(X_train.to_numpy(), y_train.to_numpy())
# %%
np.expand_dims(y_train.to_numpy(), 1)
# %%
X_train.to_numpy()
# %%
np.concatenate([X_train.to_numpy(), np.expand_dims(y_train.to_numpy(), 1)])
# %%
