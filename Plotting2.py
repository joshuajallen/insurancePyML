# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Useful settings
#%matplotlib inline
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
plt.rcParams["font.family"] = "sans-serif"

data = pd.read_parquet('C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data\\insurance_dataset.gzip')
# %%
X = data.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
#X = X.head(10000) # If wanting to reduce sample size

y = data['Total_Loss'].values
# y = y.head(10000)
# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# %%
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

#y_train = pipeline.fit_transform(y_train) # Unnecessary, but keep for now
#y_test = pipeline.transform(y_test)
# %%
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=False)
lin_reg.fit(X_train,y_train)


# %%
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
# %%
ext_values = data[data.Total_Loss < data.Total_Loss.quantile(.025)]
# %%
X = ext_values.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
#X = X.head(10000) # If wanting to reduce sample size

y = ext_values['Total_Loss'].values

# %%
X = pipeline.transform(X)
# %%
test_pred = lin_reg.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)

# %%
test_pred = model.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)
# %%
X_new = pd.DataFrame(X_test)
y_new = pd.DataFrame(y_test, columns = ["Total_Loss"])
# %%
data_new = X_new.join(y_new)
# %%
ext_values = data_new[data_new.Total_Loss < data_new.Total_Loss.quantile(.25)]
# %%
X = data_new.drop(columns=['Total_Loss'])
#X = X.head(10000) # If wanting to reduce sample size

y = data_new['Total_Loss'].values

# %%
X = pipeline.fit_transform(X_test)
# %%
test_pred = lin_reg.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)

# %%
test_pred = model.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)
# %%
