##
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
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
# define the model
model = RandomForestRegressor(verbose = 1) #bigger the data = less trees
# max features - 10 upto 105
# evaluate the model
cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# %%
model.fit(X_train, y_train)
# %%
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

"""
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed: 116.2min finished
RandomForestRegressor(verbose=1)

MAE: 117636658.97877687
MSE: 2.5539533921730908e+16
RMSE: 159810931.7967044
R2 Square 0.9610853475924148

"""