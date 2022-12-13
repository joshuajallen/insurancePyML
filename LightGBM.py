
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
X = data.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
y = data['Total_Loss'].values
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
# %%
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# %%

# evaluate lightgbm ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor

# define the model
model = LGBMRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# %%
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
def lgb_tune(x,y, hyper = None):
    if hyper is None:
        
        if y.sum() <= 500: # do four fold cross validation
            return lgb(x,y, cv = 4, tune = True)
        else: # split data set in 2 parts - one for training, the other for testing
            cv = StratifiedShuffleSplit(n_splits = int(np.where(y.sum() > 1000, 2, 1)), test_size = .2)
            return lgb(x,y, cv = cv, tune = True)
    else: 
        lgb(x,y, tune = False, hyper = hyper)
        

def lgb (x,y, tune = False, cv = 1, hyper = None):
    
    if hyper is None:
        model = LGBMRegressor(n_estimators = 100)
    else:
        model = LGBMRegressor(**hyper)
        
    if tune:
        grid = {
            'subsample': [.2, .4, .6, .8],
            'reg_lambda': [0, 1e-1, 1,10, 20, 50, 100], 
            'reg_alpha': [0, 1e-1, 1, 2, 7, 10, 50, 100],
            'num_leaves': [5,10,20,40], 
            'n_estimators': [20, 30, 40, 50, 75, 100, 200, 300, 500, 1000],
            'max_depth': [1, 2, 3, 5, 8, 15],
            'colsample_bytree': [.2, .4, .6]
        }
        model = RandomizedSearchCV(estimator = model, param_distributions = grid, 
        n_iter = 672, cv = cv, scoring = 'neg_mean_squared_error', verbose = 1)
    # try 50-100
    model.fit(x, y)
    hyper = None
    if tune:
        hyper = model.best_params_
    return model

model = lgb(X_train, y_train, cv = 2, tune = True)

# %%
model.fit(X_train, y_train)

# %%
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
# %%
# {'subsample': 0.8, 'reg_lambda': 50, 'reg_alpha': 1, 'num_leaves': 40, 'n_estimators': 1000, 'max_depth': 15, 'colsample_bytree': 0.2}