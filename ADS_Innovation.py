
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

# Useful settings
#%matplotlib inline
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

boston = load_boston()

for line in boston.DESCR.split("\n")[5:29]:
    print(line)

boston_df = pd.DataFrame(data=boston.data, columns = boston.feature_names)
boston_df["Price"] = boston.target

boston_df.head()
# %%
X = boston.data
y = boston.target
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
            'subsample': [.2, .4, .6, .8, 1],
            'reg_lambda': [0, 1e-1, 1,10, 20, 50, 100], 
            'reg_alpha': [0, 1e-1, 1, 2, 7, 10, 50, 100],
            'num_leaves': [5,10,20,40], 
            'n_estimators': [20, 30, 40, 50, 75, 100, 200, 300, 500, 1000],
            'max_depth': [1, 2, 3, 5, 8, 15],
            'colsample_bytree': [.2, .4, .6]
        }
        model = RandomizedSearchCV(estimator = model, param_distributions = grid, 
        n_iter = 500, cv = cv, scoring = 'neg_mean_squared_error', verbose = 1)
    # try 50-100
    model.fit(x, y)
    hyper = None
    if tune:
        hyper = model.best_params_
    return model

model_lgb = lgb(X_train, y_train, cv = 2, tune = True)

# %%
model_lgb.fit(X_train, y_train)

# %%
test_pred = model_lgb.predict(X_test)
train_pred = model_lgb.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
### XGBoost
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %%
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

# %%
## update parameters with best params
# Repeat for other parameters, max_depth and min_child_weight shown as examples
params = {
    # Parameters that we are going to tune.
    'max_depth':3,#6,
    'min_child_weight': 1e-3,#1,
    'eta':0.1,#.3,
    'subsample': 1,
    'colsample_bytree': 0.6,#1,
    # Other parameters
    'objective':'reg:squarederror',
    'eval_metric':"mae"
}
num_boost_round = 999

model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

# %%
test_pred = model_xgb.predict(dtest)
train_pred = model_xgb.predict(dtrain)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
### Random Forest
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)
model_rf.fit(X_train, y_train.ravel())
# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

model_rf.fit(X_train, y_train)

# %%
test_pred = model_rf.predict(X_test)
train_pred = model_rf.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

