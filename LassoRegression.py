
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
#X = X.head(10000)

y = data['Total_Loss'].values
#y = y.head(10000)
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
#y_train = pipeline.fit_transform(y_train)
#y_test = pipeline.transform(y_test)
# %%
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
## Lasso
# %%
from sklearn.linear_model import Lasso

model = Lasso(alpha=0, # CV's best alpha
              precompute=True, 
#               warm_start=True, 
#              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


# %%
#est = model.fit(X_train, y_train)
#est.summary()
# %%
#import statsmodels.api as sm
#residuals = (y_test - pred)
#fig = sm.qqplot(residuals[:,0], alpha=0.01)
#plt.show()


# %%
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
from sklearn.linear_model import LassoCV
reg = LassoCV()
reg.fit(X_train, y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
# %%
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# %%
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (16.0, 20.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.tight_layout() # Ensure variable names are not cropped out
#plt.savefig("Plots\\FeatureImportanceCorrelation.png", facecolor=fig.get_facecolor())

# %% 
train_pred = reg.predict(X_train)
plt.scatter(y_train, train_pred)
# %%
# plot feature importance anonymously.
df2 = pd.DataFrame(imp_coef, columns=["Value"])
abs_weights = np.abs(df2["Value"].values)
sorted_weights = np.sort(abs_weights)
plt.plot(sorted_weights, linestyle = "-", marker = "o", color = 'k', lw = 1)
plt.title('Absolute Feature Importance of Independent Variables', y=1, fontsize=28)
plt.xlabel("Variable",  fontsize=22)
plt.ylabel("Importance",  fontsize=22)
plt.figsize=(20, 10)
#plt.xlim(-0.4e10, 0.6e10)
plt.savefig('Plots\\FeatureImportance.png', transparent=True, bbox_inches='tight', dpi = 300)
# %%
# Multicollinearity
#https://www.andreaperlato.com/mlpost/deal-multicollinearity-with-lasso-regression/

# %% Equivalent to Ridge
from sklearn.linear_model import LassoCV
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LassoCV(alphas=0, cv=cv, n_jobs=-1)
# fit model
model.fit(X_train, y_train)
# summarise chosen configuration
print('alpha: %f' % model.alpha_)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
