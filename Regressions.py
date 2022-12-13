# All regression code for the project

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
X = data.drop(columns=['Total_Loss'])
X = X.head(10000)

y = data[['Total_Loss']]
y = y.head(10000)
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
y_train = pipeline.fit_transform(y_train)
y_test = pipeline.transform(y_test)
# %%
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=False)
lin_reg.fit(X_train,y_train)
# %%
print(lin_reg.intercept_)
# %%
lin_reg.coef_

# %%
est = sm.OLS(y_train, X_train)
est2 = est.fit()
res = est2.resid
fig = sm.qqplot(res)
plt.show()
#print(est2.summary())
# %%
coeff_df = pd.DataFrame(lin_reg.coef_.T, X.columns, columns=['Coefficient'])
coeff_df
# %%
est2.resid

# %%
pred = lin_reg.predict(X_train)
#plt.scatter(y_test, pred)

# %%
import statsmodels.api as sm
residuals = (y_train - pred)

fig = sm.qqplot(residuals[:,0], alpha=0.01)
plt.show()

# %% 
d = density(residuals)
plot(d,main='Residual KDE Plot',xlab='Residual value')
# %%
plt.hist(residuals, 100)
# %%
rand = residuals.sample(100)
fig = sm.qqplot(rand, alpha=0.2)
plt.show()

# %%
# Instantiate the linear model and visualizer
model = LinearRegression(normalize=False)
visualizer = ResidualsPlot(model, hist=True)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show() 

# %% 
data = sm.datasets.longley.load(as_pandas=False)
data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#exog = sm.add_constant(data.exog)
#mod_fit = sm.OLS(data.endog, exog).fit()
#res = mod_fit.resid # residuals
#fig = sm.qqplot(res)
#plt.show()
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# %%
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot
#os.system("start /B start cmd.exe @cmd /k px --gateway --debug --proxy=browse.vip.dmz.bankofengland.co.uk:8080 --port=8081")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load a regression dataset
#X, y = load_concrete()

# Instantiate the linear model and visualizer
model = LinearRegression(normalize=True)
visualizer = ResidualsPlot(regr, hist=False, qqplot=True)

visualizer.fit(diabetes_X_train, diabetes_y_train)  # Fit the training data to the visualizer
visualizer.score(diabetes_X_test, diabetes_y_test)  # Evaluate the model on the test data
visualizer.show()  

# %%
sns.distplot((y_test - pred), bins=50)
# %%
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
# %%
from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(lin_reg)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show() 
## Ridge
# %%
from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
 

## Lasso
# %%
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
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
