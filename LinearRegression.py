# Linear regression code and evaluation

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

# %% Multicollinearity
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
vif["features"] = X.columns
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
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
x_trainOLS = sm.add_constant(X_train)
est = sm.OLS(y_train, x_trainOLS)
est2 = est.fit()
res = est2.resid
fig = sm.qqplot(res, line = "s") # Standardised line fit due to using standard scaling
plt.show()
## QQ plot has heavy-tail
print(est2.summary())
# %%
coeff_df = pd.DataFrame(lin_reg.coef_.T, X.columns, columns=['Coefficient'])
coeff_df

# %%
pred = lin_reg.predict(X_train)
#plt.scatter(y_test, pred)

# %%
import statsmodels.api as sm
residuals = (y_train - pred)

fig = sm.qqplot(residuals, alpha=0.01)
plt.show()
# %%
# Instantiate the linear model and visualizer
from yellowbrick.regressor import ResidualsPlot

model = LinearRegression(normalize=False)
visualizer = ResidualsPlot(model, hist=False, qqplot=False)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show() 

# %%
# Take a random sample of the residuals 
import random
rand = np.random.choice(residuals, 1000)
fig = sm.qqplot(rand, alpha=0.2)
plt.rcParams["font.family"] = "sans-serif"
plt.title("QQ Plot")
plt.show()
# %%
residuals = (pred - y_train)
rand = np.random.choice(residuals, 1000)
plt.title("Residuals")
plt.scatter(pred, residuals)

# %%
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()    
# %%
from sklearn.linear_model import LassoCV
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import AlphaSelection

# Create a list of alphas to cross-validate against
alphas = np.logspace(-10, 1, 10)

# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)
visualizer.fit(X, y)
visualizer.show()

# %%
from sklearn import datasets
from yellowbrick.target import FeatureCorrelation
fig = plt.figure()

fig.set_size_inches(20, 14)
# Create a list of the feature names
features = np.array(X.columns)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()   
fig.savefig("Test2.png")  

# %%
from sklearn import datasets
from yellowbrick.target import FeatureCorrelation

# Create a list of the feature names
features = np.array(X.columns)

# Create a list of the discrete features
discrete = [False for _ in range(len(features))]
discrete[1] = True

# Instantiate the visualizer
visualizer = FeatureCorrelation(method='mutual_info-regression', labels=features)

visualizer.fit(X, y, discrete_features=discrete, random_state=0)
visualizer.show()