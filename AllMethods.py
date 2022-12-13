# Insurance Proxy Models
# Aidan Saggers
# Advanced Analytics Division, Bank of England
# Summer Research Project, University of Exeter
"""
The following notebook contains all the code required to train,
test and evaluate a number of methods to replicate the proxy
model which has produced the scenarios and scenario results 
in the dataset. Naturally the data has been removed from this
notebook and the variable names removed.
The script follows this rough outline
1. Load data and split into train and test sets
2. Exploratory Data Analysis
3. Methods, tuning and evaluations
    Linear Regression  
    Ridge Regression  
    Lasso Regression  
    Elastic Nets
    Xgboost
    Artificial Neural Networks
4. Summary of methods

To-do:
Create function/table of results for each method
"""
# %%
# List of all libraries required
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
# sklearn and general ML libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from yellowbrick.regressor import ResidualsPlot
# Method libraries
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Useful settings
#%matplotlib inline
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
directory = "C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data"

# %% Load data function and load data
def load_data(directory, filename, test_size, random_state):
    """
    Load dataset from parquet file, remove ID columns and split
    into train and test datasets.
    """
    data = pd.read_parquet(os.path.join(directory, filename))
    X = data.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
    y = data['Total_Loss'].values # Important to not create pandas
    # dataframe for dependent variable as it causes issues with plotting
    # QQ plot
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = load_data(directory, "insurance_dataset.gzip", 0.2, 42)
# %% Functions for evaluation and cross validation
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

# %% Perform scaling
pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
# Print shape of datasets to check train/test split
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# %% Linear Regression

lin_reg = LinearRegression(normalize=False)
lin_reg.fit(X_train, y_train)
pred = lin_reg.predict(X_train)

# %%
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
# %% 
def residuals_plot(model, trainx, testx, trainy, testy):
    visualizer = ResidualsPlot(model, hist=False, qqplot=False)
    visualizer.fit(trainx, trainy)  # Fit the training data to the visualizer
    visualizer.score(testx, testy)  # Evaluate the model on the test data
    visualizer.show() 

# %% Plot residuals for Linear Reg
residuals_plot(lin_reg, X_train, X_test, y_train, y_test) 
# %%
# %% QQ plot
def qqplot(title, model, trainy, pred):
    residuals = (trainy - np.squeeze(pred))
    fig = sm.qqplot(residuals, alpha=0.01, markerfacecolor='k', line = 's') # “s” - standardized line, the expected 
    # order statistics are scaled by the standard deviation of the given sample and have the mean added to them
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('QQ Plot for ' + title, fontsize=16)
    ax.xaxis.get_label().set_fontsize(14)
    ax.yaxis.get_label().set_fontsize(14)
    ax.get_lines()[1].set_color("grey")
    ax.get_lines()[1].set_lw(3)
    ax.get_lines()[1].set_alpha(0.7)
    ax.get_lines()[1].set_linestyle('--')
    ax.grid()
    plt.tight_layout()
    plt.savefig('Plots\\QQ_Plot_' + title + '.png', transparent=True)
    plt.show()
    

# %% Plot QQ plot for Lin Reg
qqplot("Linear Regression", lin_reg, y_train, pred)

# %%
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label.squeeze(), 'Predicted': predictions.squeeze()})
    print(df_results.head())
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    print(df_results.head())
    
    return df_results
    
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7,
     scatter_kws={'facecolors':'black', 'alpha':0.3})
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max(), 100000)
    plt.plot(line_coords, line_coords,  # X and y points
             color='grey', linestyle='--', alpha = 0.7, linewidth = 3)
    plt.title('Linear Assumption: Actual results vs Predicted')
    plt.show()

df_results = calculate_residuals(lin_reg, X_train, y_train)
len(np.arange(df_results.min().min(), df_results.max().max(), 100000))
linear_assumption(lin_reg, X_train, y_train)
# %%
def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')

normal_errors_assumption(lin_reg, X_train, y_train)
from statsmodels.stats.diagnostic import normal_ad
p_value = normal_ad(df_results['Residuals'])[1]
print(p_value)