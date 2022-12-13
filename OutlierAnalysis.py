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
# %%
out_percentage = 0.1
out_values = data[data.Total_Loss < data.Total_Loss.quantile(out_percentage)]
# %%
X = out_values.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
#X = X.head(10000) # If wanting to reduce sample size

y = out_values['Total_Loss'].values

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
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()

model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(210, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(420, activation='relu'))
model.add(Dropout(0.1)) # Becomes more robust, as no dependence. Randomly drops, main method to overcome overfitting.
model.add(Dense(1))
# https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
model.compile(optimizer = Adam(0.1), loss='mse') #Learning rate is small? 0.0001 try Should speed up. batch size and learning rate related?

r = model.fit(X_train, y_train,
              validation_data=(X_test,y_test),
              batch_size=52, #64
              epochs=20, verbose = 0) # maybe add more as it's learning

# %%
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# %%
