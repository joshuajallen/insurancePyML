## Artificial Neural Networks


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
# model.add(Dropout(0.2)) #Uncomment these.
model.add(Dropout(0.1)) # Becomes more robust, as no dependence. Randomly drops, main method to overcome overfitting.
model.add(Dense(1))
# https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
model.compile(optimizer = Adam(0.001), loss='mse') #Learning rate is small? 0.0001 try Should speed up. batch size and learning rate related?

r = model.fit(X_train, y_train,
              validation_data=(X_test,y_test),
              batch_size=32, #64
              epochs=10) # maybe add more as it's learning


"""
 MAE: 7798290.349006716
MSE: 132555380413144.45
RMSE: 11513269.75333873
R2 Square 0.9997980250317277       
"""

# %%
plt.figure(figsize = (10, 6))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# %%
test_pred_nn = model.predict(X_test)
train_pred_nn = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred_nn)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred_nn)

# %%
print(r.history.keys())
#  "Accuracy"
plt.plot(r.history['acc'])
plt.plot(r.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



# %%
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras import backend as K

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(105, input_dim=105, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination]) # Change to usual metric
	return model
estimator = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# %%
# A small Multilayer Perceptron (MLP) model will be defined to address this problem 
# and provide the basis for exploring different loss functions.

#The model will expect 105 features as input as defined by the problem. The model 
# will have one hidden layer with 105 nodes and will use the rectified linear 
# activation function (ReLU). The output layer will have 1 node, given the one 
# real-value to be predicted, and will use the linear activation function.

# %%
# https://stackoverflow.com/questions/40334537/how-can-i-loop-the-hidden-layers-in-the-program-written-using-python3
def make_hidden(input_num, hidden_num):
  return {'weight' :tf.Variable(tf.random_normal([input_num, 
                                                  hidden_num])),
          'biases' :tf.Variable(tf.random_normal([hidden_num]))}

def make_output(hidden_num, output_classes):
  return {'weight' :tf.Variable(tf.random_normal([hidden_num, 
                                                  n_classes])),
          'biases' :tf.Variable(tf.random_normal([n_classes]))}

n_nodes = [0, 784, 500, 500, 500]
     #     |___ dummy value so that n_nodes[i] and n_nodes[i+1] stores
     #          the input and hidden number of the i-th hidden layer
     #          (1-based) because layers[0] is the input.
def neural_network_model(data, n_nodes):
   layers = []*len(n_nodes)
   layers[0] = data
   for i in in range(1, n_nodes-1):
     hidden_i = make_hidden(n_nodes[i], n_nodes[i+1]
     layers[i] = tf.add(tf.matmul(layers[i-1], hidden_i['weight']), hidden_i['biases'])
     layers[i] = tf.nn.relu(layers[i])

   output_layer = make_output(n_nodes[-1], n_classes)
   output = tf.matmul(layers[-1], output_layer['weight']) + output_layer['biases']

   return output          