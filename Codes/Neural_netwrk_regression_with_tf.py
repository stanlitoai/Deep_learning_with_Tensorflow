import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

# Create features
X = np.array([-7, -4, -1, 2, 5, 8, 11, 14])

# Create labels
y = np.array([3, 6, 9, 12, 15, 18, 21, 24])

# Visualize it
plt.scatter(X, y)


#Before we do any modelling, can you calculate the pattern between `X` and `y`?
#For example, say I asked you, based on this data what the `y` value would be if `X` was 17.0?

#Or how about if `X` was -10.0?
#This kind of pattern discover is the essence of what we'll be building neural networks to do for us."

X.shape, y.shape

 "## Regression input shapes and output shapes\n",
  "\n",
  "One of the most important concepts when working with neural networks are the input and output shapes.\n",
  "\n",
  "The **input shape** is the shape of your data that goes into the model.\n",
  "\n",
  "The **output shape** is the shape of your data you want to come out of your model.\n",
  "\n",
  "These will differ depending on the problem you're working on.\n",
  "\n",
  "Neural networks accept numbers and output numbers. These numbers are typically represented as tensors (or arrays).\n",
  "\n",
  "Before, we created data using NumPy arrays, but we could do the same with tensors."
  
  
# Example input and output shapes of a regresson model
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
house_info, house_price

input_shape = X.shape
output_shape = y.shape

input_shape, output_shape

ndim_X = X.ndim
ndim_y = y.ndim

ndim_X, ndim_y

  "Huh?\n",
  "\n",
  "From this it seems our inputs and outputs have no shape?\n",
  "\n",
  "How could that be?\n",
  "\n",
  "It's because no matter what kind of data we pass to our model, it's always going to take as input and return as ouput some kind of tensor.\n",
  "\n",
  "But in our case because of our dataset (only 2 small lists of numbers), we're looking at a special kind of tensor, more specificially a rank 0 tensor or a scalar."


X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

X, y

X.shape, y.shape
X.ndim , y.ndim


plt.scatter(X, y)

##Steps in modelling with tensorflow
#Creating the model - define the input  and output layers, as well as the hidden layers
of a deep learning mode.

##2. Compiling a model - define the l0ss funtion( in other wprds, the function which tells our model
how wrong it is ) and the optimizer(tells our model how to improve the pattern its learnig) and
evaluation metrics (what we can use to interpret the performance of our model)

##3. Fitting a model - letting the model try to find patterns btww X and y (features and label)

#Set random seed
import keras


tf.random.set_seed(42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(ndim_X)
])


model.compile(loss = tf.keras.losses.mae, 
              optimizer= tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(X, y, epochs=100)

model.predict([17])


##Improving the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

 
model.compile(loss = tf.keras.losses.mae, 
              optimizer= tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(X, y, epochs=100)

model.predict([17])



#improving the model



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
    ])


model.compile(loss=tf.keras.losses.mae, 
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["mae"])

model.fit(X, y, epochs=100)

model.predict([17.0])



















##Using a big dataset

X = tf.range(-100, 100, 4)

#Make labels for the dataset

y = X + 10

X,y


##Visualize the data

plt.scatter(X, y)

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

len(X_train), len(X_test), len(y_train), len(y_test)


#Visualizing our data

plt.figure(figsize=(8, 5))
#Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training data")
#Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Test data")
#Show legend
plt.legend();


##Building a neural network for our data

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

#Compile the model

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

##Fitting the model

model.fit(X_train, y_train, epochs=100, verbose=1)

##Visualization
model.summary()


X[0], y[0]











################################################



##Let's create a model which builds automatically by defining the input_shape arguement

tf.random.set_seed(42)

##Create a model
model = tf.keras.Sequential([
    ##tf.keras.layers.Dense(100, activation="relu"),
    #tf.keras.layers.Dense(100, input_shape=[1], activation="relu",name="input_layr"),
    tf.keras.layers.Dense(30, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name="output_layer")
    ], name="model_1")

#Compile the model

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

##Fitting the model

model.fit(X_train, y_train, epochs=100, verbose=0)


##Visualization
model.summary()

##Lets visualize it with tensorflow
import pydot
import graphviz
from tensorflow.keras.utils import plot_model

plot_model(model=model, show_shapes=True)


##Next
##Visualizing our model's predictions


y_pred = model.predict(X_test)

y_pred 

y_test

#Visualizing our data

plt.figure(figsize=(8, 5))
#Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training data")
#Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Test data")
#Show legend
plt.legend();


##Let's create a plotting function

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     preds=y_pred):
    
    ##Plots training data, test dtat and compares predictions to ground truth labels
    plt.figure(figsize=(8,5))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="traning data")
    #Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", label="testing data")
    #Plot model's predictions in red 
    plt.scatter(test_data, preds, c="r", label="predictions")
    plt.legend();
    
    
plot_predictions()

#Next
##Evaluating our model's predictions with regression evaluation metrics

##SInce we're working on regressinon, two of the main metrics:
    #MAE = mean absolute error, "on average, how wrong is each of my model's predictions"
    #MSE = mean square error, "square the average errors"
    



model.evaluate(X_test, y_test)

#Calculate the absolute error
y_pred

y_test


#Converting y_pred to tensor

y_pred = tf.constant(y_pred)


#y_pred is = tf.Tensor: shape=(10, 1), dtype=float32, lets make the shape to be shape=(10, )

tf.squeeze(y_pred)


#Calculate the absolute error

#Calculate tmodel_1 evaluation functions

mae = mae(y_test, y_pred)

 
  ##Calculate the mean square error
mse = mse(y_test, y_pred)
mae, mse


##Make some functions to reuse MAE and MSE

def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_test,
                                         y_pred=tf.squeeze(y_pred)).numpy()


def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_test,
                                         y_pred=tf.squeeze(y_pred)).numpy()





##Let's do 3 modelling experimentrs:
    
#1. model_1 = samw as the original model, 1 layer, trained for 100 epochs

#2. model_2 = 2 layers, trained for 100 epochs

#3. model_3 = 2 layers, trained for 500 epochs
    


#MODEL MODEL_1

tf.random.set_seed(42)
    

##Create a model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1],name="output_layer")
    ], name="model_1")

#Compile the model

model_1.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

##Fitting the model

model_1.fit(X_train, y_train, epochs=100)

y_pred_1 = model_1.predict(X_test)

y_pred_1
    
    
    
plot_predictions(preds=y_pred_1)
    
#Calculate tmodel_1 evaluation functions

mae_1 = mae(y_test, y_pred_1)

 
  ##Calculate the mean square error
mse_1 = mse(y_test, y_pred_1)
mae_1, mse_1




#2. model_2 =  2 dense layers, trained for 100 epochs
tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10,  input_shape=[1]),
    tf.keras.layers.Dense(1)
    ])


#Compile the model

model_2.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mse"])

model_2.fit(X_train, y_train, epochs=100)

y_pred_2 = model_2.predict(X_test)

y_pred_2
    
    
    
plot_predictions(preds=y_pred_2)
    
#Calculate tmodel_1 evaluation functions

mae_2 = mae(y_test, y_pred_2)

 
  ##Calculate the mean square error
mse_2 = mse(y_test, y_pred_2)
mae_2, mse_2






#3. model_3 = 2 layers, trained for 500 epochs
tf.random.set_seed(42)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10,  input_shape=[1]),
    tf.keras.layers.Dense(1)
    ])


#Compile the model

model_3.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model_3.fit(X_train, y_train, epochs=500)

y_pred_3 = model_3.predict(X_test)

y_pred_3
    
    
y_test
plot_predictions(preds=y_pred_3)
    
#Calculate tmodel_1 evaluation functions

mae_3 = mae(y_test, y_pred_3)

 
  ##Calculate the mean square error
mse_3 = mse(y_test, y_pred_3)
mae_3, mse_3

##Comparing my predictions
plot_predictions(preds=y_pred)
plot_predictions(preds=y_pred_1)
plot_predictions(preds=y_pred_2)
plot_predictions(preds=y_pred_3)


##Comparing my predictions

import pandas as pd

model_results = [["model", mae, mse],
                ["model_1", mae_1, mse_1],
                ["model_2", mae_2, mse_2],
                ["model_3", mae_3, mse_3]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])

all_results

model.summary()
model_1.summary()
model_2.summary()
model_3.summary()





######################################################
Saving our models

#There are two main formats we can save our model
#1. The SavedModel format
#2. The HDF5 format

## Save model using the SavedModel format

model_2.save("best_model_SavedModel_format")



## Save model using the HDF5 format
model_2.save("best_model_HDF5_formate.h5")



######################################################
Loading our saved models


##Load in the SavedModel format model

loaded_savedMode_format_pred = tf.keras.models.load_model("best_model_SavedModel_format")

loaded_savedMode_format_pred.summary()
model_2.summary()


##Compare model_1 pred with savedModel format pred
model_2_pred = model_2.predict(X_test)

loaded_savedMode_format_pred = loaded_savedMode_format.predict(X_test)




tf.squeeze(loaded_savedMode_format_pred) == tf.squeeze(model_2_pred)


##Load in the .h5 format model

loaded_h5_format = tf.keras.models.load_model("best_model_HDF5_formate.h5")

loaded_h5_format_pred = loaded_h5_format.predict(X_test)

loaded_h5_format.summary()

model_2_pred = model_2.predict(X_test)


model_2.summary()


##Downlaoding  file from google colab
from google.cola import files

files.downlaoding("best_model_HDF5_formate.h5")

























###Putting everything together
##search medical cost dataset




import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


insurance = pd.read_csv("insurance.csv")


#Lets one hot encode our data
insurance_one_hot = pd.get_dummies(insurance)

insurance_one_hot


##Creating x and y values
X = insurance_one_hot.drop("charges", axis=1)
X.head()

len(X)

y = insurance_one_hot["charges"]
y


##creating a training and testing dataset
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=42)

len(X_train), len(y_train), len(X_test), len(y_test)

"""""
####################Not working
#Visualizing our data

plt.figure(figsize=(8, 5))
#Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training data")
#Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Test data")
#Show legend
plt.legend();

########################
##Let's create a plotting function

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     preds=y_pred):
    
    ##Plots training data, test dtat and compares predictions to ground truth labels
    plt.figure(figsize=(8,5))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="traning data")
    #Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", label="testing data")
    #Plot model's predictions in red 
    plt.scatter(test_data, preds, c="r", label="predictions")
    plt.legend();
    
    
plot_predictions()

"""""

###Build a neural network
tf.random.set_seed(42)


insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
    ])



##Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=["mae"])


###Fitting
insurance_model.fit(X_train, y_train, epochs=1000)
    
    
    
##Check the results of the insurance model on the test data

insurance_model.evaluate(X_test, y_test)


y_pred = insurance_model.predict(tf.square(X_test))

y_pred











###Build a neural network using another method_2(assignment)
tf.random.set_seed(42)


insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(80, activation="relu"),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
    ])


##Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

###Fitting
history_1 = insurance_model_2.fit(X_train, y_train, epochs=500)
    
 

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_1.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")



    
##Check the results of the insurance model on the test data

insurance_model.evaluate(X_test, y_test)


y_pred = insurance_model.predict(tf.square(X_test))

y_pred





######################################corrections

###Build a neural network using another method(from lesson)
tf.random.set_seed(42)


insurance_model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])



##Compile the model
insurance_model_1.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"])


###Fitting
insurance_model_1.fit(X_train, y_train, epochs=100, verbose=1)
    
    
    
##Check the results of the insurance model on the test data

insurance_model_1.evaluate(X_test, y_test)


y_pred = insurance_model.predict(tf.square(X_test))

y_pred





###Build a neural network using another method_3
tf.random.set_seed(42)


insurance_model_3 = tf.keras.Sequential([
    #tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])



##Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=["mae"])


###Fitting
history = insurance_model_3.fit(X_train, y_train, epochs=300)
    
##Evaluate the model
insurance_model_3.evaluate(X_test, y_test)


##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")
    



##Check the results of the insurance 

insurance_model.evaluate(X_test, y_test)
insurance_model_1.evaluate(X_test, y_test)
insurance_model_2.evaluate(X_test, y_test)
insurance_model_3.evaluate(X_test, y_test)


##Best model so far
insurance_model_2.summary()

y_pred = insurance_model.predict(tf.square(X_test))

y_pred







#####################################################################





##Another way of preprocessing our data
###Preprocessing data(normalization and stardardization)

X["age"].plot(kind="hist")
X["bmi"].plot(kind="hist")
X["children"].value_counts()

###In terms of scaling data, neural networks prefer normalization

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



insurance = pd.read_csv("insurance.csv")

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
    


###create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )

    
##Creating x and y values
X = insurance.drop("charges", axis=1)
X.head()

len(X)

y = insurance["charges"]
y


##creating a training and testing dataset
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=42)

len(X_train), len(y_train), len(X_test), len(y_test)

##Fitting the column transformer to our training data
ct.fit(X_train)

##Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)


#Lets see how our data is looking now
X_train.loc[0]
X_train_normal.loc[0]
    
X_train.shape, X_train_normal.shape

##Beautiful...our data has been norms and onehotencoded. Now let's build our neural network

##Building our neural network....Assignment


###Build a neural network using another method_2(assignment)
tf.random.set_seed(42)


insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
    ])


##Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

###Fitting
history = insurance_model.fit(X_train_normal, y_train, epochs=200)

##Evaluate the model
insurance_model.evaluate(X_test_normal, y_test)
 

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_1.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")



































!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 


























































