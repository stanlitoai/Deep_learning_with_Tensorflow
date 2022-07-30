
"""

     TIME SERIES FORECASTING FUNDAMENTALS WITH TENSORFLOW + MILESTONE POJECT 3: BitPredict


##worst tech predictions

  WHAT WE'RE GOING TO COVER
  
1. Downloading and formatting time series data (the historical price of Bitcoin)
2. Writing a preprocessing function to prepare our time series data
3. Setting up multiple time series modelling experiments
4. Building a multivaraiate model to take in multivariate time series data
5. Replicating the N-BEATS algorithm using TensorFlow
6. Making forecasts with prediction intervals
7. Demonstrating why time series forecasting can be BS with the turkey problem


"""



#!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv

#!ls
import pandas as pd


df = pd.read_csv("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=["Date"])

df.head()



df.plot.hist()



"""

    TYPES OF TIMES SERIES
    
TREND: Time series has a clear long-term increase or decrease
 (may or may not be linear)


SEASONAL : Time series affected by seasonal factors such as time of year. 
(eg: increased sales of year) or day of the week


CYCLIC: Time series shows risees and falls over an unfixed period, these tend to be 
longer/more variable than seasonal patterns


"""
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)":"Price"})


bitcoin_prices.head()


import  matplotlib.pyplot as plt

bitcoin_prices.plot(figsize=(8, 5))
plt.ylabel("BTC Price")
plt.title("Price of Bitcoin from 1 oct 2013 to 18 May 2021", fontsize=16)
plt.legend(fontsize=14);


# Importing time series data with python CSV module

import csv
from datetime import datetime


timesteps = []
btc_price = []

with open("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader) #skip first line (this gets rid of the column titles)
    for line in csv_reader:
        timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))
        btc_price.append(float(line[2])) #get the closing price as float




# View first 10 of each
timesteps[:10], btc_price[:10]



# Plot CSV

import  matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(timesteps, btc_price)
plt.ylabel("BTC Price")
plt.xlabel("Date")
plt.title("Price of Bitcoin from 1 oct 2013 to 18 May 2021", fontsize=16)
plt.legend(fontsize=14);


# Format data part 1: Create train and test sets of our time series data



timesteps =  bitcoin_prices.index.to_numpy()

prices =  bitcoin_prices["Price"].to_numpy()


timesteps[:10]

prices[:10]


# Creating train and test set(the wrong way)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(timesteps,
                                                    prices,
                                                    test_size=0.2,
                                                    random_state=42)




X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Lets plot wrong train and test splits

plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.show();







# Creating train and test set(the right way)

split_size = int(0.8 * len(prices))


X_train, y_train = timesteps[:split_size], prices[:split_size]


X_test, y_test = timesteps[split_size:], prices[split_size:]




X_train.shape, X_test.shape, y_train.shape, y_test.shape



# Lets plot right train and test splits

plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.show();





# Create a function to plot time series data

def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """
    Plots timesteps (a series of points in time ) against values (a series of values acros time)

    Parameters
    ----------
    timesteps : TYPE
        Array of timestep value
    values : TYPE
        array of value across time
    format : TYPE, optional
        style of plot. The default is ".".
    start : TYPE, optional
        where to start the plot. The default is 0. end=None.
    label : TYPE, optional
        label to show on plots. The default is None.

    Returns
    -------
    None.

    """
    # Plot the series
    
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)




plt.figure(figsize=(9, 6))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")



#%%%%STEPS

"""
\
Terms to be familiar with:
    * Horizon : number of timesteps into the future we're goiung to predict

    * Window size: number of timesteps we're goung to use to predict HORIZON    



   EXPERIMENTS WE'RE RUNNING
   
0. model_0: Naive model(baseline) 
1. model_1: Dense model, horizon =1, window = 7
2. model_2: Same as model_1, horizon =1, window = 30
3. model_3: Same as model_1, horizon =7, window = 30
4. model_4: Conv1D
5. model_5: LSTM
6. model_6: same as model_1 (but with multivariate data)
7. model_7: N-BEATS algorithm
8. model_8: Ensemble (multiple models stacked together)
9. model_9: Future prediction model
10. model_10: same as model_1 (but with turkey data introduced)


M4 competiton

"""


#%%% Model_0
"""
0. model_0: Naive model(baseline)


"""
 #Create a naive forecast
 
naive_forecast = y_test[:-1]


plt.figure(figsize=(9, 6))
#plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-",
                 label="Naive Forecast")



df.info()


"""

Evaluating a time series model

Let's look into some evaluation metrics for time series forecasting.

What are we doing?

We're predicting a number, so that means we have a form of a regression problem.

Because we're working on a regression problem, we'll need some regression-like metrics.

A few common regression metrics (which can also be used for time series forecasting):

    MAE - mean absolute error
    MSE - mean squared error
    RMSE - root mean square error
    MAPE/sMAPE - (symmetric) mean absolute percentage error
    MASE - mean absolute scaled error

For all of the above metrics, lower is better, for example, an MAE of 0 that is better than an MAE of 100.

The main thing we're evaluating here is: how do our model's forecasts (y_pred) compare against the actual values (y_true or ground truth values)?

    📖 Resource: For a deeper dive on the various kinds of time series forecasting methods see Forecasting: Principles and Practice chapter 5.8

"""
import tensorflow as tf

# MASE implementation
def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true-y_pred))

  # Find MAE of naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shift of 1)

  return mae / mae_naive_no_season





# Test MASE (this value should = 1 or be very close to 1 with the naive forecast)
mean_absolute_scaled_error(y_true=y_test[1:], y_pred=naive_forecast).numpy()








# Create a function to take in model predictions and truth values and return evaluation metrics
def evaluate_preds(y_true, y_pred):
  # Make sure float32 datatype (for metric calculations)
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various evaluation metrics 
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}




naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)

naive_results





naive_forecast[-10:]


tf.reduce_mean(y_test)


tf.reduce_min(y_test), tf.reduce_max(y_test)




"""
Other models you can use for baselines and for actual forecasts

In this notebook, we're focused on TensorFlow and deep learning models.

However there are plenty of other styles of time series forecasting models you may want to experiment with: https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/#other-kinds-of-time-series-forecasting-models-which-can-be-used-for-baselines-and-actual-forecasts
Format Data Part 2: Windowing our dataset

Why do we window?

We window our time series dataset to turn our data into a supervised learning problem.





Windowing for one week
[0, 1, 2, 3, 4, 5, 6] -> [7]
[1, 2, 3, 4, 5, 6, 7] -> [8]
[2, 3, 4, 5, 6, 7, 8] -> [9]

"""

# What we want to do with our Bitcoin data
print(f"We want to use: {btc_price[:7]} to predict this: {btc_price[7]}")



# Let's setup global variables for window and horizon size
HORIZON = 1 # predict next 1 day
WINDOW_SIZE = 7 # use the past week of Bitcoin data to make the prediction





# Create function to label windowed data
def get_labelled_windows(x, horizon=HORIZON):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1
  Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
  """
  return x[:, :-horizon], x[:, -horizon:]






# Test out the window labelling function
test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(8), 
                                                              axis=0))
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")


"""
We've got a way to label our windowed data.

However, this only works on a small scale.

We need a way to do the above across our entire time series.

We could do this with Python for loops, however, for large time series,
 that'd be quite lsow.

To speed things up, we'll leverage NumPy's array indexing 
- https://numpy.org/doc/stable/reference/arrays.indexing.html.

Our function will:

1. reate a window step of specific window size (e.g. [0, 1, 2, 3, 4, 5, 6])
2. Use NumPy indexing to create a 2D array of multiple window steps, for example: 





[[0, 1, 2, 3, 4, 5, 6],
[1, 2, 3, 4, 5, 6, 7],
[2, 3, 4, 5, 6, 7, 8]]


Uses the 2D array of multiple window steps (from 2.) to index on a target series (e.g. the historical price of Bitcoin)
Uses our get_labelled_windows() function we created above to turn the window steps into windows with a specified horizon

📖 Resource: The function we're about to create has been adapted from the 
following article: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
"""

import numpy as np

# Create function to view NumPy arrays as windows
def make_windows(x, window_size=WINDOW_SIZE, horizon=HORIZON):
  """
  Turns a 1D array into a 2D array of sequential labelled windows of 
  window_size with horizon size labels.
  """
  # 1. Create a window of specific window_size (add the horzion on the end for labelling later)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  print(f"Window indexes:\n {window_indexes, window_indexes.shape}")

  # 3. Index on the target array (a time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]
  print(f"windowed_array:\n  {windowed_array}")

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
  return windows, labels






full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)



# View the first 3 windows/ labels
for i in range(3):
    print(f"Window: {full_windows[i]} --> Labels: {full_labels[i]}")



# View the last 3 windows/ labels
for i in range(3):
    print(f"Window: {full_windows[i-3]} --> Labels: {full_labels[i-3]}")



full_windows[:5], full_labels[:5]



# Turning windows into training and testing set

def make_train_test_splits(windows, labels, test_split=0.2):
    """
    

    Splits matching pairs of windows and labels into train and test splits
    ----------
    windows : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    test_split : TYPE, optional
        DESCRIPTION. The default is 0.2.

    Returns
    -------
    None.

    """
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    
    return train_windows, test_windows, train_labels, test_labels





#create train and test windows

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

len(train_windows), len(test_windows), len(train_labels), len(test_labels)


"""
int(len(full_windows) * (1 - 0.2))
"""





# Check to see if train labels are the same (before and after window split)

np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:])



"""
# Making a modelling checkpoint callback

Because our model's performance will fluctuate from experiment to experiment, 
we're going to write a model checpoint so we can compare apples to apples.


More specifically, we want to compare each of our model's best performance against 
the other model's best preformance.


"""
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Create a function to implement  a ModelCheckpoint callback with a specific filename

def create_model_checkpoint(model_name,save_path="model_experiments"):
    return ModelCheckpoint(filepath=os.path.join(save_path, model_name), 
                           verbose=0,
                           save_best_only=True)




#%%% Model_1
"""

1. model_1: Dense model, horizon =1, window = 7

Our first deep model is going to be a simple dense model:
    * A single dense layer with 128 hidden units and ReLU
    * An output layer with linear activation (no activation)
    * Adam optimization and MAE loss function
    * Batch size of 128 (previously we've used 32)
    * 100 epochs
    
    MACHINE LEARNING PRACTITIONER


"""


import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)



model_1 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear")
    ], name="model_1_dense")





model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae","mse"])


model_1_history = model_1.fit(train_windows,
                              train_labels,
                              epochs=100,
                              verbose=1,
                              batch_size=128,
                              validation_data=(test_windows, test_labels),
                              callbacks=[create_model_checkpoint(model_1.name)])





# Evaluate model on test data

model_1.evaluate(test_windows, test_labels)


# Load in saved best performing model_1 and evaluate it on test data

model_1 = tf.keras.models.load_model("model_experiments/model_1_dense/")
model_1.evaluate(test_windows, test_labels)


naive_results


"""
# Making forecasts with a model (on the test dataset)

To make "forecasts" on the test dataset (note : these wont be actual forecast, they're only psuedo forecast because actual forecasts 
                                         are into the future),
lets write a function to:
    1. Take in a train model
    2. Takes in some input data (same kind of data the model was trained on)
    3. Passes the input data to the model's prediction method
    4. Returns the predictions
    
"""


def make_preds(model, input_data):
    """
    Uses model to make a prediction input_data

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    input_data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)




# Make predictions using model_1 on the  tet dataset and view results

model_1_preds = make_preds(model_1,
                           input_data=test_windows)


len(model_1_preds), model_1_preds[:10]


test_labels[:10]


# Evaluate preds

model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels, axis=1), y_pred=model_1_preds)

model_1_results
naive_results



# Lets plot our model 1 predictions

offset = 300

plt.figure(figsize=(10, 7))
#Account for the tet_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=test_labels[:, 0],
                 start=offset,
                 label="Test Data")


plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=model_1_preds,
                 format="-",
                 start=offset,
                 label="Model_1 preds")








######
#Assignment on Auto Regression


#%%% Model_2
"""
2. model_2: Same as model_1, horizon =1, window = 30
"""



# What we want to do with our Bitcoin data
print(f"We want to use: {btc_price[:30]} to predict this: {btc_price[30]}")



# Let's setup global variables for window and horizon size
HORIZON = 1 # predict next 1 day
WINDOW_SIZE = 30 # use the past week of Bitcoin data to make the prediction


def x():
    return x






# Test out the window labelling function
test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(31), 
                                                              axis=0))
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")


"""
We've got a way to label our windowed data.

However, this only works on a small scale.

We need a way to do the above across our entire time series.

"""


full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)



# View the first 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i]} --> Labels: {full_labels[i]}")



# View the last 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i-3]} --> Labels: {full_labels[i-3]}")



full_windows[:5], full_labels[:5]





#create train and test windows

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

len(train_windows), len(test_windows), len(train_labels), len(test_labels)






# Check to see if train labels are the same (before and after window split)

np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:])



"""
1. model_2: Dense model, horizon =1, window = 30

Our first deep model is going to be a simple dense model:
    * A single dense layer with 128 hidden units and ReLU
    * An output layer with linear activation (no activation)
    * Adam optimization and MAE loss function
    * Batch size of 128 (previously we've used 32)
    * 100 epochs
    
    MACHINE LEARNING PRACTITIONER


"""


import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)



model_2 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear")
    ], name="model_2_dense")





model_2.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae","mse"])


model_2_history = model_2.fit(train_windows,
                              train_labels,
                              epochs=100,
                              verbose=0,
                              batch_size=128,
                              validation_data=(test_windows, test_labels),
                              callbacks=[create_model_checkpoint(model_2.name)])





# Evaluate model on test data

model_2.evaluate(test_windows, test_labels)


# Load in saved best performing model_1 and evaluate it on test data

model_2 = tf.keras.models.load_model("model_experiments/model_2_dense/")
model_2.evaluate(test_windows, test_labels)


naive_results


"""
# Making forecasts with a model (on the test dataset)

To make "forecasts" on the test dataset (note : these wont be actual forecast, they're only psuedo forecast because actual forecasts 
                                         are into the future),
lets write a function to:
    1. Take in a train model
    2. Takes in some input data (same kind of data the model was trained on)
    3. Passes the input data to the model's prediction method
    4. Returns the predictions
    
"""




# Make predictions using model_1 on the  tet dataset and view results

model_2_preds = make_preds(model_2,
                           input_data=test_windows)


len(model_2_preds), model_2_preds[:10]


test_labels[:10]


# Evaluate preds

model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels, axis=1), y_pred=model_2_preds)

model_2_results
naive_results



# Lets plot our model 1 predictions

offset = 300

plt.figure(figsize=(8, 6))
#Account for the tet_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=test_labels[:, 0],
                 start=offset,
                 label="Test Data")


plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=model_2_preds,
                 format="-",
                 start=offset,
                 label="Model_2 preds")






#%% Model_3

"""

3. model_3: Same as model_1, horizon =7, window = 30

"""



# Let's setup global variables for window and horizon size
HORIZON = 7 # predict next 1 day
WINDOW_SIZE = 30 # use the past week of Bitcoin data to make the prediction








# Test out the window labelling function
test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(31), 
                                                              axis=0))
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")


"""
We've got a way to label our windowed data.

However, this only works on a small scale.

We need a way to do the above across our entire time series.

"""


full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)



# View the first 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i]} --> Labels: {full_labels[i]}")



# View the last 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i-3]} --> Labels: {full_labels[i-3]}")



full_windows[:5], full_labels[:5]





#create train and test windows

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

len(train_windows), len(test_windows), len(train_labels), len(test_labels)






# Check to see if train labels are the same (before and after window split)

np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:])


def model3():
    return x



"""
1. model_2: Dense model, horizon =1, window = 30

Our first deep model is going to be a simple dense model:
    * A single dense layer with 128 hidden units and ReLU
    * An output layer with linear activation (no activation)
    * Adam optimization and MAE loss function
    * Batch size of 128 (previously we've used 32)
    * 100 epochs
    
    MACHINE LEARNING PRACTITIONER


"""


import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)



model_3 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
    ], name="model_3_dense")





model_3.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae","mse"])


model_3_history = model_3.fit(train_windows,
                              train_labels,
                              epochs=100,
                              verbose=1,
                              batch_size=128,
                              validation_data=(test_windows, test_labels),
                              callbacks=[create_model_checkpoint(model_3.name)])





# Evaluate model on test data

model_3.evaluate(test_windows, test_labels)


# Load in saved best performing model_1 and evaluate it on test data

model_3 = tf.keras.models.load_model("model_experiments/model_3_dense/")
model_3.evaluate(test_windows, test_labels)


naive_results


"""
# Making forecasts with a model (on the test dataset)

To make "forecasts" on the test dataset (note : these wont be actual forecast, they're only psuedo forecast because actual forecasts 
                                         are into the future),
lets write a function to:
    1. Take in a train model
    2. Takes in some input data (same kind of data the model was trained on)
    3. Passes the input data to the model's prediction method
    4. Returns the predictions
    
"""




# Make predictions using model_1 on the  tet dataset and view results

model_3_preds = make_preds(model_3,
                           input_data=test_windows)


len(model_3_preds), model_3_preds[:5]


test_labels[:10]


# make our evaluation function work for larger horizons



# Create a function to take in model predictions and truth values and return evaluation metrics
def evaluate_preds(y_true, y_pred):
  # Make sure float32 datatype (for metric calculations)
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various evaluation metrics 
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)
  
  
  # Account for different sized metrics (for longer horizons, we want to reduce metrics to a single dim)
  if mae.ndim > 0:
      mae = tf.reduce_mean(mae)
      mse = tf.reduce_mean(mse)
      rmse = tf.reduce_mean(rmse)
      mape = tf.reduce_mean(mape)
      mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}







model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_3_preds)

model_3_results
naive_results


# Lets plot our model 1 predictions

offset = 300

plt.figure(figsize=(8, 6))
#Account for the tet_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=test_labels[:, 0],
                 start=offset,
                 label="Test Data")


plot_time_series(timesteps=X_test[-len(test_windows):],
                 values=tf.reduce_mean(model_3_preds, axis=1),
                 format="-",
                 start=offset,
                 label="Model_3 preds")



## Which of our model is performing the best so far

pd.DataFrame({"naive": naive_results["mae"],
              "horizon_1_window_7": model_1_results["mae"],
              "horizon_1_window_30": model_2_results["mae"],
              "horizon_7_window_30": model_3_results["mae"]}, index=["mae"]).plot(figsize=(8,5), kind="bar")




#%% Model_4

"""
4. model_4: Conv1D

"""

HORIZON = 1
WINDOW_SIZE = 7




full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)



# View the first 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i]} --> Labels: {full_labels[i]}")



# View the last 3 windows/ labels
for i in range(1):
    print(f"Window: {full_windows[i-3]} --> Labels: {full_labels[i-3]}")



full_windows[:5], full_labels[:5]





#create train and test windows

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

len(train_windows), len(test_windows), len(train_labels), len(test_labels)




#tO USE the Conv1D layer, we need an input shape of:(batch_size, timesteps, input_dim)....
# However our data isntin that shape yet



def model_4():
    
    return model_4


#Check data input shape
train_windows[0].shape



# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim) 
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")





tf.random.set_seed(42)

# Create model
model_4 = tf.keras.Sequential([
  # Create Lambda layer to reshape inputs, without this layer, the model will error
  layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), # resize the inputs to adjust for window size / Conv1D 3D input requirements
  layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
  layers.Dense(HORIZON)
], name="model_4_conv1D")

# Compile model
model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())


#model_4.summary()

# Fit model
model_4.fit(train_windows,
            train_labels,
            batch_size=128, 
            epochs=100,
            verbose=0,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_4.name)])





# Load in best performing Conv1D model and evaluate it on the test data
model_4 = tf.keras.models.load_model("model_experiments/model_4_conv1D")
model_4.evaluate(test_windows, test_labels)





# Make predictions
model_4_preds = make_preds(model_4, test_windows)
model_4_preds[:10]





# Evaluate predictions
model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_4_preds)

model_4_results




## Which of our model is performing the best so far

pd.DataFrame({"naive": naive_results["mae"],
              "horizon_1_window_7": model_1_results["mae"],
              "horizon_1_window_30": model_2_results["mae"],
              "horizon_7_window_30": model_3_results["mae"],
              "horizon_1_window_7(Conv1D)": model_4_results["mae"]}, 
             index=["mae"]).plot(figsize=(8,5), kind="bar")






#%% Model_5

"""
5. model_5: LSTM

"""
# Building an RNN model with our previous dataset



tf.random.set_seed(42)

# Let's build an LSTM model with the Functional API
inputs = layers.Input(shape=(WINDOW_SIZE))
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
# print(x.shape)
x = layers.LSTM(128, activation="relu", return_sequences=True)(x) # this layer will error if the inputs are not the right shape
x = layers.LSTM(128, activation="relu")(x) # using the tanh loss function results in a massive error
# print(x.shape)
# Add another optional dense layer (you could add more of these to see if they improve model performance)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(HORIZON)(x)
model_5 = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_lstm")

# Compile model
model_5.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# Seems when saving the model several warnings are appearing: https://github.com/tensorflow/tensorflow/issues/47554 
model_5.fit(train_windows,
            train_labels,
            epochs=100,
            verbose=0,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_5.name)])



model_5.evaluate(test_windows, test_labels)




# Load in best performing Conv1D model and evaluate it on the test data
model_5 = tf.keras.models.load_model("model_experiments/model_5_lstm")
model_5.evaluate(test_windows, test_labels)





# Make predictions
model_5_preds = make_preds(model_5, test_windows)
model_5_preds[:10]





# Evaluate predictions
model_5_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=model_5_preds)

model_5_results





## Which of our model is performing the best so far

pd.DataFrame({"naive": naive_results["mae"],
              "horizon_1_window_7": model_1_results["mae"],
              "horizon_1_window_30": model_2_results["mae"],
              "horizon_7_window_30": model_3_results["mae"],
              "horizon_1_window_7(Conv1D)": model_4_results["mae"],
              "horizon_1_window_7(LSTM)": model_5_results["mae"]}, 
             index=["mae"]).plot(figsize=(8,5), kind="bar")










#%% Model_6

"""
6. model_6: same as model_1 (but with multivariate data)


    🔑 Note: I'm putting this here again as a reminder that because neural networks are such powerful algorithms, they can be used for almost any problem, however,
    that doesn't mean they'll achieve performant or usable results. You're probably starting to clue onto this now.

Make a multivariate time series


So far all of our models have barely kept up with the naïve forecast.

And so far all of them have been trained on a single variable (also called univariate time series): the historical price of Bitcoin.

If predicting the price of Bitcoin using the price of Bitcoin hasn't worked out very well, maybe giving our model more information may help.

More information is a vague term because we could actually feed almost anything to our model(s) and they would still try to find patterns.

For example, we could use the historical price of Bitcoin as well as anyone with the name Daniel Bourke Tweeted on that day to predict the future price of Bitcoin.

But would this help?

Porbably not.

What would be better is if we passed our model something related to Bitcoin (again, this is quite vauge, since in an open system like a market, you could argue everything is related).

This will be different for almost every time series you work on but in our case, we could try to see if the Bitcoin block reward size adds any predictive power to our model(s).

What is the Bitcoin block reward size?

The Bitcoin block reward size is the number of Bitcoin someone receives from mining a Bitcoin block.

At its inception, the Bitcoin block reward size was 50.

But every four years or so, the Bitcoin block reward halves.

For example, the block reward size went from 50 (starting January 2009) to 25 on November 28 2012.

Let's encode this information into our time series data and see if it helps a model's performance.

    🔑 Note: Adding an extra feature to our dataset such as the Bitcoin block reward size will take our data from univariate (only the historical price of Bitcoin) to multivariate (the price of Bitcoin as well as the block reward size).




"""
# Let's make a multivariate time series
bitcoin_prices.head()


#Lets add the bitcoin halving events to our dataset

# Block reward values
block_reward_1 = 50 # 3 January 2009 (2009-01-03) - this block reward isn't in our dataset (it starts from 01 October 2013)
block_reward_2 = 25 # 28 November 2012 
block_reward_3 = 12.5 # 9 July 2016
block_reward_4 = 6.25 # 11 May 2020

# Block reward dates (datetime form of the above date stamps)
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-18")






# Get date indexes for when to add in different block dates
block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
block_reward_2_days, block_reward_3_days



# Add block_reward column
bitcoin_prices_block = bitcoin_prices.copy()
bitcoin_prices_block["block_reward"] = None

# Set values of block_reward column (it's the last column hence -1 indexing on iloc)
bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4
bitcoin_prices_block.head()



# Plot the block reward/price over time
# Note: Because of the different scales of our values we'll scale them to be between 0 and 1.
from sklearn.preprocessing import minmax_scale
scaled_price_block_df = pd.DataFrame(minmax_scale(bitcoin_prices_block[["Price", "block_reward"]]), # we need to scale the data first
                                     columns=bitcoin_prices_block.columns,
                                     index=bitcoin_prices_block.index)
scaled_price_block_df.plot(figsize=(10, 5));




"""


When we scale the block reward and the Bitcoin price, we can see the price goes up as the block reward goes down, perhaps this information will be helpful to our model's performance.
Making a windowed dataset with pandas

Previously, we used some custom made functions to window our univariate time series.

However, since we've just added another variable to our dataset, these functions won't work.

Not to worry though. Since our data is in a pandas DataFrame, we can leverage the pandas.DataFrame.shift() method to create a windowed multivariate time series.

The shift() method offsets an index by a specified number of periods.

Let's see it in action.
"""



# Setup dataset hyperparameters
HORIZON = 1
WINDOW_SIZE = 7



# Make a copy of the Bitcoin historical data with block reward feature
bitcoin_prices_windowed = bitcoin_prices_block.copy()

# Add windowed columns
for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
  bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)
bitcoin_prices_windowed.head(10)




"""


Now that we've got a windowed dataset, let's separate features (X) from labels (y).

Remember in our windowed dataset, we're trying to use the previous WINDOW_SIZE steps to predict HORIZON steps.

Window for a week (7) to predict a horizon of 1 (multivariate time series)
WINDOW_SIZE & block_reward -> HORIZON

[0, 1, 2, 3, 4, 5, 6, block_reward] -> [7]
[1, 2, 3, 4, 5, 6, 7, block_reward] -> [8]
[2, 3, 4, 5, 6, 7, 8, block_reward] -> [9]

We'll also remove the NaN values using pandas dropna() method, this equivalent to starting our windowing function at sample 0 (the first sample) + WINDOW_SIZE.



"""



# Let's create X & y, remove the NaN's and convert to float32 to prevent TensorFlow errors 
X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32) 
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)
X.head()
y.head()



# Make train and test sets
split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)




tf.random.set_seed(42)

# Make multivariate time series model
model_6 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  # layers.Dense(128, activation="relu"), # adding an extra layer here should lead to beating the naive model
  layers.Dense(HORIZON)
], name="model_6_dense_multivariate")

# Compile
model_6.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# Fit
model_6.fit(X_train, y_train,
            epochs=100,
            batch_size=128,
            verbose=0, # only print 1 line per epoch
            validation_data=(X_test, y_test),
            callbacks=[create_model_checkpoint(model_name=model_6.name)])





model_6.evaluate(X_test, y_test)


# Make sure best model is loaded and evaluate
model_6 = tf.keras.models.load_model("model_experiments/model_6_dense_multivariate")
model_6.evaluate(X_test, y_test)






# Make predictions
# Make predictions on multivariate data
model_6_preds = tf.squeeze(model_6.predict(X_test))
model_6_preds[:10]



# Evaluate preds
model_6_results = evaluate_preds(y_true=y_test,
                                 y_pred=model_6_preds)
model_6_results





## Which of our model is performing the best so far

pd.DataFrame({"naive": naive_results["mae"],
              "horizon_1_window_7": model_1_results["mae"],
              "horizon_1_window_30": model_2_results["mae"],
              "horizon_7_window_30": model_3_results["mae"],
              "horizon_1_window_7(Conv1D)": model_4_results["mae"],
              "horizon_1_window_7(LSTM)": model_5_results["mae"], 
              "horizon_1_window_7(Multivariate)": model_6_results["mae"]}, 
             index=["mae"]).plot(figsize=(8,5), kind="bar")





















#%% Model_7

"""

7. model_7: N-BEATS algorithm

Lets now try to build the biggest baddest(though maybe not the baddest) time series forecasting model
we've built so far.

"""


# Create NBeatsBlock custom layer 
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs): # the call method is what runs when the layer is called 
    x = inputs 
    for layer in self.hidden: # pass inputs through each hidden layer 
      x = layer(x)
    theta = self.theta_layer(x) 
    # Output the backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast









# Set up dummy NBeatsBlock layer to represent inputs and outputs
dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE, 
                                       theta_size=WINDOW_SIZE+HORIZON, # backcast + forecast 
                                       horizon=HORIZON,
                                       n_neurons=128,
                                       n_layers=4)


# Create dummy inputs (have to be same size as input_size)
dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0) # input shape to the model has to reflect Dense layer input requirements (ndim=2)
dummy_inputs




# Pass dummy inputs to dummy NBeatsBlock layer
backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
# These are the activation outputs of the theta layer (they'll be random due to no training of the model)
print(f"Backcast: {tf.squeeze(backcast.numpy())}")
print(f"Forecast: {tf.squeeze(forecast.numpy())}")


HORIZON = 1
WINDOW_SIZE = 7     


# Create N-BEATS data inputs (N-BEATS works with univariate time series)
bitcoin_prices.head()



# Add windowed columns
bitcoin_prices_nbeats = bitcoin_prices.copy()
for i in range(WINDOW_SIZE):
  bitcoin_prices_nbeats[f"Price+{i+1}"] = bitcoin_prices_nbeats["Price"].shift(periods=i+1)
bitcoin_prices_nbeats.head()



# Make features and labels
X = bitcoin_prices_nbeats.dropna().drop("Price", axis=1)
y = bitcoin_prices_nbeats.dropna()["Price"]

# Make train and test sets
split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)




# Time to make our dataset performant using tf.data API
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# Combine labels and features by zipping together -> (features, labels)
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# Batch and prefetch
BATCH_SIZE = 1024
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset, test_dataset




# Values from N-BEATS paper
N_EPOCHS = 5000
N_NEURONS = 512
N_LAYERS = 4 
N_STACKS = 30 

INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON

INPUT_SIZE, THETA_SIZE




# Make tensors
tensor_1 = tf.range(10) + 10
tensor_2 = tf.range(10)

# Subtract
subtracted = layers.subtract([tensor_1, tensor_2])

# Add
added = layers.add([tensor_1, tensor_2])

# Get outputs
print(f"Input tensors: {tensor_1.numpy()} & {tensor_2.numpy()}")
print(f"Subtracted: {subtracted.numpy()}")
print(f"Added: {added.numpy()}")



"""
Building, compiling and fitting the N-BEATS algorithm

Setup an instance of the N-BEATS block layer using NBeatsBlock (this'll be the initial block used for the network, the rest will be created as part of stacks)
Create an input layer for the N-BEATS stack (we'll be using the Keras Functional API)
Make the initial backcast and forecasts for the model with the layer created in (1)
Use for loop to create stacks of block layers
Use the NBeatsBlock class within the for loop in (4) to create blocks which return backcasts and block-level forecasts
Create the double residual stacking using subtract and add layers
Put the model inputs and outputs together using tf.keras.Model()
Compile the model with MAE loss (the paper uses multiple losses but we'll use MAE to keep it inline with our other models) and Adam optimizer with default setting as per section 5.2 of N-BEATS paper
Fit the N-BEATS model for 5000 epochs and since it's fitting for so many epochs, we'll use a couple of callbacks:
Early Stopping - because the model will be training for so long, we'll stop it early if it stops improving
Reduce LR on Plateau - if a model stops improving, try lowering the learning to reduce the amount it updates its weights each time (take smaller steps towards best performance)

"""


%%time

tf.random.set_seed(42)

# 1. Setup and instance of NBeatsBlock
nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE, 
                                 theta_size=THETA_SIZE,
                                 horizon=HORIZON,
                                 n_neurons=N_NEURONS,
                                 n_layers=N_LAYERS,
                                 name="InitialBlock")

# 2. Create input to stack
stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

# 3. Create initial backcast and forecast input (backwards prediction + horizon prediction)
residuals, forecast = nbeats_block_layer(stack_input)

# 4. Create stacks of block layers
for i, _ in enumerate(range(N_STACKS-1)): # first stack is already created in (3)

  # 5. Use the NBeatsBlock to calculate the backcast as well as the forecast
  backcast, block_forecast = NBeatsBlock(
      input_size=INPUT_SIZE,
      theta_size=THETA_SIZE,
      horizon=HORIZON,
      n_neurons=N_NEURONS,
      n_layers=N_LAYERS,
      name=f"NBeatsBlock_{i}"
  )(residuals) # pass in the residuals

  # 6. Create the double residual stacking
  residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
  forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

# 7. Put the stack model together
model_7 = tf.keras.Model(inputs=stack_input, outputs=forecast, name="model_7_NBEATS")

# 8. Compile model with MAE loss
model_7.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
model_7.fit(train_dataset,
            epochs=N_EPOCHS,
            validation_data=test_dataset,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=200,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                            patience=100,
                                                            verbose=1)])


# Evaluate N-BEats model on the test dataset
model_7.evaluate(test_dataset)




# Make predictions with N-BEATS model
model_7_preds = make_preds(model_7, test_dataset)
model_7_preds[:10]



# Plot the N-BEATS model and inspect the architecture
from tensorflow.keras.utils import plot_model
plot_model(model_7)





#%% Model_8

"""

8. model_8: Ensemble (multiple models stacked together)

"""












def model_6():
    
    return model_6





















#%% Model_9

"""

9. model_9: Future prediction model

"""












def model_9():
    
    return model_9



















#%% Model_10
"""

10. model_10: same as model_1 (but with turkey data introduced)
"""

















