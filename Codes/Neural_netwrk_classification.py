##Introducation to neural network classification with tensorflow

##Creating data to view and fit
import tensorflow as tf
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


#Make 1000 examples
n_samples = 1000

# Create a circles

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)


#Check our features
X

# Check our labels

y

##Lets visualize
import pandas as pd

circles = pd.DataFrame({"XO":X[:,0], "X1":X[:, 1], "Label": y})

circles

##Visualize with plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)



##Input and output
X.shape, y.shape


##Steps in modelling with tensorflow (Assignment)
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(None, 1000, 2)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="softmax")
    ])


##Compile our model

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )



## Fit the model

model.fit(X, y, epochs=100)


## Evaluate the model

model.evaluate()






##########################################################################3

##Steps in modelling with tensorflow with bruk
tf.random.set_seed(42)
    
model_1 = tf.keras.Sequential([
    #tf.keras.Input(shape=(None, 1000, 2)),
    #tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
    ])


##Compile our model

model_1.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["accuracy"]
    )



## Fit the model

model_1.fit(X, y, epochs=200)


## Evaluate the model
model_1.evaluate(X,y)






##########################################################################3

##Steps in modelling with tensorflow with bruk_2
tf.random.set_seed(42)
    
model_2 = tf.keras.Sequential([
    #tf.keras.Input(shape=(None, 1000, 2)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
    ])


##Compile our model

model_2.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["accuracy"]
    )



## Fit the model

model_2.fit(X, y, epochs=100)


## Evaluate the model
model_2.evaluate(X,y)


##Improving the model
##########################################################################3

##Steps in modelling with tensorflow assignment2
tf.random.set_seed(42)
    
model_3 = tf.keras.Sequential([
    #tf.keras.Input(shape=(None, 1000, 2)),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])


##Compile our model

model_3.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )



## Fit the model

history = model_3.fit(X, y, epochs=100)


## Evaluate the model
model_3.evaluate(X,y)

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")


##To visualize our model predictions, let's create a function
#Plot_decision_boundary(), this function will:
    
    ## Take in a trained model, features (x) and  labels (y)
    ## Create a meshgrid of the differengt X values
    ## Make predictions across the meshgrid
    ## Plot the predictions as well as a line between zones(where each unique class falls)
    
import numpy as np

def Plot_decision_boundary(model, X, y):
    
    # Define the axies boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    ## Create X value(we're going to make predictionsa on these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # Stack 2D arrays together
    
    # Make predictions
    y_pred = model.predict(x_in)
    
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("Doing multiclass classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    ##Plot the decision boundary
    #plt.contour(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

Plot_decision_boundary(model = model_3,
                       X=X,
                       y=y)












######################################################################

##Lets see if our model can be used for a regression probs

tf.random.set_seed(42)

X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5) #y = X + 100

X_regression, y_regression



##Split our regression data into training and test sets

X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]

y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]



##Steps in modelling with tensorflow assignment2
tf.random.set_seed(42)
    
model_4 = tf.keras.Sequential([
    #tf.keras.Input(shape=(None, 1000, 2)),
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])


##Compile our model

model_4.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["mae"]
    )



## Fit the model

history = model_4.fit(X_reg_train, y_reg_train, epochs=100)

##Make predictions with our trained model
y_reg_preds = model_4.predict(X_reg_test)
## Evaluate the model
model_4.evaluate(X,y)

import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")



# PLot the model's predictions against our reg data
plt.figure(figsize=(10, 7))
plt.scatter(X_reg_train, y_reg_train, c="b", label="Training data")
plt.scatter(X_reg_test, y_reg_test, c="g", label="Testing data")
plt.scatter(X_reg_test, y_reg_preds, c="r", label="Predictions")
plt.legend();






##The missing piece: Non-linearity

tf.random.set_seed(42)

# Create the model

model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ])


model_5.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
    )

history = model_5.fit(X, y, epochs=100)

##Visualize with plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)


##check the decision boundary for our latest model

####################################################
##Let's try building our first neural network with non linear activation function

tf.random.set_seed(42)

# Create the model

model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)
    ])


model_6.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
    )

history = model_6.fit(X, y, epochs=100)

##Visualize with plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)



####################################################
##Time to replicate the multi-layer neural network from tensorflow playground



tf.random.set_seed(42)

# Create the model

model_7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
    ])


model_7.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
    )

history = model_7.fit(X, y, epochs=1000)

##Visualize with plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)


Plot_decision_boundary(model_7, X, y)





####################################################
##Time to replicate the multi-layer neural network from tensorflow playground
##this time around, we add sigmoid


tf.random.set_seed(42)

# Create the model

model_8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])


model_8.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
    )

history = model_8.fit(X, y, epochs=300)


###Evaluate
model_8.evaluate(X, y)

##Visualize with plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)


Plot_decision_boundary(model_8, X, y)

import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")






###############################################################################

## Create a toy tensor (similiar to the data we pass into our miodels)

A = tf.cast(tf.range(-10, 10), tf.float32)

A

plt.plot(A);

#Let's start by replicating sigmoid - sigmoid(x) = 1/(1 + exp(-x))

def sigmoid(x):
    return 1/(1 + tf.exp(-x))

#Use the sigmoid function on our tensor
sigmoid(A)
plt.plot(sigmoid(A));

##Let's recreate our relu function

def relu(x):
    return tf.maximum(0, x,)

relu(A)

plt.plot(relu(A));





###############################################################
##Going back to our second data set

X,y 

#SPlit into train and test set

X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape



## Creating model

tf.random.set_seed(42)

##Create a model

model_9 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])


model_9.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
    )

history = model_9.fit(X_train, y_train, epochs=25)


###Evaluate
model_9.evaluate(X_test, y_test)


import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.title("Model_9 loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid();


# Plot the decision boundaries for the training and test sets
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
Plot_decision_boundary(model_9, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
Plot_decision_boundary(model_9, X=X_test, y=y_test)
plt.show();



#########################
## Creating model using learning rate callback

tf.random.set_seed(42)

model_10 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


model_10.compile(
    loss="binary_crossentropy",
    optimizer="Adam",
    metrics=["accuracy"]
    )

##Creating a learing rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))




history = model_10.fit(X_train, y_train, epochs=100,
                       callbacks=[lr_scheduler])


###Evaluate
model_10.evaluate(X_test, y_test)



import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.title("Model_10 loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid();

## Plot the learningm rate vs loss

lrs = 1e-4 * (10 ** (tf.range(100)/20))

len(lrs)

plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
plt.legend();
plt.grid();

plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["accuracy"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
plt.legend();
plt.grid();


##Let's try using higher ideal learning rate with the same model


#########################
## Creating model using learning rate callback

tf.random.set_seed(42)

model_11 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


model_11.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
    metrics=["accuracy"]
    )

##Creating a learing rate callback
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))




history_11 = model_11.fit(X_train, y_train, epochs=20)



###Evaluate
model_8.evaluate(X_test, y_test)
model_9.evaluate(X_test, y_test)
model_10.evaluate(X_test, y_test)
model_11.evaluate(X_test, y_test)



import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_11.history).plot()
plt.title("Model_10 loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid();

## Plot the learningm rate vs loss

lrs = 1e-4 * (10 ** (tf.range(100)/20))

len(lrs)

plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
plt.legend();
plt.grid();

plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["accuracy"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
plt.legend();
plt.grid();


# ACcuracy
# Precision
# Recall
# F1-score
#Confusion matrix
# Classification report


#Check the accuracy of our model

loss, accuracy = model_11.evaluate(X_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {(accuracy*100):.2f}%")


##Confusion matrix
from sklearn.metrics import confusion_matrix

# Make predictions
y_preds = model_11.predict(X_test)


## Covert the prediction probabilities to binary format and view the first 10

tf.round(y_preds)[:10]









#################################################################333

# Create confusion matrix

confusion_matrix(y_test,tf.round(y_preds))

## Let's make our confusion matrix beautiful

import itertools

figsize=(8, 8)

#Create the confusion matrix

cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
n_classes = cm.shape[0]

# Let's prettify it
fig, ax = plt.subplots(figsize=figsize)

# Create the matrix plot
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

# Create classes
classes = False

if classes:
    labels = classes
else:
    labels = np.arange(cm.shape[0])


# Label the axis
ax.set(title="Confusion Matrix",
       xlabel="Prediction label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels
       )

## Set threshold for different colors

threshold = (cm.max() + cm.min()) / 2.

# Plot the text on each cell

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[1, j] > threshold else "black",
             size=15)













######################################################################
#Working with a larger example (multiclass classification)

#When you have more than two classes as an option, it's known as MULTI-CLASS CLASSIFICATION

#This means if you have 3 different classses, it's multi-class classification

#It also means if you have 100 diff classes, it's multi-classification

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()


#Show the first training example
print(f"Training sample:\n{train_data[0]}\n")
print(f"Testing label:\n{train_labels[0]}\n")

train_data[0].shape, train_labels[0].shape

#Plot a single sample
import matplotlib.pyplot as plt
plt.imshow(train_data[0]);


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "shirt", "Sneakers", "Bag", "Ankle boot"]

len(class_names)

#Plot asn example image and its label
number = 20
plt.imshow(train_data[number],cmap=plt.cm.binary)
plt.title(class_names[train_labels[number]])


###Plot multiple random images of our data

import random
plt.figure(figsize=(10,10))
for i in range(6):
    ax = plt.subplot(3, 3, i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)




## Building a multi class classificATion model
""""
For our multi-class classification model, we can use a similar 
architecture to our binary classifiers, how ever, we're going to tweak few things
* Input shape = 28 * 28 (the shape of one image)
* Output shape = 10(one per class of clothing)
* Loss function = tf.keras.losses.CategoricalCrosstentropy()
    * if your labels are one_hot encoded, use CategoricalCrosstentropy()
    else:
        use SparseCategoricalCrosstentropy()
* Output layer activation = Softmax


"""""



#########################
## Creating model 

tf.random.set_seed(42)

model_12 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ])


model_12.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )

##Creating a learing rate callback
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))




non_norm_history = model_12.fit(train_data, tf.one_hot(train_labels, depth=10), epochs=100,
                                validation_data=(test_data, tf.one_hot(test_labels, depth=10)))




import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(non_norm_history.history).plot()
plt.title("Model_12 loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid();


##Check the model summary
model_12.summary()

########
train_data.min(), train_data.max()




""""
Neural networks prefer data to be scaled (or normalized), thnis means they like 
to have the numbers in the term between 0 & 1

we can get our training and testing data bte 0 & 1 by dividing by the mx....
"""""
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

train_data_norm.min(), train_data_norm.max()




###Now our data is normalized, lets build model to find patterns

#########################
## Creating model 

tf.random.set_seed(42)

model_13 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ])


model_13.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )

##Creating a learing rate callback
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))




norm_history = model_13.fit(train_data_norm, tf.one_hot(train_labels, depth=10), epochs=10,
                                validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)))




import pandas as pd

##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(non_norm_history.history).plot(title="Non normalized data")
pd.DataFrame(norm_history.history).plot()
plt.title("Model_12 loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid();



#############################################################
###Finding the ideal learning rate


tf.random.set_seed(42)

model_14 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ])


model_14.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
    )

##Creating a learing rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))




find_lr_history = model_14.fit(train_data_norm, 
                               tf.one_hot(train_labels, depth=10), 
                               epochs=40,
                                validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)),
                                callbacks=[lr_scheduler])




## Plot the learning rate decay curve
import numpy as np
import matplotlib.pyplot as plt

lrs = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrs, find_lr_history.history["loss"])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")
plt.grid();



#############################################################
###Lets refit a modekl with the ideal learinng rate


tf.random.set_seed(42)

model_15 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ])


model_15.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
    )

##Creating a learing rate callback
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))




history_15 = model_15.fit(train_data_norm, 
                               tf.one_hot(train_labels, depth=10), 
                               epochs=20,
                                validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)),
                                )



"""""
Evaluating our multi class classification model

* To evaluate our multi class classification model, we could:
    *Evaluate its performance using other classification metrics(such as confusion matrix)
    * Asses some of its predictions (through visualizations)
    
* Improve its results (by training it for longer or changing its architecture)

* Save and export it for use in an application

"""""

tf.one_hot(test_labels[:10], depth=10)
y_preds[0] = model_15.predict(test_data_norm)
tf.round(y_preds)[:10]


#################################################################333

# Create confusion matrix

confusion_matrix(test_data_norm,tf.round(y_preds))

## Let's make our confusion matrix beautiful

import itertools
from sklearn.metrics import confusion_matrix



def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(15, 15), text_size=10):
    

    #Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    
    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    
    # Label the axis
    ax.set(title="Confusion Matrix",
           xlabel="Prediction label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels
           )
    
    #Set X-axis labels to botton
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    #Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    
    ## Set threshold for different colors
    
    threshold = (cm.max() + cm.min()) / 2.
    
    # Plot the text on each cell
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    
    
    
#class_names   

make_confusion_matrix(y_true=test_labels, y_pred=y_preds,
                      classes=class_names)


   
#MAke some predictions with our model
y_preds = model_15.predict(test_data_norm)

y_preds[1], tf.argmax(y_preds[1]), class_names[tf.argmax(y_preds[1])]

# Convert all of the predictions into integers
y_preds = y_preds.argmax(axis=1)


# View the first 10 predictions labels
y_preds[:10]

##Evaluation matrics

model_15.evaluate(test_data_norm, test_labels)

model_9.evaluate(X_test, y_test)



""""
* How about we make a fun littlr function for:
    * PLot a random image
    * Make a prediction on said image
    * Label the plot with the truth label & prediction label
    
"""""
import random

def plot_random_img(model, images, true_labels, classes):
    
    # Set up random int
    i = random.randint(0, len(images))
    
    #Create predictions and targets
    target_image = images[i] 
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    
    # PLot the image
    plt.imshow(target_image, cmap=plt.cm.binary)
    
    
    #Change the color of the title depending on if the pred is right or wrong
    if pred_label == true_label:
        color = "green"
        
    else:
        color = "red"
        
    
    # Add xlabel information (prediction/ true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100*tf.reduce_max(pred_probs),
                                                     true_label,
                                                     color=color))




plot_random_img(model=model_15, images=test_data_norm, 
                true_labels=test_labels, classes=class_names)






###################################################################
#i'm working on this
import random

def plot_random_img(model, images, true_labels, classes):
    
    # Set up random int
    i = random.randint(0, len(images))
    
    #Create predictions and targets
    target_image = images[i] 
    pred_probs = model.predict(target_image.reshape(6, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    
    # PLot the image
    #plt.imshow(target_image, cmap=plt.cm.binary)
    plt.figure(figsize=(10,10))
    for i in range(6):
        ax = plt.subplot(3, 3, i+1)
        rand_index = random.choice(range(len(train_data_norm)))
        plt.imshow(target_image, cmap=plt.cm.binary)
        plt.title(class_names[pred_probs.argmax()])
        plt.axis(False)

    
    #Change the color of the title depending on if the pred is right or wrong
    if pred_label == true_label:
        color = "green"
        
    else:
        color = "red"
        
    
    # Add xlabel information (prediction/ true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100*tf.reduce_max(pred_probs),
                                                     true_label,
                                                     color=color))




plot_random_img(model=model_15, images=test_data_norm, 
                true_labels=test_labels, classes=class_names)



####What patterns is our model learning?







































































