"""""
CONVOLUTIONAL NEURAL NETWORK AND COMPUTER VISION

       WHAT WE'RE GOING TO COVER
       
* Getting a dataset to work with (pizza_steak)
* Architecture of a convolutional neural network(CNN) with tensorflow
* An end-to-end binary image classification problem
* Sets in modelling with CNNs
* Creating a CNN, compiling a model, fitting a model, evaluating a model
* An end-to-end multi-class image classification problem
* Making predictions on our own custom images


 * Computer vision is then practice of writing algorithms which discover patterns
   in visual data. Such as the camera of a self-driving car recognizing the car in front
               LET'S CODE!!!

"""""
import tensorflow as tf
import zipfile

# Unxip the downloaded file

zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()


## Inspect the data (become one with it)

!ls pizza_steak/train

!ls pizza_steak/train/steak

import os

# Walk through pizza_steak directory and list number of files

for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))

num_steak_images_train

# To visualize our images, first let's get the class names programmatically

# Get the classnames programmatically

import pathlib
import numpy as np

data_dir = pathlib.Path("pizza_steak/train")

class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

print(class_names)

# Let's visualize our data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def view_random_image(target_dir, target_class):
    #Setthe target directory(we'll view images from here)
    target_folder = target_dir+target_class
    
    #Get a random image path
    random_image = random.sample(os.listdir(target_folder), 2)
    print(random_image)
    
    #Read in the img and plot it
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    
    print(f"Image shape: {img.shape}")
    
    return img



# View a random image(steak) from the training set

img_steak = view_random_image(target_dir="pizza_steak/train/", 
                        target_class="steak")


# View a random image(pizza) from the training set

img_pizza = view_random_image(target_dir="pizza_steak/train/", 
                        target_class="pizza")

""""
 An end-to-end example
Let's build a convolutional neural nework to find patterns in our image, 
more specifically we need a way to...
* Load our images
* Preprocess our images
* Build a CNN to find patterns in our images
* Compile our CNN
* Fit the CNN to our training data

""""

import tensorflow as tf
from  tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed
tf.random.set_seed(42)

#Proprocess data (get all of the pixel values btw 0 & 1, also called scaling/norms)
train_datagen = ImageDataGenerator(rescale=1./255)
vaild_datagen = ImageDataGenerator(rescale=1./255)

# Setup paths to our data directories
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)


valid_data = vaild_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)


# Build a CNN model (same as the Tiny VGG on the CNN explainer website)

model_1 =tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    #tf.keras.layers.Activations(tf.nn.relu)
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


# Compile our model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_1 = model_1.fit(train_data, epochs=5, 
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_1.summary()
model_1.evaluate(valid_data)

##Read CNN Explainer


####Plot history
import pandas as pd
##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_1.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")







##################################################################################
# Build a NON-CNN model (model to reolicate the tensorflow playground  model)
#set random seed
tf.random.set_seed(42)

# Create a model to reolicate the tensorflow playground  model

model_2 =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])


# Compile our model
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_2 = model_2.fit(train_data, epochs=5, 
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))


model_2.summary()





##################################################################
#Despite having more params than our CNN (model_1),
#model_2 preforms terribly.....let's try to improve it
#


#set random seed
tf.random.set_seed(42)

# Create a model to reolicate the tensorflow playground  model

model_3 =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])


# Compile our model
model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_3 = model_3.fit(train_data, epochs=5, 
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))


model_1.summary()
model_3.summary()












###################################################
# Let's break it down

#Visualize data
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
steak_img = view_random_image(target_dir="pizza_steak/train/", 
                              target_class="steak")

plt.subplot(1, 2, 2)
steak_img = view_random_image(target_dir="pizza_steak/train/", 
                              target_class="pizza")




!nvidia-smi





##################################################################
#Make the creating of our model a little easier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential



# Create the model (this will be our baseline, a layer convolutional neural network)

model_4 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           strides=1,
           padding="valid",
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])

##Assignment ...*PAPER WITH CODE

model_4.compile(loss="binary_crossentropy",
                optimizer=Adam(learning_rate=0.001),
                metrics=["accuracy"])

model_4.summary()

history_4 =model_4.fit(train_data,
            epochs=5,
            steps_per_epoch=len(train_data),
            validation_data=valid_data,
            validation_steps=len(valid_data),
            )


## Evaluting our model

model_1.evaluate(valid_data)
model_2.evaluate(valid_data)
model_3.evaluate(valid_data)
model_4.evaluate(valid_data)

import pandas as pd

pd.DataFrame(history_1.history).plot(figsize=(10, 7));

#Lets make it beautiful

def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    epochs = range(len(history.history["loss"]))
    
    #Plot loss
    
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid();
    plt.Figure()
    
    #Plot accuracy
    plt.Figure()
    plt.plot(epochs, accuracy, label="Training ccuracy")
    plt.plot(epochs, val_accuracy, label="Val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()
    



plot_loss_curves(history_1)





"""
##########################################
# Adjust the model parameters

* Create a baseline
* Beat the baseline by overfitting a larger model
* Reduce overfitting


Ways to induce overfitting:
   * Increase the no of Conv layers
   * Increase the no of filters
   * Add another dense layer to the output of our flattend layer
    
    
    
REDUCE OVERFITTING:
    * Add data augmentation
    * Add regularization layers (such as MaxPool2D)
    * Add more data
        
        
"""

# Create the model(this is going to be our new baseline)
model_5 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile the model
model_5.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

# Fitting the model

history_5 = model_5.fit(train_data,
            epochs=5,
            steps_per_epoch=len(train_data),
            validation_data=valid_data,
            validation_steps=len(valid_data),
            )

# Getting the summary of our model

model_5.summary()

model_5.evaluate(valid_data)
plot_loss_curves(history_5)







## Opening our bag of tricks and find data augmentation
# create ImageDataGen training instance with data augmentation

train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.3,
                                             horizontal_flip=True)






# Create ImageDatagen without data augmentation both on train and valid dataset
train_datagen = ImageDataGenerator(rescale=1./255)
vaild_datagen = ImageDataGenerator(rescale=1./255)


"""
### WHAT IS DATA AUGMENTATION?

Data augmentation is the process of altering our training data,
leading it to have more diversity and in turn allowing our models to learn more
generalizable (hopefully) patterns.
Altering might mean adjusting the rotation of an image, flipping it, cropping it
or something similar.


  Let's write some code to visualize our image augmentation

"""


# Import data and augment it from training directory
print("Augmented training data")
train_datagen_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                                     batch_size=32,
                                                                     target_size=(224, 224),
                                                                     class_mode="binary",
                                                                    shuffle=False)
                                                                       

# Create non-augmented train data batches
print("Non-Augmented training data")
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               shuffle=False)


# Create non-augmented test data batches
print("Non-Augmented testing data")
valid_data = train_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               shuffle=False)


"""
NOTE: Data augmentation is usually only performed on the training data. Using
"ImageDataGenerator" built-in data augmentation parameters, our images are left
as they are in the directories but are modified as they're loaded into the model 

Finally.... Let's visualize some augmented data!!!!!!!!!!!!

"""

# Get sample data batches
images, labels = train_data.next()
augmented_images, augmented_labels = train_datagen_augmented.next()

# Show original image and augmented image

import random

random_number = random.randint(0, 32)
print(f"showing image number: {random_number}")
plt.imshow(images[random_number])
plt.title(f"Original image")
plt.axis(False);
plt.Figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image")
plt.axis(False);



# Create the model(this is going to be our new baseline) that trains augmented data
model_6 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile the model
model_6.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

# Fitting the model

history_6 = model_6.fit(train_datagen_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_datagen_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        )

# Getting the summary of our model

model_6.summary()

model_6.evaluate(valid_data)
plot_loss_curves(history_6)








#######################################################################


# Import data and augment it from training directory and turning ON our shuffle
print("Augmented training data")
train_data_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                                    batch_size=32,
                                                                    target_size=(224, 224),
                                                                    class_mode="binary",
                                                                    shuffle=True)
                                                                    



# Create the model(this is going to be our new baseline) that trains augmented data
model_7 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile the model
model_7.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

# Fitting the model

history_7 = model_7.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        )

# Getting the summary of our model

model_7.summary()

model_7.evaluate(valid_data)
plot_loss_curves(history_7)



"""
REPEAT UNTIL STAISFIED

Since  we`ve already beaten our baseline, there are a few things we could try to
continue to improve our model


* Increase the numvber of model layers (e.g add more Conv2D/ MaxPool2D)
* Increase the number of filters in each convolutional layer (e.g from 10 to 32 or even 64)
* Train for longer (more epochs)
* Find an ideal learning rate
* Get more data
* Use TRANSFER LEARNING to leverage whta another image model has learn and
  adjust it for our own use cases




    PRACTICE
    
Recreate the model on the CNN explainer website (same as MODEL_1) and see
how it performs on the augmented shuffled training data.

"""

# Increase the number of filters in each convolutional layer (e.g from 10 to 32 or even 64)
#we will be using 10


# Build a CNN model (same as the Tiny VGG on the CNN explainer website)
model_8 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=2, padding="valid"),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile our model
model_8.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_8 = model_8.fit(train_data_augmented, epochs=5, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_8.summary()

model_8.evaluate(valid_data)
plot_loss_curves(history_8)







"""
Same as model_8 but this time, we won`t use augmented one

"""


# Build a CNN model (same as the Tiny VGG on the CNN explainer website)
model_8p = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=2, padding="valid"),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile our model
model_8p.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_8p = model_8p.fit(train_data, epochs=5, 
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_8.summary()

model_8.evaluate(valid_data)
plot_loss_curves(history_8)



"""
Looks like model_1 preforms pretty good. lets try this out....
how about using model_1 on our augmented data? sounds cool right?
let`s dive in

"""


# Build a CNN model (same as the Tiny VGG on the CNN explainer website)

model_1_augmented =tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    #tf.keras.layers.Activations(tf.nn.relu)
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


# Compile our model
model_1_augmented.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_1_augmented = model_1_augmented.fit(train_data_augmented, epochs=5, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_1_augmented.summary()
model_1.evaluate(valid_data)
model_1_augmented.evaluate(valid_data)

plot_loss_curves(history_1_augmented)







##################################################################################


# Increase the number of filters in each convolutional layer (e.g from 10 to 32 or even 64)
#we will be using 32

# Build a CNN model (same as the Tiny VGG on the CNN explainer website)
model_9 = Sequential([
    Conv2D(filters=32,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(32, 3, activation="relu"),
    MaxPool2D(pool_size=2, padding="valid"),
    Conv2D(32, 3, activation="relu"),
    Conv2D(32, 3, activation="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile our model
model_9.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_9 = model_9.fit(train_data_augmented, epochs=5, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_9.summary()

model_9.evaluate(valid_data)
plot_loss_curves(history_9)




########################################################

# Increase the number of filters in each convolutional layer (e.g from 10 to 32 or even 64)
#we will be using 64

# Build a CNN model (same as the Tiny VGG on the CNN explainer website)
model_10 = Sequential([
    Conv2D(filters=64,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(64, 3, activation="relu"),
    MaxPool2D(pool_size=2, padding="valid"),
    Conv2D(64, 3, activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 output neuron)
    ])


# Compile our model
model_10.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting

history_10 = model_10.fit(train_data_augmented, epochs=5, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data, 
                        validation_steps=len(valid_data))

##Get model summary

model_10.summary()

model_10.evaluate(valid_data)
plot_loss_curves(history_10)







# Making a prediction with our trained model on our custom data

# Classes we`re working with

print(class_names)


# View our example image
import matplotlib.image as mpimg

steak = mpimg.imread("03-steak.jpeg")
plt.imshow(steak)
plt.axis(False);

# Create a function to import an image and resize it to be able to be used with
#our model

def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename,turns it into a tensor and rshapes it to (img_shape, img_shape, colors).

    """
    
    #Read in the image
    img = tf.io.read_file(filename)
    
    # Decoe the read file into a tensor
    img = tf.image.decode_image(img)
    
    #Resize the img
    img = tf.image.resize(img, size=(img_shape, img_shape))
    
    # Rescale the img and get all value bte 0 and 1
    img = img/255.
    return img



# Load in and preprocess our custom images
steak = load_and_prep_image("03-pizza-dad.jpeg")

steak

pred = model_1.predict(tf.expand_dims(steak, axis=0))

pred

pred_augmented = model_1_augmented.predict(tf.expand_dims(steak, axis=0))

pred_augmented

"""
Lookes like our custom data image is being put through our model, however,
it currently outputs a prediction probability, wouldn`t it be nice if we could visualize 
the image as well as the model`s predictions

"""
class_names

# We can index the predicted class by rounding the prediction probability and indexing it
# on the class pred

pred_class = class_names[int(tf.round(pred))]

pred_class


def pred_and_plot(model, filename, class_names=class_names):
    """
    Imports an iomage loacted at filename, makes a prediction with model and plots
    the image with the predicted class as the title

    """
    
    # Import the targe image and preprocess it
    img = load_and_prep_image(filename)
    
    # MAke a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred))]
    
    # Plot the image and predicted class
    print(f"Prediction Accuracy: {pred}")
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);


pred_and_plot(model_1, "03-pizza-dad.jpeg")
pred_and_plot(model_1_augmented, "03-pizza-dad.jpeg")


#  To download something on colab, you use !wget followed by the link




##EVALUATE OUR TRAINED MODELS
model_1.evaluate(valid_data)
model_2.evaluate(valid_data)
model_3.evaluate(valid_data)
model_4.evaluate(valid_data)
model_5.evaluate(valid_data)
model_6.evaluate(valid_data)
model_7.evaluate(valid_data)
model_8.evaluate(valid_data)
model_9.evaluate(valid_data)
model_10.evaluate(valid_data)








"""
Multi-class Image Classification


we`ve just been through a bunch of the following steps with a binary classification
problem (pizza bs. steak), now we`re going to step things up a notch with 10 classes of
of food (multi-class classification)

   STEPS
   
1. Become one with the data
2. Proprocess the data (get it read for a model)
3. Create a model (start with a baseline)
4. Fit the model (overfit it to make sure it works)
5. Evaluate the model
6. Adjust different hyperparameters and improve the model (try to beat 
                                                           baseline/reduce overfitting)
7. Repeat until satisfied



 working with 10 classes
"""



import tensorflow as tf
import zipfile

# Unxip the downloaded file

zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip")
zip_ref.extractall()
zip_ref.close()



# Walk through 10 classes of food 

import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")





# Another way to find out how many images are in a file
images_train = len(os.listdir("10_food_classes_all_data/train/"))

images_train

# To visualize our images, first let's get the class names programmatically

# Get the classnames programmatically

import pathlib
import numpy as np

data_dir = pathlib.Path("10_food_classes_all_data/train")

class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

print(class_names)


# Setup the train and test directories

train_dir = "10_food_classes_all_data/train/"

test_dir = "10_food_classes_all_data/test/"

# Visualize, Visualize, Visualize
import random
img = view_random_image(target_dir=train_dir,
                        target_class=random.choice(class_names))




##2. Preprocess the data (prepare it for model)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Rescale

train_gen = ImageDataGenerator(rescale=1./255)



valid_gen = ImageDataGenerator(rescale=1./255)


train_datagen = train_gen.flow_from_directory(directory=train_dir,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode="categorical")

valid_datagen = valid_gen.flow_from_directory(directory=test_dir,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode="categorical")


train_datagen[0].shape()
"""
3. Create a model (start with a baseline....)
 We`ve been talking a lot about the CNN Explainer website....
 how about we jusst take their model (also on 10 classes) and use it for our problem

"""


##################################################################
#Make the creating of our model a little easier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

len(class_names)

# Create the model (this will be our baseline, a layer convolutional neural network)

model_11 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax") # Change to have 10 output neurons and use the softmax activation function
    ])

# Compile our model
model_11.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting
history_11 = model_11.fit(train_datagen, epochs=5, 
                        steps_per_epoch=len(train_datagen),
                        validation_data=valid_datagen, 
                        validation_steps=len(valid_datagen))

##Get model summary

model_11.summary()
model_11.evaluate(valid_datagen)

##Read CNN Explainer


####Plot history
import pandas as pd
##Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_11.history["loss"]).plot()
pd.DataFrame(history_11.history["val_loss"]).plot()
pd.DataFrame(history_11.history["accuracy"]).plot()
pd.DataFrame(history_11.history["val_accuracy"]).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")

plot_loss_curves(history_11)







"""

6. Adjust different hyperparameters and improve the model (try to beat 
                                                           baseline/reduce overfitting)


Due to it's performance on the training data.. its clear that our model is learning something..

However, its not generalizing well to unseen data (overfitting)

So, lets try and fix the overfitting by

*  Get more data
* Simpily the model
* Use data augmentation
* Use transfer learning

"""

"""
* Simpify the model
* Lets try to remove 2 convolutional layers

"""



model_12 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax") # Change to have 10 output neurons and use the softmax activation function
    ])

# Compile our model
model_12.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting
history_12 = model_12.fit(train_datagen, epochs=5, 
                        steps_per_epoch=len(train_datagen),
                        validation_data=valid_datagen, 
                        validation_steps=len(valid_datagen))






model_12.summary()



"""
Looks like our "simplifying the model" experiment didn`t work...
the accuracy went down and overfitting continued...

how about we try data augmentation

trying to reduce overfitting with data augmentation

* Use data augmentation

"""

train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.3,
                                             horizontal_flip=True)

train_data_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                                   batch_size=32,
                                                                   class_mode="categorical",
                                                                   target_size=(224, 224))





# CReating our model with our augmented data

model_13 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax") # Change to have 10 output neurons and use the softmax activation function
    ])

# Compile our model
model_13.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting
history_13 = model_13.fit(train_data_augmented, epochs=5, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_datagen, 
                        validation_steps=len(valid_datagen))

##Get model summary

model_13.summary()
model_13.evaluate(valid_datagen)



plot_loss_curves(history_13)


"""
7. Repeat until satisfied

"""




# CReating our model with our augmented data

model_14 = Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax") # Change to have 10 output neurons and use the softmax activation function
    ])

# Compile our model
model_14.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fitting
history_14 = model_14.fit(train_data_augmented, epochs=10, 
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_datagen, 
                        validation_steps=len(valid_datagen))

##Get model summary

model_14.summary()
model_14.evaluate(valid_datagen)



plot_loss_curves(history_14)







"""


#####################################################################################3
#Making predictions with our trained model

Lets use our trained model to make some predictions on our own custom images!


RECONFIG PRED_AND_PLOT FUNCTION TO WORK WITH MULTI-CLASS IMAGES

"""


def pred_and_plot(model, filename, class_names=class_names):
    """
    Imports an iomage loacted at filename, makes a prediction with model and plots
    the image with the predicted class as the title

    """
    
    # Import the targe image and preprocess it
    img = load_and_prep_image(filename)
    
    # MAke a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Add in logic for multi-class and get pred_class name
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]
    
  
    
    # Plot the image and predicted class
    print(f"Prediction Accuracy: {pred}")
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);






class_names

pred_and_plot(model=model_7, filename="03-hamburger.jpeg")
pred_and_plot(model=model_7, filename="03-pizza-dad.jpeg")
pred_and_plot(model=model_7, filename="03-steak.jpeg")
pred_and_plot(model=model_7, filename="03-sushi.jpeg")




"""
Saving and loading our model

"""

model_1.save("saved_trained_model_1")
model_7.save("saved_trained_model_7")
model_8.save("saved_trained_model_8")
model_9.save("saved_trained_model_9")



"""
Load in a trained model and evaluate it
"""

import tensorflow as tf
loaded_model_1 = tf.keras.models.load_model("saved_trained_mo0del_1")
loaded_1 = tf.keras.models.load_model("saved_model")
loaded_1.evaluate(valid_data)
loaded_2 = tf.keras.models.load_model("saved_model")
loaded_3 = tf.keras.models.load_model("saved_model")

loaded_model_1.summary()

loaded_model_1.evaluate(valid_data)










##EVALUATE OUR TRAINED MODELS
model_1.evaluate(valid_data)
#model_2.evaluate(valid_data)
#model_3.evaluate(valid_data)
#model_4.evaluate(valid_data)
#model_5.evaluate(valid_data)
model_6.evaluate(valid_data)
model_7.evaluate(valid_data)
model_8.evaluate(valid_data)
model_9.evaluate(valid_data)
#model_10.evaluate(valid_data)
model_11.evaluate(valid_data)
model_12.evaluate(valid_data)
model_13.evaluate(valid_data)
model_14.evaluate(valid_data)










































