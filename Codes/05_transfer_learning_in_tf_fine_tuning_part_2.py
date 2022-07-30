"""
TRANSFER LEARNING WITH TENSORFLOW PART 2: FINE-TUNING

          WHAT WE`RE GOING TO COVER

* Introduce fine-tuning transfer learning with Tensorflow
* Introduce the keras Functional API to build models
* Using a small dataset to experiment faster (e.g 10% of training samples)
* Data augmentation (making ur training set more diverse without adding samples)
* Running a series of experiments on our food vision data
* Introduce the ModelCheckpoint callback to save intermediate training results

"""

# Importing the helper function we will be using....

from helper_functions import create_tensorboard_callback
from helper_functions import plot_loss_curves, unzip_data, walk_through_dir, load_and_prep_image
from helper_functions import pred_and_plot


"""
 Lets get some data
* This time we`re going to see how we can use the pretrained models within tf.keras.application
and apply the to our own peoblem (recognizing images of food).

"""

#Get 10 percent of training data of 10 classes of food101


unzip_data("10_food_classes_10_percent.zip")

# Check out how many images and subdirectories are in our dataset
walk_through_dir("10_food_classes_10_percent")

train_dir = "10_food_classes_10_percent/train"
test_dir ="10_food_classes_10_percent/test"

import tensorflow as tf
import keras

IMG_SIZE = (224, 224)
BATCH_SIZE=32

train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            batch_size=BATCH_SIZE)
                                                                            

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                image_size=IMG_SIZE,
                                                                label_mode="categorical",
                                                                batch_size=BATCH_SIZE)
                                                                            


#Check out class name
train_data_10_percent.class_names


"""

# Model 0: Building a transfer learning model using Keras Functional API

The sequential API is straight-forward, it runs our layers in sequential order.
But the functionsal API gives us more flexibility with our models.



"""
from tensorflow.keras import applications
from tensorflow.keras import layers
# 1. Create base model with tf
base_model = applications.EfficientNetB0(include_top=False)

#2 Freeze the base model (so the underlying pre-trained patterns aren`t updated)
base_model.trainable = False

#3. Create inputs into our model
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")

#4. if using ResNet50v2, u will need to nornalize inputs (u dont have to for EfficientNet)
#x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)


#5. Pass the input to the base_model
x = base_model(inputs)
print(f"Shape after passing inputs through base model: {x.shape}")

#6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computation)
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"shape after GlobalAveragePooling2D: {x.shape}")

#7. Create the output activation layer
outputs = layers.Dense(10, activation="softmax", name="Output_layer")(x)


#8. Combine the inputs with the outputs into a model
model_0 = tf.keras.Model(inputs, outputs)





#9. Compile the model
model_0.compile(loss="categorical_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])



#10. fit the model
model_0_history = model_0.fit(train_data_10_percent,
                              epochs=5,
                              steps_per_epoch=len(train_data_10_percent),
                              validation_data=test_data,
                              validation_steps=int(0.25 * len(test_data)),
                              callbacks=[create_tensorboard_callback(dir_name="transfer_learning", 
                                                                     experiment_name="10_percent_features_extraction")])


# Evaluate on the full test model
model_0.evaluate(test_data)


# Check the layers i our base model
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name)


#Summary of the base model
base_model.summary()


model_0.summary()

#Plot loss curve
plot_loss_curves(model_0_history)




#11. save
model_0.save("Model_0")









#############################################################################
# USing EfficientNetB7


from tensorflow.keras import applications
from tensorflow.keras import layers
# 1. Create base model with tf
base_model_1 = applications.EfficientNetB7(include_top=False)

#2 Freeze the base model (so the underlying pre-trained patterns aren`t updated)
base_model_1.trainable = False

#3. Create inputs into our model
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")

#4. if using ResNet50v2, u will need to nornalize inputs (u dont have to for EfficientNet)
#x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)


#5. Pass the input to the base_model
x = base_model_1(inputs)
print(f"Shape after passing inputs through base model: {x.shape}")

#6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computation)
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"shape after GlobalAveragePooling2D: {x.shape}")

#7. Create the output activation layer
outputs = layers.Dense(10, activation="softmax", name="Output_layer")(x)


#8. Combine the inputs with the outputs into a model
model_B7 = tf.keras.Model(inputs, outputs)





#9. Compile the model
model_B7.compile(loss="categorical_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])



#10. fit the model
model_B7_history = model_B7.fit(train_data_10_percent,
                              epochs=5,
                              steps_per_epoch=len(train_data_10_percent),
                              validation_data=test_data,
                              validation_steps=int(0.25 * len(test_data)),
                              callbacks=[create_tensorboard_callback(dir_name="transfer_learning_B7", 
                                                                     experiment_name="10_percent_features_extraction")])


# Evaluate on the full test model
model_B7.evaluate(test_data)


class_names=train_data_10_percent.class_names

load_and_prep_image("03-hamburger.jpeg")
pred_and_plot(model=model_B7, 
              filename="03-hamburger.jpeg",
              class_names=class_names)
# Check the layers i our base model
for layer_number, layer in enumerate(base_model_1.layers):
    print(layer_number, layer.name)


#Summary of the base model
base_model.summary()


model_0.summary()



#Plot loss curve
plot_loss_curves(model_B7_history)


"""

#### Getting a feature vector from a trained model
Lets demonstrate the Global Average Pooling 2D layer......

We have a tensor after our model goes through "base_model" of shape (None, 7, 7, 2560).....

But then when it passes through GlobalAveragePooling2D, it turns into (None, 2560)......

Lets use a similiar shape tensor of (1, 4, 4, 3) and then pass it to GlobalAveragePooling2D

"""

# Define the input shape

input_shape = (1, 4, 4, 3)

# Create a random tensor
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print(f"Random input tensor:\n {input_tensor}")

# Pass the random tensor through a global average pooling 2D layer
global_average_pooled_tensor = layers.GlobalAveragePooling2D()(input_tensor)


global_max_pooled_tensor = layers.GlobalMaxPool2D()(input_tensor)

print(f"2D global average pooled tensor:\n {global_average_pooled_tensor}\n")
print(f"2D global max pooled tensor:\n {global_max_pooled_tensor}\n")


# Check the shape of the different tensors
print(f"Shape of input tensor : {input_tensor.shape}")
print(f"Shape of global average pooled tensor 2D tensor: {global_average_pooled_tensor.shape}")
print(f"Shape of global max pooled tensor 2D tensor: {global_max_pooled_tensor.shape}")









"""
###############################################################################################

# Running a series of transfer learning experiments

we`ve seen the incredible resukts transfer learning can get with only 10% 
of the training data, but how does it go with 1% of training data
.....how about we set up a bunch of experiment to find out

1. 'model_1' - use feature extraction transfer learning with 1% of the training
data with data augmentation.

2. 'model_2' - use feature extraction transfer learning with 10% of the training
with data augmentation.

3. `model_3` - use fine-tuning transfer learning on 10% of the training data with 
data augmentation

4. `model_4` - use fine-tuning transfer learning on 100% of the training data with 
data augmentation

NOTE: throughout all the experiments, the same test dataset will be used to evaluate
our model....This ensures consistency across evaluation metrics.

"""
"""
   GETTING AND PREPROCESSING DATA FOR MODEL_1
   
"""
# Download and unzip data - preprocessed from food101

#!wget (link)
unzip_data("10_food_classes_1_percent.zip")

#Create training and test dirs
train_data_1_percent = "10_food_classes_1_percent/train"
test_data = "10_food_classes_1_percent/test"

#How many images are we working with?
walk_through_dir("10_food_classes_1_percent")

#Set data loaders

from tensorflow.keras import preprocessing

IMG_SIZE = (224, 224)
BATCH_SIZE =32


train_data_1_percent = preprocessing.image_dataset_from_directory(directory=train_data_1_percent,
                                                                  label_mode="categorical",
                                                                  image_size=IMG_SIZE,
                                                                  batch_size=BATCH_SIZE
                                                                  )


test_data = preprocessing.image_dataset_from_directory(directory=test_data,
                                                      label_mode="categorical",
                                                      image_size=IMG_SIZE,
                                                      batch_size=BATCH_SIZE
                                                      )



## Adding data augmentation in the model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing


# Create data augmentation stage with horizontal flipping, rotations, zooms, etc....

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    #preprocessing.Rescaling(1./255) #keep for models like ResNet50V2 but not EfficientNet
    ], name="data_augmentation")



# Visualize our data augmentation layer (and see what happens to our data)
#view a random image and compare it to its augmented version

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random


def show_augmented_and_norm_img(target_dir=True):
    
    target_class = random.choice(train_data_1_percent.class_names)
    #print(target_class)
    target_dir = "10_food_classes_1_percent/train/" + target_class
    random_image = random.choice(os.listdir(target_dir))
    random_img_path = target_dir + "/" + random_image
    print(random_img_path)
    
    # Read and plot in the random  image
    img = mpimg.imread(random_img_path)
    plt.imshow(img)
    plt.title(f"Original random image from class:  {target_class}")
    plt.axis(False)
    
    
    # Now let`s plot our augmented random
    plt.figure()
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
    plt.imshow(tf.squeeze(augmented_img)/255.)
    plt.title(f"Augmented random image from class:  {target_class}")
    plt.axis(False);



show_augmented_and_norm_img()



## 1. 'model_1' - use feature extraction transfer learning with 1% of the training
#data with data augmentation.

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing

"""

# Setup input shape and base model,freezing the base model layers
from tensorflow.keras import applications

input_shape = (224, 224, 3)

base_model_2 = applications.EfficientNetB0(include_top=False)
base_model_2.trainable = False

#Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentations Sequential model as a layer
x = data_augmentation(inputs)

# Give base_model the input (after augmentation ) and dont train it
x = base_model_2(x, training=False)

# Pool output features of the base model
x = layers.GlobalAveragePooling2D(name="Global_Average_Pooling_2D")(x)

# Put a dense layer on as the output
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)



# Make the model using the input and output
from keras import Model
model_1 = Model(inputs, outputs)



#COmpile the model
model_1.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])


# Fit the model
history_1_percent = model_1.fit(train_data_1_percent, 
                                epochs=5,
                                steps_per_epoch=len(train_data_1_percent),
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                #Track model training logs
                                callbacks=[create_tensorboard_callback(dir_name="trainsfer_learing",
                                                                       experiment_name="1_percent_data_aug")]
                                )


plot_loss_curves(history_1_percent)
model_1.summary()

results_1_percent_data_aug = model_1.evaluate(test_data)






"""
##################################################################################
2. 'model_2' - use feature extraction transfer learning with 10% of the training
with data augmentation.

"""


# Check out how many images and subdirectories are in our dataset
walk_through_dir("10_food_classes_10_percent")

train_dir = "10_food_classes_10_percent/train"
test_dir ="10_food_classes_10_percent/test"

import tensorflow as tf
import keras

IMG_SIZE = (224, 224)
BATCH_SIZE=32



from tensorflow.keras import preprocessing


train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            batch_size=BATCH_SIZE)
                                                                            

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                image_size=IMG_SIZE,
                                                                label_mode="categorical",
                                                                batch_size=BATCH_SIZE)
                                                                            



## Adding data augmentation in the model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing


# Create data augmentation stage with horizontal flipping, rotations, zooms, etc....

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    #preprocessing.Rescaling(1./255) #keep for models like ResNet50V2 but not EfficientNet
    ], name="data_augmentation")




# Visualize our data augmentation layer (and see what happens to our data)
#view a random image and compare it to its augmented version

def show_augmented_and_norm_img(target_dir=True):
    
    target_class = random.choice(train_data_1_percent.class_names)
    #print(target_class)
    target_dir = "10_food_classes_10_percent/train/" + target_class
    random_image = random.choice(os.listdir(target_dir))
    random_img_path = target_dir + "/" + random_image
    print(random_img_path)
    
    # Read and plot in the random  image
    img = mpimg.imread(random_img_path)
    plt.imshow(img)
    plt.title(f"Original random image from class:  {target_class}")
    plt.axis(False)
    
    
    # Now let`s plot our augmented random
    plt.figure()
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
    plt.imshow(tf.squeeze(augmented_img)/255.)
    plt.title(f"Augmented random image from class:  {target_class}")
    plt.axis(False);



show_augmented_and_norm_img()







"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing

"""

# Setup input shape and base model,freezing the base model layers
from tensorflow.keras import applications

input_shape = (224, 224, 3)

base_model2 = applications.EfficientNetB0(include_top=False)
base_model2.trainable = False

#Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentations Sequential model as a layer
x = data_augmentation(inputs)

# Give base_model the input (after augmentation ) and dont train it
x = base_model2(x, training=False)

# Pool output features of the base model
x = layers.GlobalAveragePooling2D(name="Global_Average_Pooling_2D")(x)

# Put a dense layer on as the output
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)



# Make the model using the input and output
from keras import Model
model_2 = Model(inputs, outputs)



#COmpile the model
model_2.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])


# Fit the model
history_10_percent_model2 = model_2.fit(train_data_10_percent, 
                                epochs=5,
                                steps_per_epoch=len(train_data_10_percent),
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                #Track model training logs
                                callbacks=[create_tensorboard_callback(dir_name="trainsfer_learing",
                                                                       experiment_name="10_percent_data_aug")]
                                )


plot_loss_curves(history_10_percent_model2)
model_2.summary()

results_10_percent_data_aug = model_2.evaluate(test_data)


"""
###Creating tensorflow callback modeolcheckpoint

The MOdelCheckpoint callbackintermediately svaes our model(the full model or just the weights) 
during training. This is useeful so we can come and start where we left off

"""
from tensorflow.keras import callbacks

#Set checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"

# Create a ModelCheckpoint callback that saves the model`s weights only

checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=False,
                                                save_freq="epoch", 
                                                verbose=1)






# Fit the model2 passing in the ModelCheckpoint callback
initial_epochs = 5


history_10_percent_data_aug = model_2.fit(train_data_10_percent, 
                                epochs=initial_epochs,
                                steps_per_epoch=len(train_data_10_percent),
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                #Track model training logs
                                callbacks=[create_tensorboard_callback(dir_name="trainsfer_learing",
                                                                       experiment_name="10_percent_data_aug1"),
                                           checkpoint_callback]
                                )






plot_loss_curves(history_10_percent_data_aug)
model_2.summary()

results_10_percent_data_aug1 = model_2.evaluate(test_data)





###Loading in checkpointed weights
# Loading in checkpointed weights returns a model to a specific checkpoint

# Load in saved model weights and evaluate model

model_2.load_weights(checkpoint_path)


# Evaluate model_2 with loaded weights

loaded_weights_model_results = model_2.evaluate(test_data)



# How to compair two different two numbers to know if theyre the same and to know how close they re
import numpy as np


np.isclose(np.array(results_10_percent_data_aug1), np.array(loaded_weights_model_results))


1 == 1

results_10_percent_data_aug1 == loaded_weights_model_results












"""
################################################################################################
3. `model_3` - use fine-tuning transfer learning on 10% of the training data with 
data augmentation

FINE-TUNING USUALLY WORKS BEST AFTER TRAINING A FEATURE EXTRACTION MODEL FOR A FEW EPOCHS
WITH LARGE AMOUNT OF CUSTOM

"""


# Layers in loaded model
model_2.layers


# Are these layers trainable?

for layer in model_2.layers:
    print(layer, layer.trainable)


# What layers are in our base model (EfficientNetB0) and are they trainable?

for i, layer in enumerate(model_2.layers[2].layers):
    print(i, layer.name, layer.trainable)



# How many trainable variables are in our base model?
print(len(model_2.layers[2].trainable_variables))


# To begin fine-tuning, lets start by setting the last 10 layers of our base_model.trainable = True

base_model2.trainable = True

# Freeze all layers except for the last 10
for layer in base_model2.layers[:-10]:
    layer.trainable = False


#from tensorflow.keras.optimizers import Adam
# Recompile (we have to recompile every time we make a change)
model_2.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.0001), #when fine-tuning, u typically want to lower ur learning rate  by 10x*
                metrics=["accuracy"])

#For best learning rate to use on ur fine-tuning, go to ULMFIT PAPER

# Check which layers are tunable (trainable)
for layer_number, layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)



# How many trainable variables are in our base model?
print(len(model_2.layers[2].trainable_variables))


#Fine tune for another 5 epochs

fine_tune_epochs = initial_epochs + 5

# Refit the model (same as mdel_2 except with more trainable layers)

history_fine_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data,
                                               validation_steps=int(0.25 * len(test_data)),
                                               initial_epoch=history_10_percent_data_aug.epoch[-1], # start training from the previous epoch
                                               callbacks=[create_tensorboard_callback(dir_name="trainsfer_learing",
                                                                                      experiment_name="10_percent_fine_tune_last_10")]
                                               )




plot_loss_curves(history_fine_10_percent_data_aug)
model_2.summary()


results_fine_tune_10_percent_data_aug = model_2.evaluate(test_data)



results_10_percent_data_aug1



"""
## The 'plot_loss_curves' function wporks great with models which have only been 
fit once, however, we want something to compare our series of running 'fit()'
with another....(e.g. before and after fine-tuning)

"""

# Lets create a function to compare training histories
from helper_functions import compare_historys

compare_historys(history_10_percent_data_aug,
                 history_fine_10_percent_data_aug,
                 initial_epochs=5)











"""
##############################################################################################

4. `model_4` - use fine-tuning transfer learning on 100% of the training data with 
data augmentation

"""





"""
version 2
##############################################################################################

4. `model_4` - use fine-tuning transfer learning on 100% of the training data with 
data augmentation

"""




# Check out how many images and subdirectories are in our dataset
walk_through_dir("10_food_classes_10_percent")

train_dir = "10_food_classes_all_data/train"
test_dir ="10_food_classes_all_data/test"

import tensorflow as tf
import keras

IMG_SIZE = (224, 224)
BATCH_SIZE=32



from tensorflow.keras import preprocessing


train_data_all_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            batch_size=BATCH_SIZE)
                                                                            

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                image_size=IMG_SIZE,
                                                                label_mode="categorical",
                                                                batch_size=BATCH_SIZE)
                                                                            



# Load weights from checkpoint, that way can fine-tune from the same stage the 10 
#percent data model was fine-tuned from

model_2.load_weights(checkpoint_path)

model_2.evaluate(test_data)


#Check to see if our model_2 has been reverrted back to feature extraction result
results_10_percent_data_aug1


# Check which layers are tunable (trainable)
for layer_number, layer in enumerate(model_2.layers):
    print(layer_number, layer.name, layer.trainable)


# Check which layers are tunable (trainable)
for layer_number, layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)



# Are these layers trainable?

for layer in model_2.layers:
    print(layer, layer.trainable)


# What layers are in our base model (EfficientNetB0) and are they trainable?

for i, layer in enumerate(model_2.layers[2].layers):
    print(i, layer.name, layer.trainable)



# How many trainable variables are in our base model?
print(len(model_2.layers[2].trainable_variables))

# Recompile (we have to recompile every time we make a change)
model_2.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.0001), #when fine-tuning, u typically want to lower ur learning rate  by 10x*
                metrics=["accuracy"])



# Continue to train and fin e-tune the model to our data (100% of training dtat)

#Fine tune for another 5 epochs

fine_tune_epochs = initial_epochs + 5

# Refit the model (same as mdel_2 except with more trainable layers)

history_fine_all_data_aug1 = model_2.fit(train_data_all_data,
                                        epochs=fine_tune_epochs,
                                        validation_data=test_data,
                                        validation_steps=int(0.25 * len(test_data)),
                                        initial_epoch=history_10_percent_data_aug.epoch[-1], # start training from the previous epoch
                                        callbacks=[create_tensorboard_callback(dir_name="trainsfer_learing",
                                                                               experiment_name="full_10_classes_fine_tune_last_10")]
                                        )





plot_loss_curves(history_fine_all_data_aug1)
model_2.summary()


results_fine_tune_all_data_aug1 = model_2.evaluate(test_data)



results_10_percent_data_aug1






compare_historys(history_fine_10_percent_data_aug,
                 history_fine_all_data_aug1,
                 initial_epochs=5)























