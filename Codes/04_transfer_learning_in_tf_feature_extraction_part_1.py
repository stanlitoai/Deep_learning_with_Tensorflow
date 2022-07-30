"""
* Introduce transfer learning with tensorflow
* Using a small dataset to experiment faster (10% of training sample)
* Building a transfer learning feature extraction model with tensorflow Hub
* Use tensorBoard to track modelling experiments and results.



  DO THE BEST YOU CAN UNTIL YOU KNOW BETTER. THEN WHEN YOU KNOW BETTER, DO BETTER.
     
* Tensorflow Learning with Tensorflow Part 1: Feature Extraction

Transfer learning is leaveraging a working model`s existing architecture and learned 
paterns for our problem

   THERE ARE TWO MAIN BENEFITS:

*1. Can leverage an existing neural network architecture proven to work on problem
similiar to our own

*2. Can leverage a working neural network archtecture which has already learned patterns
on similiar data to our own, then we can adapt tghose patterns to our own data

"""



import tensorflow as tf
import zipfile

# Unxip the downloaded file

zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_ref.extractall()
zip_ref.close()



## Inspect the data (become one with it)

!ls 10_food_classes_10_percent/train

!ls 10_food_classes_10_percent/test

import os

# Walk through pizza_steak directory and list number of files

for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("10_food_classes_10_percent/train/chicken_curry"))

num_steak_images_train


from  tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5


train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"


# Set the seed
tf.random.set_seed(42)

#Proprocess data (get all of the pixel values btw 0 & 1, also called scaling/norms)
train_datagen = ImageDataGenerator(rescale=1./255)
vaild_datagen = ImageDataGenerator(rescale=1./255)



print("Training images: ")
train_data_10_percent = train_datagen.flow_from_directory(directory=train_dir,
                                                          target_size=IMG_SHAPE,
                                                          batch_size=BATCH_SIZE,
                                                          class_mode="categorical")



print("Testing images: ")
test_data = train_datagen.flow_from_directory(directory=test_dir,
                                             target_size=IMG_SHAPE,
                                             batch_size=BATCH_SIZE,
                                             class_mode="categorical")



"""
 Setting up callbacks (things to run while our models trains)

Callbacks are extra functionality you can add to ur models to be predformed during or after
 training. some of the most popular callbacks...
 
* Tracking experiments with the TensorBoard callback.
* Model checkpoint with the modelcheckpoint callback
* Stopping a model from training (before it trains too long or overfis) 
wit the earlyStopping callback

"""



# Create TensorBoard callback (functionized because we need to create a new one for each model)
!pip install --upgrade tensorflow_hub

import datetime

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback



import tensorflow_hub as hub

# Lets compare the following two models
#resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
resnet_url = "imagenet_resnet_v2_50_feature_vector_5"

efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
efficientnet_url = "efficientnet_b0_feature-vector_1"


efficientnet7_url = "efficientnet_b7_feature-vector_1"




## Import dependencies

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Lets make a create_model() function to create a model from URL

def create_model(model_url, num_classes=10):
    """
    Takes a Tensorflow Hub url and creates a keras Sequential model with it.
    
    
    Args:
        model_url (str) : A TensorFLow Hub feature extraction URL.
        num_classes (int) : Number of output neurons in the output layer,
        should be equal to number of target classes, default 10.
        
        
    Returns:
        An uncomplied keras Sequential model with model_url as feature extractor
        layer and Dense output layer with num_classes output neurons.
    """
    
    # Download the pretrained model and save it as keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             name="Feature_extraction_layer",
                                             trainable=False,
                                             input_shape=IMG_SHAPE+(3,))
    
    
    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="Output_layer")
        ])
    
    return model




# Creating and testing ResNet TensorFlow Hub Feature extraction model

resnet_model = create_model(resnet_url,
                            num_classes=train_data_10_percent.num_classes)





resnet_model.summary()


# Compile our resnet model
resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])


#fitting
resnet_history = resnet_model.fit(train_data_10_percent,
                             epochs=EPOCHS,
                             steps_per_epoch=len(train_data_10_percent),
                             validation_data=test_data,
                             validation_steps=len(test_data),
                             callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                    experiment_name="resnet50V2")])





#Plot our loss curves
import matplotlib.pyplot as plt


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
    plt.grid()
    
    
    #Plot accuracy
    plt.Figure()
    plt.plot(epochs, accuracy, label="Training ccuracy")
    plt.plot(epochs, val_accuracy, label="Val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend();
    



plot_loss_curves(resnet_history)



























####################################################################################################
# Creating and testing ResNet TensorFlow Hub Feature extraction model



efficientnet_model = create_model(efficientnet_url,
                            num_classes=train_data_10_percent.num_classes)



efficientnet_model.summary()



# Compile our resnet model
efficientnet_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])


#fitting

efficientnet_history = efficientnet_model.fit(train_data_10_percent,
                             epochs=EPOCHS,
                             steps_per_epoch=len(train_data_10_percent),
                             validation_data=test_data,
                             validation_steps=len(test_data),
                             callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                    experiment_name="efficientnetB0")])






plot_loss_curves(efficientnet_history)









####################################################################################################
# Creating and testing efficientnet7_model TensorFlow Hub Feature extraction model



efficientnet7_model = create_model(efficientnet7_url,
                            num_classes=train_data_10_percent.num_classes)



efficientnet_model.summary()
efficientnet7_model.summary()


# Compile our resnet model
efficientnet7_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])


#fitting

efficientnet7_history = efficientnet7_model.fit(train_data_10_percent,
                                                epochs=EPOCHS,
                                                steps_per_epoch=len(train_data_10_percent),
                                                validation_data=test_data,
                                                validation_steps=len(test_data),
                                                callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                       experiment_name="efficientnetB7")])






plot_loss_curves(efficientnet7_history)





"""
DIFFERENT TYPES OF TRANSFER LEARNING

* "As is" transfer learning - using an existing model with no changes what so ever 
(e.g using imageNet model on a 1000 ImageNet classes)

* "Feature Extaction" transfer learning - use the prelearned patterns of an existing model 
(e.g EfficientNetB0 and B7 trained on ImageNet) and adjust the output layer for ur own problem 
(e.g 1000 classes to 10mclasses of food)  

* "Fine-tuning" transfer learning - use the prelearned patterns of an existing model and "fine-tune" many or all of the
underlying layers (including new output layers)



"""

# COMPARING OUR MODELS

# Upload TensorBoard dev records

!tensorboard dev upload --logdir ./tensorflow_hub/ \
    --name "EfficientNetB0 vs, ResNet50V2" \
    --description "comparing" \
    --one_shot


# To delete an experiment
!tensorboard dev delete --experiment_id ........................e











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



pred_and_plot(model_1, "03-pizza-dad.jpeg")
pred_and_plot(model_1_augmented, "03-pizza-dad.jpeg")

"""

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
    












