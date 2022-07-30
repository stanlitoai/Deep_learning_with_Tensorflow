"""
   MILESTONE FOOD VISION
   
   
GPU compatible with mixed precision training

* K80 (not compatible)
*P100 (not compatible)
* Tesla T4 (compatible)


knowing this, in order to use mixed precision training , we need access to Tesla T4
(from within Google Colab) or if we`re using our own hardware , oour GPU needs a score of 7.0+

"""


# Importing some helper functions
from helper_functions import plot_loss_curves

from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_historys


import tensorflow_datasets as tfds


# List all available datasets

datasets_list = tfds.list_builders()
print("food101" in datasets_list)


# LOad in the data (takes 5-6 mins to download in google colab)

(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)



# Features of Food101 from TFDS
ds_info.features


#Get the classnames
class_names = ds_info.features["labels"].names

class_names

# Take one sample of the train data
train_one_sample = train_data.take(1)


# Output info about our train8ng sample
for image, label in train_one_sample:
    print(f"""
          Image shape: {image.shape}
          image datatype: {image.dtype}
          Target class from Food101 (tensor form): {label}
          class name (str from): {class_names[label.numpy()]}
          """)


# What are the min and the max of our images

import tensorflow as tf

tf.reduce_min(image), tf.reduce_max(image)



# Plot an images tensor
import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False);

# Create preprocessing functions for our data



def preprocessing_img(image, label, img_shape=224):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
        Coverts image datatype from 'unit8' --> float32 and reshapes
        image to [img_shape, img_shape, color_channel]
    label : TYPE
        DESCRIPTION.
    img_shape : TYPE, optional
        DESCRIPTION. The default is 224.

    Returns
    -------
    return (float32_image, labe) tuple

    """
    image = tf.image.resize(image, [img_shape, img_shape])
    #image = image/225.
    
    return tf.cast(image, tf.float32), label



# Preprocess a single sample img and check the outputs

preprocessing_img = preprocessing_img(image, label)[0]

print(f"image before preprocessing: \n {image[:2]}...., \nshape: {image.shape}, \nDatatype: {image.dtype}")
print(f"image after preprocessing: \n {preprocessing_img[:2]}...., \nshape: {preprocessing_img.shape}, \nDatatype: {preprocessing_img.dtype}")


# Map preprocessing function to training ( and parallelize)
train_data = train_data.map(map_func=preprocessing_img, num_parallel_cells=tf.data.AUTOTUNE)

# Shuffle train_data and turn into batches and prefetch it( load it fast)

train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)



# Map preprocessing function to test
test_data = test_data.map(map_func=preprocessing_img, num_parallel_cells=tf.data.AUTOTUNE).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)#.cache()


train_data, test_data

# Create modelling callbacks
from helper_functions import create_tensorboard_callback

checkpoint_path = "model_checkpoints/cp.ckpt"

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                      save_weights_only=True,
                                      monitor="val_accuracy",
                                      save_best_only=True)








# Setup mixed precision training
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers

mixed_precision.set_global_policy("mixed_float16")




#Build feature extraction model
# Create base model

from tensorflow.keras import applications

input_shape = (224, 224, 3)

base_model = applications.EfficientNetB0(include_top=False)
base_model.trainable = False

#Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

#x = preprocessing.Rescaling(1./255)(x)

# Add in data augmentations Sequential model as a layer
#x = data_augmentation(inputs)

# Give base_model the input (after augmentation ) and dont train it
x = base_model(inputs, training=False)

# Pool output features of the base model
x = layers.GlobalAveragePooling2D(name="Global_Average_Pooling_2D")(x)

# Put a dense layer on as the output
x = layers.Dense(101)(x)

outputs= layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)



# Make the model using the input and output
from keras import Model

model = Model(inputs, outputs)




#COmpile the model
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model.summary()


# Checking the layer type in the model

for num, layer in enumerate( model.layers):
    print(num, layer.name,layer.trainable, layer.dtype, layer.dtype_policy)


#check the dtype_policy attributes of layers in base model
for num, layer in enumerate( model.layers[1].layers[:20]):
    print(num, layer.name,layer.trainable, layer.dtype, layer.dtype_policy)



# Fit the feature extraction model
food101_all_data_history = model.fit(train_data, 
                                    epochs=3,
                                    steps_per_epoch=len(train_data),
                                    validation_data=test_data,
                                    validation_steps=int(0.15 * len(test_data)),
                                    #Track model training logs
                                    callbacks=[create_tensorboard_callback(dir_name="training_logs",
                                                                           experiment_name="food101_all_data"),
                                               checkpoint_callback]
                                    )

































































































































































































































































































































