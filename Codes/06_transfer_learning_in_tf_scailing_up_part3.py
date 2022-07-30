"""
    TRANSFER LEARNING WITH TENSORFLOW PRT3 (FOOD VISION MINI)
  WHAT WE`RE GOING TO COVER
  
* Downloading and preparing 10% of all Food101 classes (7500+ training images)

* Training a transfer learning feature extraction model

* Fine-tuning our feature extraction model( Food vision mini) to beat the original food101 paper with 
only 10% of the data

* Evaluating food vision mini`s predictions

* Finding the most wrong predictions ( on the test dataset)

* Making predictions with food vision mini on our own custom images.

"""
import helper_functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir


unzip_data("101_food_classes_10_percent.zip")

train_dir = "101_food_classes_10_percent/train/"
test_dir = "101_food_classes_10_percent/test/"



# How many images/classes are there?
walk_through_dir("101_food_classes_10_percent")


# Setup data inputs
import tensorflow as tf
from tensorflow.keras import preprocessing
IMG_SIZE = (224, 224)

train_data_all_10_percent = preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                       label_mode="categorical",
                                                                       batch_size=32,
                                                                       image_size=IMG_SIZE)

test_data = preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                      label_mode="categorical",
                                                                      batch_size=32,
                                                                      image_size=IMG_SIZE,
                                                                      shuffle=False)


"""
TRain a big dog model with transfer learning on 10% 101 food classes

Here are the steps we`re going to take:
    * Create a ModelCheckpoint callback
    * Create a data augmentation layer to build data augmentation right into our model
    * Build a headless (on top layeers) EfficientNetB0 backboned-models (we`ll createb our own output layer)
    * Compile our model
    * Feature extract for 5 full passes (5 epochs on the train data set and validate on 15% of the test data, to save epoch time)


"""

# Create checkpoint callback
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = "101_food_classes_10_percent_data_model_checkpoint"
checkpoint_path_1 = "101_food_classes_10_percent_data_model_checkpoint/cp.ckpt"

checkpoint_callback = ModelCheckpoint(checkpoint_path_1,
                                      save_weights_only=True,
                                      monitor="val_accuracy",
                                      save_best_only=True)




# Create data augmentation layer to incorporate it right into the model

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


# Setup data augmentation

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    #preprocessing.Rescaling(1./255) #keep for models like ResNet50V2 but not EfficientNet
    ], name="data_augmentation")


# Setup the base model and freeze its layers (this will extract features)

from tensorflow.keras import applications

input_shape = (224, 224, 3)

base_model = applications.EfficientNetB0(include_top=False)
base_model.trainable = False

#Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentations Sequential model as a layer
x = data_augmentation(inputs)

# Give base_model the input (after augmentation ) and dont train it
x = base_model(x, training=False)

# Pool output features of the base model
x = layers.GlobalAveragePooling2D(name="Global_Average_Pooling_2D")(x)

# Put a dense layer on as the output
outputs = layers.Dense(len(train_data_all_10_percent.class_names),
                       activation="softmax", name="output_layer")(x)



# Make the model using the input and output
from keras import Model

model = Model(inputs, outputs)


# Get a summary
model.summary()


#COmpile the model
model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# Fit the model
all_history_10_percent = model.fit(train_data_all_10_percent, 
                                epochs=5,
                                steps_per_epoch=len(train_data_all_10_percent),
                                validation_data=test_data,
                                validation_steps=int(0.25 * len(test_data)),
                                #Track model training logs
                                callbacks=[create_tensorboard_callback(dir_name="transfer_learing",
                                                                       experiment_name="all_data_10_percent_data_aug"),
                                           checkpoint_callback]
                                )





# Evaluate on the whole test dataset
all_data_10_percent_data_aug = model.evaluate(test_data)

plot_loss_curves(all_history_10_percent)


#Fine-tuning

#unfreeze all of the layers in the base model

# Are these layers trainable?

for layer in model.layers:
    print(layer, layer.trainable)


# What layers are in our base model (EfficientNetB0) and are they trainable?

for i, layer in enumerate(model.layers[2].layers):
    print(i, layer.name, layer.trainable)



# How many trainable variables are in our base model?
print(len(model.layers[2].trainable_variables))


# To begin fine-tuning, lets start by setting the last 10 layers of our base_model.trainable = True

base_model.trainable = True

# Freeze all layers except for the last 10
for layer in base_model.layers[:-5]:
    layer.trainable = False


#from tensorflow.keras.optimizers import Adam
# Recompile (we have to recompile every time we make a change)
model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #when fine-tuning, u typically want to lower ur learning rate  by 10x*
                metrics=["accuracy"])



#Fine tune for another 5 epochs
fine_tune_epochs = 10


# Fit the model
history_all_classes_10_percent_fine_tuning = model.fit(train_data_all_10_percent, 
                                epochs=fine_tune_epochs,
                                steps_per_epoch=len(train_data_all_10_percent),
                                validation_data=test_data,
                                validation_steps=int(0.15 * len(test_data)),
                                initial_epoch=all_history_10_percent.epoch[-1]
                                #Track model training logs
                                
                                )



all_classes_10_percent_fine_tune_results = model.evaluate(test_data)


plot_loss_curves(history_all_classes_10_percent_fine_tuning)

compare_historys(all_history_10_percent, 
                 history_all_classes_10_percent_fine_tuning,
                 initial_epochs=5)


result_101_food_classes_10_percent = model.evaluate(test_data)

model.summary()

#Saving our model
model.save("food_101_classes_10_percent_model", save_format='h5')
model.save("my_model.h5")

from tensorflow.keras.models import load_model
loaded_model = load_model("food_101_classes_10_percent_model")
loaded_model_1 = load_model("my_model.h5")

loaded_model.evaluate(test_data)

loaded_model_1.evaluate(test_data)
model.save("saved_trained_model_101_food")



"""
Evaluating the performance of the big dog model across all different class....
lets make ssome predictions, visualize them and then later find out which predictions 
were most wrong.....

"""
import tensorflow as tf
from tensorflow.keras.models import load_model
unzip_data("06_101_food_class_10_percent_saved_big_dog_model.zip")


model = load_model("06_101_food_class_10_percent_saved_big_dog_model")


#Evaluating the downloaded model we just downlaoded

result_downloaded_model = model.evaluate(test_data)

"""
Making predictions with our downloaded model

"""

preds_probs = model.predict(test_data, verbose=1)

preds_probs.shape


preds_probs[:10]

#WE get one prediction probs per class( in our case there`s 101 prediction probabilities)

print(f"Number of prediction probs 0: {len(preds_probs[0])}")
print(f"what pred probs sample 0 looks like : \n {preds_probs[0]}")
print(f"The class with the highest preds probs by the model for sample 0: {preds_probs[0].argmax()}")


test_data.class_names


test_data.class_names[52]


# Get pred classes of each label
pred_classes = preds_probs.argmax(axis=1)


# How do they look?
pred_classes[:10]


pred_classes[0]


# To get our test labels, we need to unravel our test_data BatchDataset

y_labels = []
for images, labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())

y_labels[:10]



"""
Evaluating our model`s predictions

one way to check that our model`s predictions array is in the same order as our test 
labels array is to find the accuracy score

"""

# Lets try sklearn`s accuracy score function

from sklearn.metrics import accuracy_score

sk_accuracy = accuracy_score(y_true=y_labels, 
                             y_pred=pred_classes)


sk_accuracy

#get let of class names
class_names = test_data.class_names

class_names[:10]


#Lets visualize: making confusion matrix
from helper_functions import make_confusion_matrix

make_confusion_matrix(y_true=y_labels,
                      y_pred=pred_classes,
                      classes=class_names,
                      figsize=(100, 100),
                      text_size=40,
                      savefig=True
                      )




"""
## Lets keep the evaluation train going, time for a classification report

sklearn has a helpful function for acquiring many different classification metrics 
per class (e.g. precision, recall and F1) called classification_repoert. 

lets try it out
"""


from sklearn.metrics import classification_report

print(classification_report(y_true=y_labels,
                            y_pred=pred_classes))




# Getting a dictionary for our classsification report

classification_report_dict = classification_report(y_true=y_labels,
                                                   y_pred=pred_classes,
                                                   output_dict=True
                                                   )


classification_report_dict


# Create empty dictionary

class_f1_scores = {}
for k, v in classification_report_dict.items():
    if k == "accuracy":
        break
    else:
        class_f1_scores[class_names[int(k)]] = v["f1-score"]

class_f1_scores


# Turn f1-scores into dataFrame for visualization

import pandas as pd

f1_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score",
                                                                                   ascending=False)




f1_scores[:10]


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(f1_scores["class_name"])
ax.set_xlabel("F1-score")
ax.set_title("F1-scores for 101 Different Food Classes (predicted by Food mini")
ax.invert_yaxis()
plt.grid();





"""
## Visualizing predictions on custom images

Now, this is the real test, how does our model go  on food images not even in our test dataset (images of our own).

To visualiz our model`s predictions on our own images, we`ll need a function


* Read in a target img filepath using tf.io.read_file()
* Turn the img into a Tensor using tf.io.decode_image()
* Resize the image tensor to be the same size as the img our model has been trained
on using tf.image.resize()
* Scale the images to get all of the pixel vales bte 0 and 1
(if necessary)

"""
# Create  a function to load and prepare images

import tensorflow as tf

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  #import tensorflow as tf
  
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img



"""
Specifically, we`ll write some code to :
    * Load a few random imgs from the dataset
    * Make predictions on the loaded imgs
    * Plot the origial img(s) alone with the model`s predictions,
    prediction probability and truth label
    
"""
# Make preds on a series of random imgs

import os 
import random
plt.figure(figsize=(13, 10))

for i in range(6):
    class_name = random.choice(class_names)
    filename = random.choice(os.listdir(test_dir + "/" + class_name))
    filepath = test_dir + class_name + "/" + filename
    
    # Load the img and make some predictions
    img = load_and_prep_image(filepath, scale=False)
    img_expanded = tf.expand_dims(img, axis=0)
    pred_prob = model.predict(img_expanded)
    pred_class = class_names[pred_prob.argmax()]
    
    # Plot the images
    plt.subplot(3, 2, i+1)
    plt.imshow(img/225.)
    
    if class_name == pred_class:
        title_color = "g"
    else:
        title_color = "r"
        
    plt.title(f"actual: {class_name}, pred: {pred_class}, probs: {pred_prob.max():.2f}", c=title_color)
    plt.axis(False)



"""
 Finding the most wrong predictions
 
To find our where our model is most wrong, lets write some code to find out the following

*1. Get all of the image file paths in the test dataset using list_file() method

*2. Create a pandas dataframe of the img filpaths, ground truth labels, pred classes
(from our model).max prediction probs, pred class names, ground truth class names.

*3. Use our Dataframe to find all the wrong predictions( where the ground truthn label doesnt match the predictions)

*4. Sort the Dataframe based on wrong predictions (have the highest prediction probs at the top)

*5.  Visualize the image with the highest predictions probs but have the wrong prediction.

"""

"""
## 1. Get all of the image file paths in the test dataset using list_file() method

"""
filepaths = []
for filepath in test_data.list_files("101_food_classes_10_percent/test/*/*.jpg",
                                     shuffle=False):
    filepaths.append(filepath.numpy())




filepaths[:10]



"""
*2. Create a pandas dataframe of the img filpaths, ground truth labels, pred classes
(from our model).max prediction probs, pred class names, ground truth class names.
"""

import pandas as pd
pred_df = pd.DataFrame({"img_path": filepaths,
                        "y_true": y_labels,
                        "y_pred": pred_classes,
                        "pred_conf": preds_probs.max(axis=1),
                        "y_true_classname": [class_names[i] for i in y_labels],
                        "y_pred_classname": [class_names[i] for i in pred_classes]
                        })


pred_df



"""
*3. Use our Dataframe to find all the wrong predictions( where the ground truthn label doesnt match the predictions)

"""

pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]

pred_df.head()




"""
*4. Sort the Dataframe based on wrong predictions (have the highest prediction probs at the top)

"""

top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:100]

top_100_wrong.head()



"""
*5.  Visualize the image with the highest predictions probs but have the wrong prediction.

"""

images_to_view = 9
start_index = 10

plt.figure(figsize=(15, 10))
for i, row in enumerate(top_100_wrong[start_index: start_index+images_to_view].itertuples()):
    plt.subplot(3, 3, i+1)
    img = load_and_prep_image(row[1], scale=False)
    _, _, _, _, pred_prob, y_true_classname, y_pred_classname, _ = row
    plt.imshow(img/225.)
    plt.title(f"actual: {y_true_classname}, pred: {y_pred_classname} \nprobs: {pred_prob}")
    plt.axis(False)






"""
Test out the big dog model on our own custom images

"""
# Get custom images

unzip_data("custom_food_images.zip")



# Get the custom food images filepaths
custom_food_images = ["custom_food_images/" + img_path for img_path in os.listdir("custom_food_images")]

custom_food_images


# Make preds on and plot custom food images

for img in custom_food_images:
    img = load_and_prep_image(img, scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[pred_prob.argmax()]
    
    #PLot the appropriate info
    plt.figure()
    plt.imshow(img/225.)
    plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
    plt.axis(False)




































