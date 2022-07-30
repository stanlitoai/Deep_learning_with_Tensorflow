#!ls pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/


# Start our experiments using the 20k dataset with numbers replaced by "@" sign
data_dr = "pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

# Check all of the filenames in the target directory

import os
filenames = [data_dr + filename for filename in  os.listdir(data_dr)]

filenames


# Preprocess Data

# Create function to read the lines of a document

def get_lines(filename):
    """

    Parameters
    ----------
    filename : TYPE = string
        Reads filename (a text filename and returns the lines of the text as a list)
        
    Args:
        filename: a string containing the target filepath.

    Returns:
        A list of strings with one string per line from the target filename.

    """
    with open(filename, "r") as f:
        
        return f.readlines()


 
# Lets read in the training lines

train_lines = get_lines(data_dr+"train.txt") # read the lines within the training files

train_lines[:50]


"""
How i think our data would be best represented .....

[{'line_number':0,
  'text': "Emotional eating is associated with overeating and the development of obesity",
  'total_lines': 11}
 .......]


# Let`s write a function which turns each of our datasets inth the above format
so we can continue to prepare our dataset

"""


def Preprocess_text_with_line_numbers(filename):
    """
    

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns a list of dictionaries of abstract line data.
    
    Takes in filename, reads it contents and sorts through each line,
    extracting things like thetarget label, the text of the sentence, how many
    sentences are in the current abstract and what sentence number the target line is
    -------
    

    """
    input_lines = get_lines(filename) # get all lines from filename
    abstract_lines = "" # create an empty abstract
    abstract_samples = [] # create an empty list of abstracts
    
    # Loop through each line in the target file
    
    for line in input_lines:
        if line.startswith("###"): # check to see if there is an ID line
            abstract_id = line
            abstract_lines = "" # reset the abstract string if the line is an ID line
            
            
        elif line.isspace(): # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines
            
            
            # Iterate through each line line in a single abstract and count them at the same time
            
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} # create an empty dictionary for ech line
                target_text_split = abstract_line.split("\t") # split target label from text
                
                line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract
                line_data["target"] = target_text_split[0] # get trget label
                line_data["text"] = target_text_split[1].lower() #get target text and lower it
                line_data["total_lines"] = len( abstract_line_split) -1 # how many total lines are there in the target abstract? (start from 0)
                
                abstract_samples.append(line_data) # add line data to abstract samples list
                
        else: # if the above conditions aren`t fulfilled, the line contains a labelled sentence
            abstract_lines += line
            
    return abstract_samples
            
        

 

#Get data from file and preprocess it

#%%time

train_samples = Preprocess_text_with_line_numbers(data_dr+"train.txt")
val_samples = Preprocess_text_with_line_numbers(data_dr+"dev.txt")
test_samples = Preprocess_text_with_line_numbers(data_dr+"test.txt")

print(len(train_samples), len(val_samples), len(test_samples))


# check the first abstarct of our training data
train_samples[:50]



"""
Now taht our data is in the format of list of dictionaries, how about us turn it into a DataFrame to further visulaze it

"""
import pandas as pd
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)


train_df.head(14)

# Distribution of labels intraining data

train_df.target.value_counts()

# Lets check the lenght of different lines

train_df.total_lines.plot.hist();


# Get lists of sentences 



## Convert abstract text lines into lists

train_sent = train_df["text"].tolist()
val_sent = val_df["text"].tolist()
test_sent = test_df["text"].tolist()


train_sent[:10]



import sklearn
print(sklearn.__version__)

# One hot encode labels
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse=False)

train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))




train_labels_one_hot

# Extract labels ("target" columns) and encode them into integers

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())


train_labels_encoded


# Get class names and number of classes from labelEncoder instance
num_classes = len(label_encoder.classes_)


class_names = label_encoder.classes_


num_classes, class_names

"""

0. model_0 = Naive Bayes with TF-IDF encoder (baseline)
1. model_1 = Conv1D with token embedding
2. model_2 = TensorFlow Hub pretrained Feature Extractor
3. model_3 = Conv1D with character embeddings
4. model_4 = Pretrained token embeddings (same as 2)+ character embedding(same as 3)
5. modeel_5 = Pretrained token embeddings (same as 2)+ character embedding(same as 3)+positional embeddings

"""

"""
0. model_0 = Naive Bayes with TF-IDF encoder (baseline)

"""
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline




# Create tokenization and modelling pipline

model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()), # convert words to numbers using tfidf
    ("clf", MultinomialNB()) # Model the text
    
    ])



# Fit the pipline to the training data
model_0.fit(train_sent, train_labels_encoded)


#Evaluate our baseline model

baseline_score = model_0.score(val_sent, val_labels_encoded)

print(f"Our baseline model achieve an accuracy of: {baseline_score*100:.2f}%")





#MAke predictions

baseline_preds = model_0.predict(val_sent)

baseline_preds[:20]
val_labels_encoded[:20]





"""

## Creating an evaluation function for our model experiments

#Let`s create a function to compare our model`s predictions with the truth labels using the following metrics:
    * Accuracy
    * Precision
    * Recall
    * F1-score


"""


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    

    Parameters
    Caluates model accuracy, precision, recall and f1-score of a binary classification model.
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Calculate model accuracy
    model_accuarcy = accuracy_score(y_true, y_pred)*100
    
    # calculate model precision, recall and f1-score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    model_results = {"accuracy": model_accuarcy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1
                     }
    
    return model_results





# Get baseline results

baseline_results= calculate_results(y_true=val_labels_encoded,
                                    y_pred=baseline_preds)



baseline_results





"""
1. model_1 = Conv1D with token embedding

68k
331k
"""
import numpy as np

sent_lens = [len(sentence.split()) for sentence in train_sent]
avg_sent_len = np.mean(sent_lens)
avg_sent_len

import matplotlib.pyplot as plt
plt.hist(sent_lens, bins=20);


# How long of sentence lenght covers 95% of examples?
output_seq_len = int(np.percentile(sent_lens, 95))

output_seq_len


# Text vectorization (tokenization)

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

max_tokens=68000

# Using the default Text Vectorization parameter

text_vect = TextVectorization(max_tokens=max_tokens,
                              ngrams=None, # cfreate group of n-word
                              output_mode="int",
                              output_sequence_length=output_seq_len,
                              pad_to_max_tokens=False
                              )



# Find the average number of tokens (words) in the training tweets
round(sum([len(i.split()) for i in train_sent])/len(train_sent))




# Fit the text vectorizer to the training text

text_vect.adapt(train_sent)



# Choose a random sentense from the training dataset and tokenize it
import random
random_sentence = random.choice(train_sent)

print(f"Original text: \n {random_sentence}\
      \n\n Vectorized version:")

text_vect([random_sentence])




# Get the unique words in the vocabulary

rct_20k_text_vocab = text_vect.get_vocabulary() # get all of the unique words in  our training data

top_5_words = rct_20k_text_vocab[:5] # get the most common words

bottom_5_words = rct_20k_text_vocab[-5:] # get the least common words

print(f"Number of words in vocab: {len(rct_20k_text_vocab)}")
print(f"5 most common words in vocab: {top_5_words}")
print(f"5 least common words in vocab: {bottom_5_words}")



# Get the config of our text vectorizer
text_vect.get_config()





"""
# Creating and embedding layer

THe parameters we care most about for our embedding layer:

* "input_dim" = the size of our vocabulary
* "output_dim" = the size of the output embedding vector, for example, a value 
of 100 would mean each token gets represented by a vector 100 long
* "input_length" =  length of the sequence being passed to embedding layers

"""

from tensorflow.keras.layers import Embedding

embedding = Embedding(input_dim=len(rct_20k_text_vocab), #input shape
                      output_dim=128, # output shape
                      mask_zero=True
                      )



#embedding(text_vect(train_sent))

# Choose a random sentense from the training dataset and tokenize it

random_sentence = random.choice(train_sent)

print(f"Original text: \n {random_sentence}\
      \n\n Embedded version:")

# Embed the random sentence (turn it into dense vectors of fixed size)
sample_embed = embedding(text_vect([random_sentence]))

sample_embed


###################
# Creating datatsets (making sure our data loads as fast as possible)

# Turn our data into Tensorflow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sent, train_labels_one_hot))
val_dataset = tf.data.Dataset.from_tensor_slices((val_sent, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sent, test_labels_one_hot))


train_dataset

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset




"""

from tensorflow.keras import layers
embedding_test = embedding(text_vect(["this is a test sentence"])) #turn target sequence into embedding

conv_1d = layers.Conv1D(filters=32,
                        kernel_size=5,
                        activation="relu",
                        padding="valid")


conv_1d_output = conv_1d(embedding_test) # pass test embedding through conv1d layer
max_pool = layers.GlobalMaxPool1D()

max_pool_output = max_pool(conv_1d_output) # equivqlent to "get the most important feature" or "get the featuire with the heighest value"


embedding_test.shape, conv_1d_output.shape, max_pool_output.shape


"""

"""
1. model_1 = Conv1D with token embedding

"""



#Create 1-dimensional convolutional layer to model sequences
from  tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")

text_vector = text_vect(inputs) # turn the input text to numbers

token_embeddings = embedding(text_vector) # Create an embedding of the numberized inputs
#print(x.shape)
x = layers.Conv1D(64, 5, activation="relu", padding="same")(token_embeddings)
x = layers.GlobalAveragePooling1D()(x)

outputs = layers.Dense(num_classes, activation="softmax")(x) # Out put layer

model_1 = tf.keras.Model(inputs, outputs, name="model_1_Conv1D")




#compile
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_1.summary() 

from helper_functions import create_tensorboard_callback

save_dir = "skimlit_logs"

# Fit the model


history_model_1 = model_1.fit(train_dataset,
                              steps_per_epoch=int(0.1*len(train_dataset)),
                              epochs=5,
                              validation_data=val_dataset,
                              validation_steps=int(0.1* len(val_dataset)),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_1_conv1D")])





# Check the results
model_1.evaluate(val_dataset)

# Make some predictions and evaluate those

model_1_preds_probs = model_1.predict(val_dataset)

model_1_preds_probs[:10]
model_1_preds_probs.shape

# Convert ped probs to classes
model_1_preds = tf.argmax(model_1_preds_probs, axis=1)


class_names
model_1_preds[:10]



# Calculate our model_1 results

model_1_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_1_preds)


model_1_results

baseline_results



#%%%

"""

2. model_2 = TensorFlow Hub pretrained Feature Extractor

#####################
Now lets use pretrained word embeddings from tensorflow Hub, more 
specifically the universal sentence encoder:
    https://tfhub.dev/google/universal-sentence-encoder/4

you can also check out HuggingFace.co/models


"""


import tensorflow_hub as hub
tf_hub_embed_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                       trainable=False,
                       name="universal-sentence-encoder")


#Test out the pretrained embedding on a random sentence

random_train_sent = random.choice(train_sent)
print(f"Random sentence: \n {random_train_sent}")

use_embed_sent = tf_hub_embed_layer([random_train_sent])
print(f"Sentence after embedded: \n {use_embed_sent[0][:30]} \n")
print(f"Length of sentence embedding: {len(use_embed_sent[0])}")





######################################

# Create a keras layer using the USE pretrained layer from tensorflow hub

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="USE")



# Create model usingf the Sequential API
from tensorflow.keras import Sequential

model_2 =  Sequential([
    sentence_encoder_layer,
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")],
    name="model_2_USE"
    )

###################################################3




################################
############OR
#################################

inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_embedding = tf_hub_embed_layer(inputs)
x = layers.Dense(128, activation="relu")(pretrained_embedding)
outputs = layers.Dense(len(class_names), activation="softmax")(x)


model_2 = tf.keras.Model(inputs, outputs, name="model_2_USE")




##################################################






model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_2.summary() 


# Fit the model

history_model_2 = model_2.fit(train_dataset,
                              steps_per_epoch=int(0.1*len(train_dataset)),
                              epochs=5,
                              validation_data=val_dataset,
                              validation_steps=int(0.1* len(val_dataset)),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_2_tf_hub_pretrained")])





# Check the results
model_2.evaluate(val_dataset)

# Make some predictions and evaluate those

model_2_preds_probs = model_2.predict(val_dataset)

model_2_preds_probs[:10]
model_2_preds_probs.shape

# Convert ped probs to classes
model_2_preds = tf.argmax(model_2_preds_probs, axis=1)


class_names
model_2_preds[:10]
val_labels[:10]



# Calculate our model_1 results

model_2_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_2_preds)


model_2_results

baseline_results





#%%%%%%%


"""

3. model_3 = Conv1D with character embeddings

Creating a character-level tokenizer


"""

train_sent[:5]




# Make function to split sentences into character
def split_chars(text):
    return " ".join(list(text))


#" ".join(list(train_sent[5]))

split_chars(random_train_sent)


# Split sequence-level data splits into character-level data splits

train_chars = [split_chars(sent) for sent in train_sent]
val_chars = [split_chars(sent) for sent in val_sent]
test_chars = [split_chars(sent) for sent in test_sent]

train_chars[:5]


# Whats the average character lenght
char_lens = [len(sent) for sent in train_sent]
mean_char_len = np.mean(char_lens)

mean_char_len

#Check the distribution of our squences at a character-level
import matplotlib.pyplot as plt

plt.hist(char_lens, bins=20)


# Find wht character4 lenght covers 95% of squences
output_seq_len = int(np.percentile(char_lens, 95))

output_seq_len



#Get all keyboards characters
import string

alph = string.ascii_lowercase + string.digits + string.punctuation

alph





# Create char-level token vectorizer instance



# Text vectorization (tokenization)

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

NUM_CHAR_TOKENS = len(alph) + 2

# Using the default Text Vectorization parameter

char_vect = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                              output_sequence_length=output_seq_len,
                              name="char_vectorizer"
                              )



# Fit the text vectorizer to the training text

char_vect.adapt(train_chars)



# Check character vocab stats
char_vocab = char_vect.get_vocabulary()
print(f"Number of different characters in character vocab: {len(char_vocab)}")
print(f"5 most common characters: {char_vocab[:5]}")
print(f"5 least common characters: {char_vocab[-5]}")







# Choose a random sentense from the training dataset and tokenize it
import random
random_train_chars = random.choice(train_chars)

print(f"charified text: \n {random_train_chars}")
print(f"\n Length of random_train_chars: {len(random_train_chars.split())}")
vect_chars = char_vect([random_train_chars])
print(f"\n vectorized chars:\n {vect_chars}")
print(f"\nLength of vectorized chars: {len(vect_chars[0])}")




#######character embedding

from tensorflow.keras.layers import Embedding

char_embedding = Embedding(input_dim=len(char_vocab), #input shape
                      output_dim=25, # output shape
                      mask_zero=True,
                      name="char_embed"
                      )



#embedding(text_vect(train_sent))

# Choose a random sentense from the training dataset and tokenize it

random_train_chars

print(f"charified text: \n {random_train_chars}\n")
      
# Embed the random sentence (turn it into dense vectors of fixed size)
sample_embed = char_embedding(char_vect([random_train_chars]))
print(f" Embedded version: \n {sample_embed}")

sample_embed.shape






#Create 1-dimensional convolutional layer to model sequences
from  tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")

char_vectorizer = char_vect(inputs) # turn the input text to numbers

char_embeddings = char_embedding(char_vectorizer) # Create an embedding of the numberized inputs
#print(x.shape)
x = layers.Conv1D(64, 5, activation="relu", padding="same")(char_embeddings)
#x = layers.GlobalAveragePooling1D()(x)
x = layers.GlobalMaxPool1D()(x)

outputs = layers.Dense(num_classes, activation="softmax")(x) # Out put layer

model_3 = tf.keras.Model(inputs, outputs, name="model_3_Conv1D")




#compile
model_3.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_3.summary() 



# Creating datatsets (making sure our data loads as fast as possible)

# Turn our data into Tensorflow Datasets
train_char_dataset = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot))
val_char_dataset = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot))
test_char_dataset = tf.data.Dataset.from_tensor_slices((test_chars, test_labels_one_hot))


train_char_dataset = train_char_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_char_dataset = val_char_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_char_dataset = test_char_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


train_char_dataset

train_dataset


from helper_functions import create_tensorboard_callback

save_dir = "skimlit_logs"

# Fit the model


history_model_3 = model_3.fit(train_char_dataset ,
                              steps_per_epoch=int(0.1*len(train_char_dataset)),
                              epochs=5,
                              validation_data=val_char_dataset,
                              validation_steps=int(0.1* len(val_char_dataset)),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_3_conv1D")])





# Check the results
model_3.evaluate(val_dataset)

# Make some predictions and evaluate those

model_3_preds_probs = model_3.predict(val_char_dataset)

model_3_preds_probs[:10]
model_3_preds_probs.shape

# Convert ped probs to classes
model_3_preds = tf.argmax(model_3_preds_probs, axis=1)


class_names
model_3_preds[:10]


# Calculate our model_1 results

model_3_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_3_preds)


model_3_results

baseline_results


#%%%%%%

"""

4. model_4 = Pretrained token embeddings (same as 2)+ character embedding(same as 3)


1. create a token-level embedding model (same as model_2)
2. crteate a character-level model (same as model_3)
3. combine 1 & 2 with a concatenate("layers.concatenate")
4. Build a series of output layers ontop of 3 similiar bto figure 1and section 4.2 of the paper
5. Construct a model which takes token and character-level sequences as input and produces
   sequence label probabilities as output
   


"""

# 1. Setup token input/model

token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embed_layer(token_inputs)
token_output = layers.Dense(128, activation="relu")(token_embeddings)

token_model = tf.keras.Model(token_inputs,
                             token_output)




#2. Setup car input/model
char_inputs = layers.Input(shape=[], dtype=tf.string, name="char_input")
char_vects = char_vect(char_inputs)
char_embeddings = char_embedding(char_vects)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)

char_model = tf.keras.Model(token_inputs,
                             token_output)



#3. concatenate token and char inputs (create hybride token embedding)
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output, 
                                                                  char_model.output])


#4. Create output layers -adding in Dropout, similiar to figure 1and section 4.2 of the paper

combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(128, activation="relu")(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)



#5. Construct model with char and token inputs
from tensorflow.keras import Model
model_4 = Model(inputs=[token_model, char_model],
                outputs=output_layer,
                name="model_4_token_and_char_embeddings")



model_4.summary() 



# Plot hybrid token and character model
from tensorflow.keras.utils import plot_model

plot_model(model_3,
           show_shapes=True,
           show_layer_names=True,
           show_dtype=True,
           to_file="model_3.png")










#compile
model_4.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])





# Combining token and character data into a tf.data Dataset
# Combine chars and tokens into a dataset\n",
train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sent, train_chars)) # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels)) # combine data and labels

# Prefetch and batch train data
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# Repeat same steps validation data
val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sent, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)



# Check out training char and token embedding dataset
train_char_token_dataset, val_char_token_dataset




history_model_4 = model_4.fit(train_char_token_dataset,
                              steps_per_epoch=int(0.1*len(train_char_token_dataset)),
                              epochs=5,
                              validation_data=val_char_token_dataset,
                              validation_steps=int(0.1* len(val_char_token_dataset)),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_3_conv1D")])





# Check the results
model_4.evaluate(val_char_token_dataset)

# Make some predictions and evaluate those

model_4_preds_probs = model_4.predict(val_char_token_dataset)

model_4_preds_probs[:10]
model_4_preds_probs.shape

# Convert ped probs to classes
model_4_preds = tf.argmax(model_4_preds_probs, axis=1)


class_names
model_4_preds[:10]


# Calculate our model_1 results

model_4_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_4_preds)


model_4_results


#%time
baseline_results



#%%
"""
5. modeel_5 = Pretrained token embeddings (same as 2)+ character embedding(same as 3)+positional embeddings

NOTE: Any engineering features used to train a model need to be available at test time.
In our case, line numbers and totle lines are available

"""

train_samples[:]


train_df.head()


# Create positional embeddings

#How many different line number are there

train_df["line_number"].value_counts()

# Check the distribution of "line_number column

train_df.line_number.plot.hist()



# Use Tensorflow to create one-hot-encoded tensors of our "line_number" column

train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

train_line_numbers_one_hot[:10], train_line_numbers_one_hot.shape





#How many different  numbers of line are there

train_df["total_lines"].value_counts()

# Check the distribution of "line_number column

train_df.total_lines.plot.hist()

# Check the coverage of a "totle_line" vslue of 20
np.percentile(train_df.total_lines, 95)




# Use Tensorflow to create one-hot-encoded tensors of our "line_number" column

train_total_line_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_line_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_line_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

train_total_line_one_hot[:10], train_total_line_one_hot.shape




"""
### Building a tribrid embedding model

1. Create a token-level model
2. Create a character-level model
3. Create a model for "line-number" feature
4. Create a model for "total_lines" feature
5. Combine the outputs 1 & 2 using tf.keras.layers.concatenate
6. Combine the outputs 3,4,5 using tf.keras.layers.concatenate
7. Create an output layer to accept the tribried embedding and output label probabilities
8. Combine the inputs of 1,2,3,4 and outputs of 7 into a tf.keras.Model


"""




# 1. Setup token input/model

token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embed_layer(token_inputs)
token_output = layers.Dense(128, activation="relu")(token_embeddings)

token_model = tf.keras.Model(token_inputs,
                             token_output)




#2. Setup char input/model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vects = char_vect(char_inputs)
char_embeddings = char_embedding(char_vects)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)

char_model = tf.keras.Model(char_inputs,
                             char_bi_lstm)




#3. Create a model for "line-number" feature

from  tensorflow.keras import layers

line_num_inputs = layers.Input(shape=(15,), dtype=tf.float32, 
                              name="line_number_input")

x = layers.Dense(32, activation="relu")(line_num_inputs)
#x = layers.GlobalMaxPool1D()(x)
line_num_model = tf.keras.Model(line_num_inputs, x)






#4. Create a model for "total_lines" feature

from  tensorflow.keras import layers

total_line_inputs = layers.Input(shape=(20,), dtype=tf.float32, 
                              name="total_line_input")

y = layers.Dense(32, activation="relu")(total_line_inputs)
#x = layers.GlobalMaxPool1D()(x)
total_line_model = tf.keras.Model(line_num_inputs, y)





#5. Combine the outputs 1 & 2 using tf.keras.layers.concatenate

combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output, 
                                                                  char_model.output])

z = layers.Dense(256, activation="relu")(combined_embeddings)
z = layers.Dropout(0.5)(z)

#6. Combine the outputs 3,4,5 using tf.keras.layers.concatenate
tribrid_embeddings = layers.Concatenate(name="char_token_positional_embedding")([line_num_model.output, 
                                                                  total_line_model.output,
                                                                  z])


#7. Create an output layer to accept the tribrid embedding and output label probabilities
output_layer = layers.Dense(5, activation="softmax", name="output_layer")(tribrid_embeddings)


#8. Combine the inputs of 1,2,3,4 and outputs of 7 into a tf.keras.Model

from tensorflow.keras import Model
model_5 = Model(inputs=[line_num_model.input,
                        total_line_model.input,
                        token_model.input, 
                        char_model.input],
                outputs=output_layer,
                name="model_5_tribrid_embedding_model")






# Compile token,char and positional embedding model
model_5.compile(loss=tf.keras.losses.categorical_crossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])











# Create training and valiadation datasets (with all four kinds of input data)
train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                               train_total_line_one_hot,
                                                                train_sent,
                                                                train_chars))

train_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)

train_char_token_pos_dataset = tf.data.Dataset.zip((train_char_token_pos_data, train_char_token_pos_labels))

train_char_token_pos_dataset = train_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Do the same as above but for the validation dataset
val_char_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_line_one_hot,
                                                              val_sent,
                                                              val_chars))

val_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)

val_char_token_pos_dataset = tf.data.Dataset.zip((val_char_token_pos_data, val_char_token_pos_labels))

val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
      




## Fitting, evaluating and making predicitions with our tribrid model

# Fit our tribrid embedding model
history_model_5 = model_5.fit(train_char_token_pos_dataset,
                              
                              steps_per_epoch=int(0.1 * len(train_char_token_pos_dataset)),
                              epochs=3,
                              validation_data=val_char_token_pos_dataset,
                              validation_steps=int(0.1 * len(val_char_token_pos_dataset)))




# Make predictions with the char token pos model
model_5_pred_probs = model_5.predict(val_char_token_pos_dataset, verbose=1)
model_5_pred_probs


# Convert pred probs to pred labels
model_5_preds = tf.argmax(model_5_pred_probs, axis=1)
model_5_preds



# Calculate results of char token pos model
model_5_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_5_preds)
model_5_results










## Compare model results

# Combine model results into a dataframe
all_model_results = pd.DataFrame({"model_0_baseline": baseline_results,
                                  "model_1_custom_token_embedding": model_1_results,
                                  "model_2_pretrained_token_embedding": model_2_results,
                                  "model_3_custom_char_embedding": model_3_results,
                                  "model_4_hybrid_char_token_embedding": model_4_results,
                                  "model_5_pos_char_token_embedding": model_5_results})

all_model_results = all_model_results.transpose()
all_model_results






# Reduce the accuracy to same scale as other metrics
all_model_results["accuracy"] = all_model_results["accuracy"]/100



# Plot and compare all model results\n",
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))



# Sort models results by f1-score\n",
all_model_results.sort_values("f1", ascending=True)["f1"].plot(kind="bar", figsize=(10, 7));


# Save the best performing model to SavedModel format (default)\n",
model_5.save("skimlit_tribrid_model")


#Load in best performing model
import tensorflow as tf

loaded_model_5 = tf.keras.models.load_model("skimlit_tribrid_model")


loaded_model_5.summary()



# Make predictions with the loaded moel on the validation set
loaded_pred_probs = loaded_model_5.predict(val_char_token_pos_dataset)
loaded_preds = tf.argmax(loaded_pred_probs, axis=1)
loaded_preds[:10]


# Calculate the results of our loaded model
loaded_model_results = calculate_results(y_true=val_labels_encoded,
                                         y_pred=loaded_preds)
loaded_model_results


assert model_5_results == loaded_model_results

loaded_model_5.summary()




# Plot hybrid token and character model
from tensorflow.keras.utils import plot_model

plot_model(model_3,
           show_shapes=True,
           show_layer_names=True,
           show_dtype=True,
           to_file="model_3.png")



#%%











import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

!wget https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
!mkdir skimlit_gs_model
!unzip skimlit_tribrid_model.zip -d skimlit_gs_model






# Load in downloaded Google Storage model
loaded_gs_model = tf.keras.models.load_model("skimlit_tribrid_model/")


# Evalaute the loaded Google Storage model
loaded_gs_model.evaluate(val_char_token_pos_dataset)





"""

## Your challenges\n",

"Try these out before moving onto the next module.\n",

"The GitHub has an example of how to do each but I'd encourage you to try it for yourself first (you've got this!).\n",

"1. Turn the test data samples into a tf.data Dataset (fast loading) and then evaluate (make predictions) the best performing model on the test samples.
"2. Find the most wrong predictions from 1 (these are the samples where the model has predicted the wrong label with the highest prediction probability).
"3. Make example predictions (on RCT abstracts from the wild), you can go to PubMed to find these: https://pubmed.ncbi.nlm.nih.gov/, find and use our model to predict on 3-4 different abstracts from the wild.
"  * Some examples: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/skimlit_example_abstracts.json 
"4. See the exercises and extra curriculum section on GitHub to further test your skills (for section 09)

"> See the full course materials (including an example of how to do the above) on GitHub: https://github.com/mrdbourke/tensorflow-deep-learning"




"""










































































