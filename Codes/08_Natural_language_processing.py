"""
WHAT WE`RE GOING TO COVER

* Downloading and preparing a text dataset
* How to prepare text data for modelling (tokenization and embedding)
* Setting up multiple modelling experiments with recurrent neural networks (RNNs)
* Building a text feature extraction model using tensorflow Hub
* Finding the most wrong prediction examples
* Using a model we`ve built to make predictions on text from the wild

              LET`S CODE
              
Introduction to NLP fundamentals in tensorflow

NLP has the goal of deriving information out of natural language (could be sequences text or speech).

Anothor common term NLP robs is sequence probs (seq2seq)              
              
"""


from helper_functions import plot_loss_curves, unzip_data, create_tensorboard_callback, compare_historys

"""
#3  Get a text dataset

The dataset w`re be using will kaggle`s introduction to NLP dataset
 (text sample of Tweets labelled as disaster or not disaster)


"""

#unzip_data("nlp_getting_started.zip")

import pandas as pd

train_df = pd.read_csv("nlp_getting_started/train.csv")
test_df = pd.read_csv("nlp_getting_started/test.csv")

train_df.head()


# Shuffle training dataframe

train_df_shuffled = train_df.sample(frac=1, random_state=42)

train_df_shuffled.head()

#How many examples of each class?

train_df.target.value_counts()

test_df.head()

# How many total sample
len(train_df), len(test_df)


# Lets visualize the data

import random
random_index = random.randint(0, len(train_df)-5)

for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
    print(f"Text: \n{text}\n")
    print("-----\n")




### Split data into training and validation
from sklearn.model_selection import train_test_split

train_sent, val_sent, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                  train_df_shuffled["target"].to_numpy(),
                                                                  test_size=0.1,
                                                                  random_state=42)




len(train_sent),len(val_sent), len(train_labels), len(val_labels)


## Visualize

train_sent[:10], train_labels[:10]


"""
## Convert text to numbers

When dealing with a text probs, one of the fist things u`ll have to do before u can build a model
is to convert ur text to number....

There are a few ways to do this, namely:
    * Tokenziation - direct mapping of token (a token could be a word or a character) to number.
    
    * Embedding - create a matrix of feature vector for each token (the size of the feature vector
      can be defines and this embadding can be leaarned)




"""

# Text vectorization (tokenization)

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Using the default Text Vectorization parameter

text_vect = TextVectorization(max_tokens=None,
                              standardize="lower_and_strip_punctuation",
                              split="whitespace",
                              ngrams=None, # cfreate group of n-word
                              output_mode="int",
                              output_sequence_length=None,
                              pad_to_max_tokens=False
                              )



# Find the average number of tokens (words) in the training tweets
round(sum([len(i.split()) for i in train_sent])/len(train_sent))


# Setup Text vectorization variable

max_vocab_length = 10000
max_length = 15


text_vect = TextVectorization(max_tokens=max_vocab_length,
                            output_mode="int",
                            output_sequence_length=max_length)



# Fit the text vectorizer to the training text

text_vect.adapt(train_sent)


# Create a sample sentence and tokenize it
sample_sent = "There`s a flood in my street"

text_vect([sample_sent])


# Choose a random sentense from the training dataset and tokenize it

random_sentence = random.choice(train_sent)

print(f"Original text: \n {random_sentence}\
      \n\n Vectorized version:")

text_vect([random_sentence])




# Get the unique words in the vocabulary

words_in_vocab = text_vect.get_vocabulary() # get all of the unique words in  our training data

top_5_words = words_in_vocab[:5] # get the most common words

bottom_5_words = words_in_vocab[-5:] # get the least common words

print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"5 most common words in vocab: {top_5_words}")
print(f"5 least common words in vocab: {bottom_5_words}")




"""
# Creating and embedding layer

THe parameters we care most about for our embedding layer:

* "input_dim" = the size of our vocabulary
* "output_dim" = the size of the output embedding vector, for example, a value 
of 100 would mean each token gets represented by a vector 100 long
* "input_length" =  length of the sequence being passed to embedding layers

"""

from tensorflow.keras.layers import Embedding

embedding = Embedding(input_dim=max_vocab_length, #input shape
                      output_dim=128, # output shape
                      embeddings_initializer="uniform",
                      input_length=max_length  # how long is each input
                      )



#embedding(text_vect(train_sent))

# Choose a random sentense from the training dataset and tokenize it

random_sentence = random.choice(train_sent)

print(f"Original text: \n {random_sentence}\
      \n\n Embedded version:")

# Embed the random sentence (turn it into dense vectors of fixed size)
sample_embed = embedding(text_vect([random_sentence]))

sample_embed




"""
 Modelling a text dataset (running a series of experiments)

Now we`ve a got way to turn our text sequences into numbers, its time to start building
a series of modelling experiments.


we`ll start a baseline and move on from there

* Model 0: Naive Bayes (baseline).
* Model 1: Feed-forward neural network (dense model).
* Model 2: LSTM model (RNN).
* Model 3: GRU model (RNN).
* Model 4: Bidirectional-LSTM model (RNN).
* Model 5: 1D Convolutional Neural Network (CNN).
* Model 6: TensorFlow Hub Pretrained Extractor (using transfer learning for NLP).
* Model 7: same as model 6 with 10% of training data.



How are we going to approach all of these?

Use the standard stteps in modelling with tensorflow:
    * Create the model
    * Build a model
    * Fit a model
    * Evaluate our model





"""


"""

* Model 0: Naive Bayes (baseline).

To create our baseline, we`ll use Sklearn`s Multinomial Naive Bayes using the TF-IDF formula to convert our words to numbers


NOTE: its common pratices to use non-DL algorithms as a baseline  because of their speed
and then later use DL to see if you can improve upon them

"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Create tokenization and modelling pipline

model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
    ("clf", MultinomialNB()) # Model the text
    
    ])



# Fit the pipline to the training data
model_0.fit(train_sent, train_labels)


#Evaluate our baseline model

baseline_score = model_0.score(val_sent, val_labels)

print(f"Our baseline model achieve an accuracy of: {baseline_score*100:.2f}%")





#MAke predictions

baseline_preds = model_0.predict(val_sent)

baseline_preds[:20]
val_labels[:20]


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

baseline_results= calculate_results(y_true=val_labels,
                                    y_pred=baseline_preds)



baseline_results




"""

* Model 1: Feed-forward neural network (dense model).

""" 

# Create a Tensorboard callback

from helper_functions import create_tensorboard_callback

save_dir = "model_logs"


# Build model with the Functional API

from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype=tf.string)

x = text_vect(inputs) # turn the input text to numbers

x = embedding(x) # Create an embedding of the numberized inputs
x = layers.GlobalAveragePooling1D()(x) #codense the feature vector for each token to one vector
#x = layers.GlobalMaxPool1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x) # Out put layer

model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")



model_1.summary()

# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



# Fit the model
model_1_history = model_1.fit(x=train_sent,
                              y=train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=save_dir,
                                                                     experiment_name="model_1_dense")]
                              )





# Check the results
model_1.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_1_preds_probs = model_1.predict(val_sent)

model_1_preds = tf.squeeze(tf.round(model_1_preds_probs))



model_1_preds


# Calculate our model_1 results

model_1_results = calculate_results(y_true=val_labels,
                                    y_pred=model_1_preds)




baseline_results


# Visualizing our model`s learned word embeddings with TensorFlow

# Get the vocabulary from the text vectorization layer

words_in_vocab = text_vect.get_vocabulary()

len(words_in_vocab), words_in_vocab[:10]



model_1.summary()




# Get the weight matrix of embedding layer
# (these are the numerical representations of each token in our training data, wch have been learned for 5_epochs)

embed_weights = model_1.get_layer("embedding").get_weights()[0]
print(embed_weights.shape)



# Create embedding files (we got this from tf word embedding doce)

import io
out_v = io.open("vectors.tvs", 'w', encoding='utf-8')
out_m = io.open("metadata.tvs", 'w', encoding='utf-8')

for index, word in enumerate(words_in_vocab):
    if index == 0:
        continue
    
    vec = embed_weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')
    
out_v.close()
out_m.close()


# Donwload files from Colab to upload to projector

try:
    from google.colab import files
    files.download("vectors.tsv")
    files.download("metadata.tsv")
    
except Exception:
    pass




"""
* Model 2: LSTM model (RNN).

LSTM = long short term memory (one of the most popular LSTM cells)

Our structure of an RNN typically looks like this:

    Input(text) --> Tokenize --> Embedding --> Layers (RNNs/Dense) --> Output (label probs)
"""

# Create an LSTM model

from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")

x = text_vect(inputs) # turn the input text to numbers

x = embedding(x) # Create an embedding of the numberized inputs
#print(x.shape)
#x = layers.LSTM(64, return_sequences=True)(x)
#print(x.shape)
x = layers.LSTM(64)(x)
#print(x.shape)
#x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x) # Out put layer

model_2 = tf.keras.Model(inputs, outputs, name="model_2_LSTM")


model_2.summary() 


# Compile the model

model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_2_history = model_2.fit(train_sent,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_2_LSTM")])






# Check the results
model_2.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_2_preds_probs = model_2.predict(val_sent)

model_2_preds_probs[:10]

model_2_preds = tf.squeeze(tf.round(model_2_preds_probs))



model_2_preds[:10]
val_labels[:10]





# Calculate our model_1 results

model_2_results = calculate_results(y_true=val_labels,
                                    y_pred=model_2_preds)


model_2_results

baseline_results




"""
* Model 3: GRU model (RNN).


"""

# Build an RNN using the GRU cell

from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")

x = text_vect(inputs) # turn the input text to numbers

x = embedding(x) # Create an embedding of the numberized inputs
#print(x.shape)
x = layers.GRU(64)(x)
"""
print(x.shape)
x = layers.LSTM(42, return_sequences=True)(x)
print(x.shape)
x = layers.GRU(99)(x)
print(x.shape)
x = layers.Dense(64, activation="relu")(x)
"""
#x = layers.GlobalAveragePooling1D()(x) #codense the feature vector for each token to one vector
outputs = layers.Dense(1, activation="sigmoid")(x) # Out put layer

model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")


model_3.summary() 






# Compile the model

model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_3_history = model_3.fit(train_sent,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_3_GRU")])






# Check the results
model_3.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_3_preds_probs = model_3.predict(val_sent)

model_3_preds_probs[:10]

model_3_preds = tf.squeeze(tf.round(model_3_preds_probs))



model_3_preds[:10]
val_labels[:10]





# Calculate our model_1 results

model_3_results = calculate_results(y_true=val_labels,
                                    y_pred=model_3_preds)


model_3_results

baseline_results








"""
* Model 4: Bidirectional-LSTM model (RNN).

Normal RNN`s go from left to right (just like you`d read an ENglish sentence) 
however, a bidirectional RNN goes from right to left and from left to right

"""

# Build Bidirectional-LSTM

from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")

x = text_vect(inputs) # turn the input text to numbers

x = embedding(x) # Create an embedding of the numberized inputs
#print(x.shape)
x = layers.Bidirectional(layers.LSTM(64))(x)
#x = layers.Bidirectional(layers.GRU(64))(x)

outputs = layers.Dense(1, activation="sigmoid")(x) # Out put layer

model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")


model_4.summary() 






# Compile the model

model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_4_history = model_4.fit(train_sent,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_4_Bidirectional_LSTM")])






# Check the results
model_4.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_4_preds_probs = model_4.predict(val_sent)

model_4_preds_probs[:10]

model_4_preds = tf.squeeze(tf.round(model_4_preds_probs))



model_4_preds[:10]
val_labels[:10]





# Calculate our model_1 results

model_4_results = calculate_results(y_true=val_labels,
                                    y_pred=model_4_preds)


model_4_results

baseline_results



"""
## Convolution Neural Networks for Text (and other types of sequences)

we`ve used CNNs for images but images are typically 2D (height andd width
                                                        )...
however, our text data is 1D

Previously we`ve Conv2D for our image data but now we`re going to use Conv1D.



The typical structuire of a Conv1D model for sequences (in our case, text)

Inputs(text) -> Tokenization -> Embedding -> Layer(s)(typically Conv1D + pooling)
-> Outputs (class probs)

"""


"""

* Model 5: 1D Convolutional Neural Network (CNN).

"""


embedding_test = embedding(text_vect(["this is a test sentence"])) #turn target sequence into embedding
conv_1d = layers.Conv1D(filters=32,
                        kernel_size=5,
                        activation="relu",
                        padding="valid")


conv_1d_output = conv_1d(embedding_test) # pass test embedding through conv1d layer
max_pool = layers.GlobalMaxPool1D()

max_pool_output = max_pool(conv_1d_output) # equivqlent to "get the most important feature" or "get the featuire with the heighest value"


embedding_test.shape, conv_1d_output.shape, max_pool_output.shape




#Create 1-dimensional convolutional layer to model sequences

inputs = layers.Input(shape=(1,), dtype="string")

x = text_vect(inputs) # turn the input text to numbers

x = embedding(x) # Create an embedding of the numberized inputs
#print(x.shape)
x = layers.Conv1D(64, 5, activation="relu", padding="valid")(x)
x = layers.GlobalMaxPool1D()(x)

outputs = layers.Dense(1, activation="sigmoid")(x) # Out put layer

model_5 = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")




#compile
model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_5.summary() 


# Fit the model


model_5_history = model_5.fit(train_sent,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_5_conv1D")])





# Check the results
model_5.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_5_preds_probs = model_5.predict(val_sent)

model_5_preds_probs[:10]

model_5_preds = tf.squeeze(tf.round(model_5_preds_probs))



model_5_preds[:10]
val_labels[:10]



# Calculate our model_1 results

model_5_results = calculate_results(y_true=val_labels,
                                    y_pred=model_5_preds)


model_5_results

baseline_results






"""

* Model 6: TensorFlow Hub Pretrained Feature Extractor (using transfer learning for NLP).

hugging face just like tensorflow_hub for text classification
"""


import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed_smaples = embed([sample_sent,
                       "when you call the universal sentence encoder on a sentence, it turns it into numbers."])

print(embed_smaples[0][:50])





# Create a keras layer using the USE pretrained layer from tensorflow hub

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="USE")

# Create model usingf the Sequential API
from tensorflow.keras import Sequential

model_6 =  Sequential([
    sentence_encoder_layer,
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")],
    name="model_6_USE"
    )



model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_6.summary() 


# Fit the model


model_6_history = model_6.fit(train_sent,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_6_pretrained_model")])





# Check the results
model_6.evaluate(val_sent, val_labels)

# Make some predictions and evaluate those

model_6_preds_probs = model_6.predict(val_sent)

model_6_preds_probs[:10]

model_6_preds = tf.squeeze(tf.round(model_6_preds_probs))



model_6_preds[:10]
val_labels[:10]



# Calculate our model_1 results

model_6_results = calculate_results(y_true=val_labels,
                                    y_pred=model_6_preds)


model_6_results

baseline_results





"""
* Model 7: same as model 6 but with 10% of training data.

"""

# Create subsets of 10% of the training data

train_10_percent = train_df_shuffled[["text", "target"]].sample(frac=0.1, random_state=42)

train_10_percent.head(), len(train_10_percent)

train_sent_10_percent = train_10_percent["text"].to_list()

train_labels_10_percent = train_10_percent["target"].to_list()


"""
To create a model the same as previous model you`ve created .you can use the "tf.keras.models.clone_model()" method

"""

model_7 = tf.keras.models.clone_model(model_6)

#compile model
model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])






model_7.summary() 


# Fit the model


model_7_history = model_7.fit(train_sent_10_percent ,
                              train_labels_10_percent,
                              epochs=5,
                              validation_data=(val_sent, val_labels),
                              callbacks=[create_tensorboard_callback(save_dir,
                                                                     experiment_name="model_7_pretrained_model_10_perce")])





# Check the results
model_7.evaluate(val_sent, val_labels)


# Make some predictions and evaluate those

model_7_preds_probs = model_6.predict(val_sent)

model_7_preds_probs[:10]

model_7_preds = tf.squeeze(tf.round(model_7_preds_probs))



model_7_preds[:10]
val_labels[:10]



# Calculate our model_1 results

model_7_results = calculate_results(y_true=val_labels,
                                    y_pred=model_7_preds)


model_7_results

baseline_results











































