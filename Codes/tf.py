import tensorflow as tf
import numpy as np
mat1 =tf.constant([[1, 2, 3],
                   [7, 2, 1],
                   [3, 3, 3]])
mat2 =tf.constant([[3, 5],
                   [6, 7],
                   [1, 8]])

X =tf.constant([[1, 2],
                [3, 4],
                [5, 6]])

y =tf.constant([[7, 8],
                [9, 10],
                [11, 12]])

tf.tensordot(tf.transpose(X), y, axes=1)
##Perform matrix multiplication between x and y(transpose)

tf.matmul(X, tf.transpose(y))

##Perform matrix multiplication between x and y(reshaped)
tf.matmul(X, tf.reshape(y, shape=(2, 3)))


##check the value of y, reshape y AND transpose y

print("Normal Y:")
print(y, "\n") #\n is for newline

print("Y reshaped t0 (2, 3): ")
print(tf.reshape(y, shape=(2,3)), "\n")


print("Y transposed:")
print(tf.transpose(y))

##Changing the datatype of a tensor

tf.__version__


B = tf.constant([1.7, 7.4])
B.dtype


C = tf.constant([17, 4])
C.dtype

C1 = tf.cast(C, dtype=tf.int16)

#Aggregating tensors
##Getting the absolute value

D = tf.constant([-7, -10])
D

##Get the absolute value

tf.abs(D)

##Lets go through the following form of aggregation:
    #Get the min
    #Get the max
    #Get the mean of a tensor
    #Get the sum of a tensor
    
    
#Creating a random tensor with values between 0 and 100 of size 50

E = tf.constant(np.random.randint(0, 100, size=50))
E

tf.size(E), E.shape, E.ndim

#Finding the minimum
tf.reduce_mean(E)
tf.reduce_max(E)
tf.reduce_min(E)
tf.reduce_sum(E)

##Eg find the varient and standard deviation of E

import keras.p
import tensorflow_probability as tfp

##Create a new tensor fr finding positional min and max

tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
F

#find the positional max

tf.argmax(F)
F[tf.argmax(F)]

tf.reduce_max(F)


assert F[tf.argmax(F)] == tf.reduce_max(F)

F[tf.argmax(F)] == tf.reduce_max(F)




#find the positional min

tf.argmin(F)
F[tf.argmin(F)]

tf.reduce_min(F)


assert F[tf.argmin(F)] == tf.reduce_min(F)

F[tf.argmin(F)] == tf.reduce_min(F)


#Squeezing a tensor(removing all single dimensions)
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
G.shape


G_squeezed = tf.squeeze(G)

G_squeezed, G_squeezed.shape



##Create a list of indices

some_list = [0, 1, 2, 3] ##could be red, green, blue, purple

#one hot encode our list of indices
tf.one_hot(some_list, depth=4)



##Specify custom values for one hot encoding

tf.one_hot(some_list, depth=4, on_value="yo i love deep learning", off_value="i also like to dance")


##Squaring, log, square_root
H = tf.range(1, 10)
H

#Square it
tf.square(H)

#log
tf.math.log(tf.cast(H, dtype=tf.float32))

#square_root
tf.sqrt(tf.cast(H, dtype=tf.float32))

##Tensors and numpy

#tensoef
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


tf.config.list_physical_devices("GPU")

##check the diif btw Nvidia K80s, T4sw, P4s and P100s

!nvidia-smi

##Book to cover .... HANDS-ON MACHINE LEARN WITH SCIKIT-LEARN,KERAS AND TF BOOK BY AURELIEN GERON

import matplotlib.pyplot as plt
import pandas as pd















                                                                                                                       






