import json

t = open("training.json")
te = open("test.json")

train_data = json.load(t)

test_data = json.load(te)

print(train_data)
print(test_data)
t.close()
te.close()

import pandas as pd





df = pd.DataFrame(data=data)
df = pd.DataFrame({"category": train_data.keys()})

df.head()




#########################################################
import pandas as pd
import numpy as np
import os

import cv2 as cv
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize

import warnings

warnings.filterwarnings("ignore")

os.listdir("malaria")


path_malaria = "malaria/training.json"

df_malaria = pd.read_json(path_malaria)

def get_num_cells(x):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    num_cells = len(x)
    return num_cells

df_malaria["num_cells"] = df_malaria["objects"].apply(get_num_cells)

#Drop the objects column
df_malaria = df_malaria.drop('objects', axis=1) 


# Create new columns

df_malaria["image_id"] = 0
df_malaria["pathname"] = 0
df_malaria["r"] = 0
df_malaria["c"] = 0
df_malaria["channels"] = 0


#df_malaria
for i in range(0, len(df_malaria)):
    
    img_dict =  df_malaria.loc[i, "image"]
    
    
    df_malaria.loc[i, "image_id"] = img_dict["checksum"]
    df_malaria.loc[i, "pathname"] = img_dict["pathname"]
    df_malaria.loc[i, "r"] = img_dict["shape"]["r"]
    df_malaria.loc[i, "c"] = img_dict["shape"]["c"]
    df_malaria.loc[i, "channels"] = img_dict["shape"]["channels"]


df_malaria = df_malaria.drop("image", axis=1)



df_malaria.head()












































