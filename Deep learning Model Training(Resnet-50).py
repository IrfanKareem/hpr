#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Connect the google drive and adjust the path
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive')


# In[ ]:


#Import the Libraries
import numpy as np 
import pandas as pd 
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.optimizers import RMSprop
from keras.applications.resnet50 import ResNet50


# In[ ]:


#Give the lables path of each dataset
df_train = pd.read_excel('/content/drive/My Drive/Labels_1.xlsx')
df_train.head()


# In[ ]:


targets_series = pd.Series(df_train['Level'])
one_hot = pd.get_dummies(targets_series, sparse = True)


# In[ ]:


one_hot_labels = np.asarray(one_hot)


# In[ ]:


#Adjust the Image Size
im_size1 = 224 
im_size2 = 224 


# In[ ]:


x_train = []
y_train = []
x_test = []


# In[ ]:


#Iterate over the dataset
i = 0 
for f, breed in tqdm(df_train.values[:6786]): #adjust the number of training samples in the dataset here
  if type(cv2.imread('Train/{}'.format(f)))==type(None):
    continue 
  else: 
    img = cv2.imread('Train/{}'.format(f)) 
    label = one_hot_labels[i] 
    x_train.append(cv2.resize(img, (im_size1, im_size2))) 
    y_train.append(label) 
    i += 1 
np.save('x_train2',x_train) 
np.save('y_train2',y_train) 
print('Done')


# In[ ]:


x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')


# In[ ]:


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.


# In[ ]:


print(x_train_raw.shape)
print(y_train_raw.shape)


# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.1, random_state=1)


# In[ ]:


num_class = y_train_raw.shape[1]


# In[ ]:


#ResNet50 with imagenet weights & Chage the respective basemodel here after importing
base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(im_size1, im_size2, 3))

# Add/Adjust a new top layer
x = base_model.output
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

# The model Training
model = Model(inputs=base_model.input, outputs=predictions)

#ajust the optimzer and performce metrices and los 
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
model.summary()


# In[ ]:


model.fit(X_train, Y_train,  epochs=40, validation_data=(X_valid, Y_valid), verbose=1)

