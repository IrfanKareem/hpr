#Connect the google drive and adjust the path
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive')

#Import the Libraries
import numpy as np 
import pandas as pd 
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import imagenet_utils
from skimage import data, color, feature
import os
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.applications.resnet import ResNet101
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.applications import MobileNet
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D


#Give the labels path of each dataset
df_train = pd.read_excel('/content/drive/My Drive/Labels.xlsx')

df_train.head()

targets_series = pd.Series(df_train['Level'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)


#Adjust the Image Size 
im_size1 = 224
im_size2 = 224

x_train = []
y_train = []
x_test = []


#Iterate over the dataset
i = 0 
for f, breed in tqdm(df_train.values):
  if type(cv2.imread('IMAGES/{}'.format(f)))==type(None):
    continue 
  else: 
    img = cv2.imread('IMAGES/{}'.format(f)) 
    label = one_hot_labels[i] 
    x_train.append(cv2.resize(img, (im_size1, im_size2))) 
    y_train.append(label) 
    i += 1 
np.save('x_trainuURF',x_train) 
np.save('y_trainURF',y_train) 
print('Done')

x_train = np.load('x_trainuURF.npy')
y_train = np.load('y_trainURF.npy')

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.

print(x_train_raw.shape)
print(y_train_raw.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.1, random_state=1)

num_class = y_train_raw.shape[1]

#ResNet50 without weights
base_model = ResNet50(weights = None, include_top=False, input_shape=(im_size1, im_size2, 3))

# Add a new top layer
x = base_model.output
x = AveragePooling2D(pool_size=(4, 4))(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
#x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

# The model train
model = Model(inputs=base_model.input, outputs=predictions)

#ajust the optimzer and performce metrices and los 
model.compile(loss='categorical_crossentropy', 
              optimizer='SGD', 
              metrics=['accuracy'])


callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
model.summary()


model.fit(X_train, Y_train, epochs=5, validation_data=(X_valid, Y_valid), verbose=1,shuffle=True)

X_train_features = []
X_valid_features = []

feature_network = Model(base_model.input, model.get_layer('flatten').output)

X_train_features = feature_network.predict(X_train)  
X_valid_features = feature_network.predict(X_valid)  #

print(X_train_features.shape)
print(X_valid_features.shape)
print(Y_train.shape)
print(Y_valid.shape)


from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time

# Hyper parameters
param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
start_time = time.time()
svm_orig = svm.LinearSVC(max_iter=1000, dual=False)
svm_orig = GridSearchCV(svm_orig, param_grid)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)
svm_orig.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))
# Print model with chosen hyperparameters
print(svm_orig)
# Predict on test data
svm_predict_orig = svm_orig.predict(X_valid_features)
# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)




