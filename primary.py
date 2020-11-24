# -*- coding: utf-8 -*-

import os
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import functions as fns
import logging
import seaborn as sns
import tensorflow 
import pandas as pd
import numpy as np
import os
import keras
import sklearn
import random
import cv2
import math
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from get_data import obtain_data
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


###Download and Extract Data.
obtain_data('wiki', 'imdb')

data_sources = ['wiki', 'imdb']
###Data cleaning and filtering.
df = pd.DataFrame()
for source in data_sources:
    IMAGE_DIRECTORY = '{}_crop'.format(source)
    MATDB_FILE = os.path.join(IMAGE_DIRECTORY, '{}.mat'.format(source))

    matdb = scipy.io.loadmat(MATDB_FILE)['{}'.format(source)][0, 0]
    print('MATLAB database rows: {}'.format(
        str(len(matdb["face_score"][0]))))

    rows = []
    MINIMUM_FACE_SCORE = 1.0

    print(source)

    for i in range(len(matdb["face_score"][0])):
        dob = int(matdb["dob"][0][i])
        face_score = matdb["face_score"][0][i]
        second_face_score = matdb["second_face_score"][0][i]

        if dob <= 366:
            continue

        if face_score < MINIMUM_FACE_SCORE or np.isinf(face_score):
            continue

        if (~np.isnan(second_face_score)) and second_face_score > 0.0:
            continue

        file_path = os.path.join(
            IMAGE_DIRECTORY, matdb["full_path"][0][i][0])
        age = fns.calculate_age(dob, int(matdb["photo_taken"][0][i]))
        gender_id = matdb["gender"][0][i]
        # print(source)
        if os.path.isfile(file_path):
            gender = fns.get_gender(gender_id)
            age_group_id = fns.get_age_group_id(age)
            age_group_label = str(age_group_id)
            age_group = fns.get_age_group(age_group_id)
            rows.append({'file_path': file_path, 'gender_id': gender_id, 'gender': gender,
                         'age': age, 'age_group_id': age_group_id, 'age_group_label': age_group_label,
                         'age_group': age_group})
        else:
            print(
                'Image file does not exist! Skipping record for image: {}'.format(file_path))

    initial_df = pd.DataFrame(rows, columns=['file_path', 'gender_id', 'gender', 'age',
                                             'age_group_id', 'age_group_label', 'age_group'])

    if df.empty:
        df = initial_df
    else:
        df = pd.concat([initial_df, df])

df = df.drop_duplicates()
df = df.dropna()
df = df.astype({'gender_id': 'int64'})

df.drop(df[df.age < 0].index, inplace=True)
df.drop(df[df.age > 100].index, inplace=True)

## Cleaned data Summary
df.describe()

##Age Distribution Chart
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.displot(df, x="age", bins=20, binwidth=3)
plt.xlabel("Age", size=13)
plt.ylabel("Count", size=13)
plt.title("Age Distribution", size=17)

###Size of the bucket “15 to 25 years old”
age15_25 = sum(map(lambda i: i >14 and i < 26 , df['age']))
print('Size of the bucket “15 to 25 years old”:') 
print(age15_25)

###Percentage of this population is "30 year old males"

age30 = df[df["age"] == 30]
male30 = ((sum(map(lambda i: i == 'Male', age30['gender']))/df['age'].count())*100).round(2)

print('Percentage of this population is "30 year old males”:') 
print(male30)

#Train--Test Split
train_df = None
validation_df = None
test_df = None

for age_group_id in df.age_group_id.unique():
    split_df, tmp_test_df = sklearn.model_selection.train_test_split(df[df.age_group_id == age_group_id],
                                                                      test_size=0.1)
    tmp_train_df, tmp_validation_df = sklearn.model_selection.train_test_split(split_df, test_size=0.2)
    
    if train_df is None:
        train_df = tmp_train_df.copy(deep=True)
    else:
        train_df = train_df.append(tmp_train_df, ignore_index=True)
        
    if validation_df is None:
        validation_df = tmp_validation_df.copy(deep=True)
    else:
        validation_df = validation_df.append(tmp_validation_df, ignore_index=True)

    if test_df is None:
        test_df = tmp_test_df.copy(deep=True)
    else:
        test_df = test_df.append(tmp_test_df, ignore_index=True)

print('Train Count')
print(train_df['age'].count())
print('Validate Count')
print(validation_df['age'].count())
print('Test Count')
print(test_df['age'].count())

# Value Def
IMAGE_HEIGHT_PIXELS = 224
IMAGE_WIDTH_PIXELS = 224
IMAGE_COLOR_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 128

#Preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,                                                                                                                                                                            
        zoom_range=0.2,
        horizontal_flip=True)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file_path',
        y_col='age_group',
        target_size=(IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical')

validation_generator = valid_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='file_path',
        y_col='age_group',
        target_size=(IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

#Calculation for Focal Loss Fun
print(train_df.groupby(["age"]).agg(["count"]))
age_dist = [train_df["age"][(train_df.age >= x -10) & (train_df.age <= x)].count() for x in range(10, 101, 10)]
age_dist = [age_dist[0]] + age_dist + [age_dist[-1]]
print(age_dist)


#Image Augmentation Layer def
img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


#Model V1 
def build_model(num_classes):
    inputs = layers.Input(shape=(IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS, IMAGE_COLOR_CHANNELS))
    x = img_augmentation(inputs)
    model = DenseNet121(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="DenseNet121")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss=["mae", fns.focal_loss(age_dist)], metrics=["mae","accuracy"]
    )
    return model

model = build_model(num_classes=NUM_CLASSES)

model.summary()

adjust = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=2, min_lr=1e-3)
check = ModelCheckpoint('model.h5', verbose=2, save_best_only=True)

EPOCHS = 50
# Fits-the-model
history = model.fit_generator(train_generator,
               epochs=50,
               verbose=1                 ,
               callbacks=[adjust, check],
               validation_data=validation_generator)

#Graph Val vs Train
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Model V2
def unfreeze_model(model):
    #unfreeze the top 17 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-17:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss=["mae", fns.focal_loss(age_dist)], metrics=["mae","accuracy"]
    )


unfreeze_model(model)

check = ModelCheckpoint('age_modelv2.h5', verbose=2, save_best_only=True)
EPOCHS = 10  
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[check], verbose=1)

#Graph Val vs Train
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Test
age_predictions = []
age_results = []

for index, row in test_df.iterrows():
    file_path, age_group_id = row['file_path'], row['age_group_id']
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=IMAGE_COLOR_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    age_prediction = prediction[0].argmax()
    age_result = 0
    if age_group_id == age_prediction:
        age_result = 1
    age_predictions.append(age_prediction)
    age_results.append(age_result)
    


test_results_df = test_df.copy()
test_results_df['age_predicted'] = age_predictions
test_results_df['age_result'] = age_results

print('Accuracy: {:.2%}'.format(test_results_df['age_result'].sum() / len(test_results_df.index)))
