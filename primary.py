# -*- coding: utf-8 -*-

import os
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import functions as fns
import logging
import seaborn as sns
import matplotlib.pyplot as plt
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from get_data import obtain_data

# Download and Extract Data.
obtain_data('wiki', 'imdb')

data_sources = ['wiki', 'imdb']
# Data cleaning and filtering.
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
df.describe()

# Age Distribution Chart
plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
sns.displot(df, x="age", bins=20, binwidth=3)
plt.xlabel("Age", size=13)
plt.ylabel("Count", size=13)
plt.title("Age Distribution", size=17)

# Size of the bucket “15 to 25 years old”
age15_25 = sum(map(lambda i: i > 14 and i < 26, df['age']))
print('Size of the bucket “15 to 25 years old”:')
print(age15_25)

# Percentage of this population is "30 year old males"

age30 = df[df["age"] == 30]
male30 = ((sum(map(lambda i: i == 'Male',
                   age30['gender'])) / df['age'].count()) * 100).round(2)

print('Percentage of this population is "30 year old males”:')
print(male30)