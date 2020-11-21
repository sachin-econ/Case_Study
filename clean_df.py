import os
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import functions as fns
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def data_sources(load_source):
    data_sources = [load_source]
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

    df.info()

    df.dtypes

    df = df.drop_duplicates()
    df = df.dropna()
    df = df.astype({'gender_id': 'int64'})
    df.info()

    df.describe()

    df.drop(df[df.age < 0].index, inplace=True)
    df.drop(df[df.age > 100].index, inplace=True)
    df.describe()


