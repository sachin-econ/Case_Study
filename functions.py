# -*- coding: utf-8 -*-
"""Untitled15.ipynb

"""
import datetime
import dateutil
import sys
import requests


def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(
            target_tensor, zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * \
            tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(
            classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(
            target_tensor, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed


def calculate_age(dob, photo_taken_year):
    days = dob % 1
    birth_date = datetime.date.fromordinal(
        dob) + datetime.timedelta(days=days) - datetime.timedelta(days=366)
    return dateutil.relativedelta.relativedelta(datetime.date(photo_taken_year, 7, 1), birth_date).years


def get_age_group_id(age):
    if age < 18:
        return 0
    if age <= 24:
        return 1
    if age <= 29:
        return 2
    if age <= 34:
        return 3
    if age <= 39:
        return 4
    if age <= 44:
        return 5
    if age <= 54:
        return 6
    if age <= 64:
        return 7
    if age <= 74:
        return 8
    return 9


def get_age_group(age_range_id):
    if age_range_id == 0:
        return '< 18'
    if age_range_id == 1:
        return '18 - 24'
    if age_range_id == 2:
        return '25 - 29'
    if age_range_id == 3:
        return '30 - 34'
    if age_range_id == 4:
        return '35 - 39'
    if age_range_id == 5:
        return '40 - 44'
    if age_range_id == 6:
        return '45 - 54'
    if age_range_id == 7:
        return '55 - 64'
    if age_range_id == 8:
        return '65 - 74'
    return '75+'


def get_gender(gender_id):
    if gender_id == 0:
        return 'Female'
    if gender_id == 1:
        return 'Male'
    return 'Unknown'



def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format(
                    '█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
