import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os
import json
import math

from tqdm import tqdm

from sklearn.utils import shuffle
from math import ceil

from utils.dir import create_dir

def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb': tf.io.FixedLenFeature([], tf.string),
        'track_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    track_id = content['track_id']
    emb = content['emb']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, track_id)


def _float_feature(value):
    ### Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature(value):
    ###  Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array
    

def parse_single_music(data, labels):
    track_id, categories, music = data
    max_depth = len(categories[0])
    data = {}
    for level in range(1, max_depth+1):
        level_labels = []
        for cat in categories:
            if cat[level-1] != "":
                label = labels[f'label_{level}'][cat[level-1]]
                if label not in level_labels:
                    level_labels.append(label)
            else:
                if len(level_labels) == 0:
                    level_labels.append(-1)
        data[f'label{level}'] =  _int64List_feature(level_labels)

    data['features'] = _float_feature(music)
    data['track_id'] = _int64_feature(track_id)
    
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def generate_tf_record(df, df_features, args, tf_path='val'):
    create_dir(tf_path)
    
    df = df.merge(df_features, on='track_id')

    batch_size = 1024 * 50  # 50k records from each file batch
    count = 0
    total = math.ceil(len(df) / batch_size)
    for i in range(0, len(df), batch_size):
        batch_df = df[i:i + batch_size]
        tfrecords = [parse_single_music(data, args.labels) for data in batch_df.values]
        path = f"{tf_path}/{str(count).zfill(10)}.tfrecord"

        # with tf.python_io.TFRecordWriter(path) as writer:
        with tf.io.TFRecordWriter(path) as writer:
            for tfrecord in tfrecords:
                writer.write(tfrecord.SerializeToString())

        print(f"{count} {len(tfrecords)} {path}")
        count += 1
        print(f"{count}/{total} batchs / {count * batch_size} processed")

    print(f"{count}/{total} batchs / {len(df)} processed")
