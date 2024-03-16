import pandas as pd
import numpy as np
import tensorflow as tf


import os
import json
import math

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


def get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


def load_dataset(path, dataset='music_style'):
    tfrecords_path = os.path.join(path, 'tfrecords', dataset)

    tfrecords_path = [os.path.join(tfrecords_path, path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)

    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature', 'track_id']
    )

    df.dropna(inplace=True)

    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print('Erro ao carregar features')


    return df




def __split_data__(group, percentage=0.1):
    if len(group) == 1:
        return group, group

    shuffled = shuffle(group.values)
    finish_test = int(ceil(len(group) * percentage))

    first = pd.DataFrame(shuffled[:finish_test], columns=group.columns)
    second = pd.DataFrame(shuffled[finish_test:], columns=group.columns)

    return first, second


# In[58]:


def select_dataset(tracks_df, args):
    #     dataset_testset_path = os.path.join(tfrecord_path,'test')
    #     dataset_validationset_path = os.path.join(tfrecord_path,'val')
    #     dataset_trainset_path = os.path.join(tfrecord_path,'train')

    df = load_dataset(args.dataset_path, dataset=args.embeddings)

    df.dropna(inplace=True)

    tracks_df = tracks_df.merge(df, on='track_id')

    tracks_df.loc[:, 'labels_1'] = tracks_df.labels_1.astype(str).progress_apply(lambda x: args.labels['label1'][x])
    tracks_df.loc[:, 'labels_2'] = tracks_df.labels_2.astype(str).progress_apply(lambda x: args.labels['label2'][x])
    tracks_df.loc[:, 'labels_3'] = tracks_df.labels_3.astype(str).progress_apply(lambda x: args.labels['label3'][x])
    tracks_df.loc[:, 'labels_4'] = tracks_df.labels_4.astype(str).progress_apply(lambda x: args.labels['label4'][x])
    tracks_df.loc[:, 'labels_5'] = tracks_df.labels_5.astype(str).progress_apply(lambda x: args.labels['label5'][x])

    tests = []
    trains = []
    validations = []
    groups = tracks_df.groupby("labels_5")

    count = 0
    items_count = 0
    total = len(groups)
    total_items = len(tracks_df)
    oversampling_size = 30  # int(group_sizes.mean() + group_sizes.std() * 2)
    print(f"oversampling_size: {oversampling_size}")

    for code, group in groups:
        test, train_to_split = __split_data__(group, 0.01)  # 10%
        train_to_split = train_to_split
        validation, train = __split_data__(train_to_split, 0.01)  # %1

        tests.append(test)
        validations.append(validation)

        ## this increase the numner of samples when classes has low quantity
        count_train = len(train)
        if count_train < oversampling_size:
            train = train.sample(oversampling_size, replace=True)

        trains.append(train)

        count += 1
        items_count += count_train

    df_test = pd.concat(tests, sort=False).sample(frac=1).reset_index(drop=True)
    # .to_csv(dataset_testset_path, index=False,quoting=csv.QUOTE_ALL)
    df_val = pd.concat(validations, sort=False).sample(frac=1).reset_index(drop=True)
    df_train = pd.concat(trains, sort=False).sample(frac=1).reset_index(drop=True)

    return df_train, df_test, df_val



def _bytes_feature(value):
    ### Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


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


# In[62]:


def parse_single_music(data, labels):
    track_id, _, cat1, cat2, cat3, cat4, cat5, music = data

    label1 = np.array([cat1, labels['label1_count']], np.int64)
    label2 = np.array([cat2, labels['label2_count']], np.int64)
    label3 = np.array([cat3, labels['label3_count']], np.int64)
    label4 = np.array([cat4, labels['label4_count']], np.int64)
    label5 = np.array([cat5, labels['label5_count']], np.int64)

    # define the dictionary -- the structure -- of our single example
    data = {
        'label1': _int64List_feature(label1),
        'label2': _int64List_feature(label2),
        'label3': _int64List_feature(label3),
        'label4': _int64List_feature(label4),
        'label5': _int64List_feature(label5),
        # 'features' : _bytes_feature(serialize_array(music)),
        'features': _float_feature(music),
        'track_id': _int64_feature(track_id)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


# In[63]:


def generate_tf_record(df,args, tf_path='val'):
    create_dir(tf_path)

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

    return tf_path



def create_metadata(metadata_path, args):
    with open(metadata_path, 'w+') as f:
        f.write(json.dumps({
            'sequence_size': args.sequence_size,
            'n_levels': args.labels_size,
            'labels_size': [args.labels['label1_count'], args.labels['label2_count'],
                            args.labels['label3_count'], args.labels['label4_count'],
                            args.labels['label5_count']],
            'val_path': args.val_path,
            'train_path': args.train_path,
            'test_path': args.test_path,
            'trainset_count': len(args.df_train),
            'validationset_count': len(args.df_val),
            'testset_count': len(args.df_test)
        }))

