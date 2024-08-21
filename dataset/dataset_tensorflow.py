
import tensorflow as tf
import math
import os
import pandas as pd
from utils.dir import create_dir


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
    

def get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


def load_features(path, dataset='music_style'):
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


def create_example(data):
    track_id, labels, music = data
    data = {}

    data['features'] = _float_feature(music)
    data['track_id'] = _int64_feature(track_id)

    for idx, level in enumerate(labels, start=1):
        label_key = f'level{idx}'
        data[label_key] =  _int64List_feature(level)

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def generate_tf_record(df, tf_path='val'):
    create_dir(tf_path)

    batch_size = 1024 * 50  # 50k records from each file batch
    count = 0
    total = math.ceil(len(df) / batch_size)
    for i in range(0, len(df), batch_size):
        batch_df = df[i:i + batch_size]
        tfrecords = [create_example(data) for data in batch_df.values]
        path = f"{tf_path}/{str(count).zfill(10)}.tfrecord"

        # with tf.python_io.TFRecordWriter(path) as writer:
        with tf.io.TFRecordWriter(path) as writer:
            for tfrecord in tfrecords:
                writer.write(tfrecord.SerializeToString())

        print(f"{count} {len(tfrecords)} {path}")
        count += 1
        print(f"{count}/{total} batchs / {count * batch_size} processed")

    print(f"{count}/{total} batchs / {len(df)} processed")
