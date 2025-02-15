#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import torch
from essentia.standard import MonoLoader, FrameGenerator, TensorflowPredictEffnetDiscogs, TensorflowInputMusiCNN


tqdm.pandas()



args = pd.Series({
    "root_dir":"/home/bruno/storage/data",
    "dataset_path":"/home/bruno/storage/data/fma/fma_large",
    "embeddings":"music_style",
    "top_genres": ["Rock", "Electronic"]
})



base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join('/'.join(args.root_dir.split('/')[:-1]), "models")


metadata_path_fma = os.path.join(base_path,"fma_metadata")


if args.embeddings == "music_style":
    model_path = os.path.join(models_path,args.embeddings,"discogs-effnet-bs64-1.pb")


df = pd.read_csv(os.path.join(metadata_path_fma,"tracks_valid.csv"))

if args.top_genres:
    print(f"Using top genres list. {args['top_genres']}")
    df = df[df['track_genre_top'].isin(args['top_genres'])]


df = df[['track_id','file_path']]



#model = TensorflowPredictEffnetDiscogs(graphFilename=model_path, output="PartitionedCall")

model = TensorflowInputMusiCNN()

def create_dir(path):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(path)
    return True


def extract_feature(file_path,model):
    ### Configuração do model para extrair a representação do aúdio
    # model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
    audio = MonoLoader(filename=file_path, sampleRate=16000)()

    # Criar frames de 512 samples
    frames = list(FrameGenerator(audio, frameSize=512, hopSize=256))

    # Aplicar MusiCNN em cada frame e armazenar as ativações
    activations = np.array([model(frame) for frame in frames])

    # Concatenar todas as ativações em um único vetor
    final_feature_vector = np.concatenate(activations, axis=0).tolist()
    
    return final_feature_vector


def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path

df['file_path'] = df.track_id.apply(lambda x: find_path(str(x),args.dataset_path))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def parse_single_music(data,music):
    # cat1, cat2, cat3, cat4, cat5 = data
    track_id, track_title, valid_genre, file_path = data
    
    #define the dictionary -- the structure -- of our single example
    data = {
        'emb' : _bytes_feature(serialize_array(music)),
        'track_id' : _int64_feature(track_id)
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def process_df(df,i,count,batch_size,tfrecords_path):
    batch_df = df[i:i+batch_size]
        
    tqdm.pandas()

    X = batch_df.file_path.progress_apply(lambda x: extract_feature(x,model))   

    print("Extraiu as features")

    tfrecords = [parse_single_music(data, x) for data, x in zip(batch_df.values, X)]

    path = os.path.join(tfrecords_path,f"{str(count).zfill(10)}.tfrecord")

    #with tf.python_io.TFRecordWriter(path) as writer:
    with tf.io.TFRecordWriter(path) as writer:
        for tfrecord in tfrecords:
            writer.write(tfrecord.SerializeToString())

    print(f"{count} {len(tfrecords)} {path}")




def generate_tf_records(df,filename="train"):
    
    tfrecords_path = os.path.join(args.dataset_path,"tfrecords",filename)
    
    create_dir(tfrecords_path)
    
    batch_size = 1024 * 10  # 10k records from each file batch
    
    with Parallel(n_jobs=10, require='sharedmem') as para:
        print("Estamos usando paralelismo!!!")
        para(delayed(process_df)(df,i,count,batch_size,tfrecords_path) for count,i in enumerate(range(0, len(df), batch_size)))

##### 
## Pt files
#####

def create_example(data):
    track_id, _, music = data

    example = {
        'features': music,
        'track_id': track_id
    }
    
    return example

def process_df_topt(df,i,count,batch_size,pt_path,model):
    tqdm.pandas()
    batch_df = df[i:i+batch_size]
    batch_df["features"] = batch_df.file_path.progress_apply(lambda x: extract_feature(x,model))   
    
    print("Extraiu as features")

    path = os.path.join(pt_path, f"{str(count).zfill(10)}.pt")

    create_dir(pt_path)

    pt_records = [create_example(data) for data in batch_df.values]
    torch.save(pt_records, path)

    print(f"{count} {len(pt_records)} {path}")
    count += 1

    

    print(f"{count} {len(batch_df)} {path}")

def generate_pt_files(df, model, filename="train"):
    
    csv_path = os.path.join(args.dataset_path, "csv_rock_electronic", filename)
    
    batch_size = 1024 * 1  # 1k records from each file batch
    
    with Parallel(n_jobs=4, require='sharedmem') as para:
        print("Estamos usando paralelismo!!!")
        para(delayed(process_df_topt)(df,i,count,batch_size,csv_path, model) for count,i in enumerate(range(0, len(df), batch_size)))

### CSV files

def process_df_tocsv(df,i,count,batch_size,csv_path,model):
    tqdm.pandas()
    batch_df = df[i:i+batch_size]
    X = batch_df.file_path.progress_apply(lambda x: extract_feature(x,model))   
    batch_df.loc[:,"features"] = X
    
    print("Extraiu as features")

    path = os.path.join(csv_path,f"{str(count).zfill(10)}.csv")

    batch_df.to_csv(path, index=False)

    print(f"{count} {len(batch_df)} {path}")

def process_simple_df(df, csv_path, model):
    tqdm.pandas()
    X = df.file_path.progress_apply(lambda x: extract_feature(x, model))
    df["features"] = X
    
    print("Extraiu as features")

    path = os.path.join(csv_path, f"output.csv")

    # with tf.python_io.TFRecordWriter(path) as writer:

    df.to_csv(path, index=False)

    print(f"{len(df)} {path}")

def generate_csv_files(df, model, filename="train"):
    
    csv_path = os.path.join(args.dataset_path, "pt_rock_electronic", filename)
    
    batch_size = 1024 * 1  # 1k records from each file batch
    
    with Parallel(n_jobs=4, require='sharedmem') as para:
        print("Estamos usando paralelismo!!!")
        para(delayed(process_df_tocsv)(df,i,count,batch_size,csv_path, model) for count,i in enumerate(range(0, len(df), batch_size)))


def simple_generate_csv_files(df, model, filename="train"):
    csv_path = os.path.join(args.dataset_path, "csv_rock_electronic", filename)

    process_simple_df(df, csv_path, model)





#df = df.sample(10)

#generate_csv_files(df,model,filename=args.embeddings)

generate_pt_files(df,model,filename=args.embeddings)

#simple_generate_csv_files(df,model,filename=args.embeddings)





