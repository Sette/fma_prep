#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import shutil
import json
import ast
import os
from tqdm.notebook import tqdm

from utils.dir import create_dir, __load_json__
from dataset.labels import __create_labels__, get_labels_name, split_label, get_all_structure, convert_label_to_string, parse_label
from dataset.dataset import select_dataset, generate_tf_record, create_metadata

# In[2]:


tqdm.pandas()

# In[3]:

def get_labels_per_level():
    print('teste')

def remover_sublistas_redundantes(lista_de_listas):
    elementos_vistos = set()
    sublistas_filtradas = []

    for sublista in lista_de_listas:
        if not any(elem in elementos_vistos for elem in sublista):
            sublistas_filtradas.append(sublista)
            elementos_vistos.update(sublista)

    return sublistas_filtradas


def prepare_paths(args):
    ## Define job paths
    fma_path = os.path.join(args.root_dir,"fma")
    args['dataset_path'] = os.path.join(args.root_dir, "fma")
    job_path = os.path.join(fma_path,"trains")
    args['job_path'] = os.path.join(job_path, args.train_id)
    args['tfrecord_path'] = os.path.join(args.job_path, "tfrecords")
    args['models_path'] = os.path.join(args.root_dir, "models")
    args['metadata_path'] = os.path.join(fma_path, "fma_metadata")
    args['metadata_train_path'] = os.path.join(job_path, "metadata.json")
    args['categories_labels_path'] = os.path.join(job_path, "labels.json")

    ## Remove files from path
    shutil.rmtree(job_path)

    # In[8]:

    ## Create poth if it isn't exist
    create_dir(job_path)
    # Load tracks_df
    tracks_df = pd.read_csv(os.path.join(args.metadata_path, 'tracks_valid.csv'))

    # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # In[13]:

    ## Get sample size from args parameter
    tracks_df = tracks_df.sample(frac=args.sample_size)

    tracks_df["track_genres_all"] = tracks_df.track_genres_all.apply(lambda x: ast.literal_eval(x))
    tracks_df.drop(columns=['track_genres'], inplace=True)
    tracks_df.dropna(inplace=True)


    return args, tracks_df

def prepare_labels(tracks_df,args):
    ##### Labels
    # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # Mapear os identificadores numéricos de gêneros para os nomes dos gêneros
    genre_names = dict(zip(genres_df['genre_id'], genres_df['title']))

    # Inicialize uma lista para armazenar todos os caminhos de gêneros para cada exemplo
    estruturas = []

    # Iterar sobre as faixas e seus gêneros associados
    for track_genres in tracks_df['track_genres_all']:
        #caminho_name = [genre_names[genre_id] for genre_id in track_genres]
        caminho_id = [get_all_structure(genre_id, genres_df) for genre_id in track_genres]
        estruturas.append(caminho_id)

    for idx, caminho in enumerate(estruturas):
        caminho.sort(key=len, reverse=True)
        estruturas[idx] = remover_sublistas_redundantes(caminho)

    ## Get structure form hierarchical classification
    tracks_df['full_genre_id'] = estruturas
    tracks_df = tracks_df[['track_id_', 'full_genre_id']]

    ## Calculate labels_size
    max_depth = tracks_df.full_genre_id.apply(lambda x: len(x))
    max_depth = int(max_depth.max())
    print(max_depth)
    labels_name = []
    for level in range(max_depth):
        labels_name.append(f'label_{level+1}')
    print(labels_name)
    tqdm.pandas()
    ## Gnetare categories_df
    #labels =  tracks_df.full_genre_id.progress_apply(lambda x: get_labels_per_level(x))

    #all_levels = categories_df.label5.progress_apply(lambda x: split_label(x))
    all_labels = []
    for idx, row in enumerate(tracks_df.full_genre_id):
        for labels in row:
            labels.extend([0] * (max_depth - len(labels)))
            all_labels.append(labels)
            
    categories_df = pd.DataFrame(all_labels, columns=labels_name).drop_duplicates()
    
    categories_df[f'label_{max_depth+1}_name'] = [get_labels_name(categorie, genres_df) for categorie in categories_df.values]

    print(categories_df)
    # Write labels file
    with open(args.categories_labels_path, 'w+') as f:
        f.write(json.dumps(__create_labels__(categories_df, max_depth)))

    return tracks_df

def split_dataset(tracks_df,args):
    #### Split dataset

    df_train, df_test, df_val = select_dataset(tracks_df)

    val_path = generate_tf_record(df_val, tf_path=os.path.join(args.tfrecord_path, 'val'))
    test_path = generate_tf_record(df_test, tf_path=os.path.join(args.tfrecord_path, 'test'))
    train_path = generate_tf_record(df_train, tf_path=os.path.join(args.tfrecord_path, 'train'))

    ## Create metadata file
    create_metadata(args.metadata_path)

    train_path.to_csv(os.path.join(args.job_path, "train.csv"), index=False)
    test_path.to_csv(os.path.join(args.job_path, "test.csv"), index=False)
    val_path.to_csv(os.path.join(args.job_path, "val.csv"), index=False)


def run():
    args = pd.Series({
        "root_dir": "/mnt/disks/data/",
        "dataset_path": "/mnt/disks/data/fma/fma_large",
        "embeddings": "music_style",
        "sequence_size": 1280,
        "train_id": "hierarchical_hiclass",
        'sample_size': 1/100
    })

    args, tracks_df = prepare_paths(args)
    tracks_df = prepare_labels(tracks_df,args)
    #split_dataset(tracks_df,args)




