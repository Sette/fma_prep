#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import json
import ast
import os
from tqdm.notebook import tqdm

from fma_prep.utils.dir import create_dir
from fma_prep.dataset.labels import __create_labels__, get_all_structure, get_labels_name
from fma_prep.dataset.dataset_tensorflow import generate_tf_record
from fma_prep.dataset.dataset import select_dataset, create_metadata, load_features
from sklearn.preprocessing import MultiLabelBinarizer
# In[2]:


tqdm.pandas()

def remover_sublistas_redundantes(lista_de_listas):
    max_depth = max([len(value) for value in lista_de_listas])
    new_sublist = []
    for sublista in lista_de_listas:
        if len(sublista) == max_depth:
            new_sublist.append(sublista)

    return new_sublist

def prepare_paths(args):
    ## Define job paths
    fma_path = os.path.join(args.root_dir,"fma")
    job_path = os.path.join(fma_path,"trains")
    args['job_path'] = os.path.join(job_path, args.train_id)
    args['tfrecord_path'] = os.path.join(args.job_path, "tfrecords")
    args['torch_path'] = os.path.join(args.job_path, "torch")
    args['models_path'] = os.path.join(args.root_dir, "models")
    args['metadata_path'] = os.path.join(fma_path, "fma_metadata")
    args['metadata_train_path'] = os.path.join(args['job_path'], "metadata.json")
    args['categories_labels_path'] = os.path.join(args['job_path'], "labels.json")

    ## Create poth if it isn't exist
    create_dir(args['job_path'])
    # Load tracks_df
    tracks_df = pd.read_csv(os.path.join(args.metadata_path, 'tracks_valid.csv'))

    # Loand genres df
    #genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # In[13]:

    ## Get sample size from args parameter
    tracks_df = tracks_df.sample(frac=args.sample_size)

    tracks_df["track_genres_all"] = tracks_df.track_genres_all.apply(lambda x: ast.literal_eval(x))
    tracks_df.drop(columns=['track_genres'], inplace=True)
    tracks_df.dropna(inplace=True)
    tracks_df.rename(columns={'track_id_':'track_id'},inplace=True)


    return tracks_df, args


# Função para dividir os rótulos em níveis
def split_labels(all_labels, level):
    return [label[level] if len(label) > level else None for labels in all_labels for label in labels]


def prepare_labels(tracks_df, args):
    ##### Labels
     # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # Mapear os identificadores numéricos de gêneros para os nomes dos gêneros
    
    # Inicialize uma lista para armazenar todos os caminhos de gêneros para cada exemplo
    estruturas = []

    # Iterar sobre as faixas e seus gêneros associados
    for track_genres in tracks_df['track_genres_all']:
        caminho_id = [get_all_structure(genre_id, genres_df) for genre_id in track_genres]
        estruturas.append(caminho_id)

    for idx, caminho in enumerate(estruturas):
        caminho.sort(key=len, reverse=True)
        estruturas[idx] = remover_sublistas_redundantes(caminho)

    ## Get structure form hierarchical classification
    #print(estruturas)
    tracks_df.loc[:, 'y_true'] = estruturas

    ## Calculate labels_size
    max_depth = tracks_df.y_true.apply(lambda x: max([len(value) for value in x]))
    max_depth = int(max_depth.max())
    args['max_depth'] = max_depth
    print(f'max depth: {max_depth}')
    
    labels_name = []
    for level in range(1, max_depth+1):
        labels_name.append(f'level{level}')
        idx = level-1
         # Extrai os rótulos do nível atual
        level_labels = split_labels(tracks_df['y_true'], idx)
         # Remove valores None
        level_labels = [[label] for label in level_labels if label is not None]
        
        # Cria e aplica o MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binarized = mlb.fit_transform(level_labels)

        binarized_labels = [binarized[i] if i < len(binarized) else [0] * len(mlb.classes_) for i in range(len(tracks_df))]

        tracks_df.loc[:, labels_name[idx]] = binarized_labels

    tracks_df['all_binarized'] = tracks_df.apply(lambda row: [sublist for sublist in row[labels_name]], axis=1)

    tracks_df = tracks_df[['track_id', 'y_true', 'all_binarized']]

    #all_levels = categories_df.label5.progress_apply(lambda x: split_label(x))
    all_labels = []
    for idx, row in enumerate(tracks_df.y_true):
        for labels in row:
            labels.extend([0] * (max_depth - len(labels)))
            all_labels.append(labels)
            
    categories_df = pd.DataFrame(all_labels, columns=labels_name).drop_duplicates()

    
    categories_df[f'level{max_depth}_name'] = [get_labels_name(categorie, genres_df) for categorie in categories_df.values]


    # Write labels file
    with open(args.categories_labels_path, 'w+') as f:
        labels = __create_labels__(categories_df, max_depth)
        args['levels_size'] = labels['levels_size']
        f.write(json.dumps(labels))

    
    return tracks_df, args



def split_dataset(tracks_df,args):
    #### Split dataset

    df_train, df_test, df_val = select_dataset(tracks_df, args)

    args['val_path'] = os.path.join(args.tfrecord_path, 'val')
    args['test_path'] = os.path.join(args.tfrecord_path, 'test')
    args['train_path'] = os.path.join(args.tfrecord_path, 'train')

    args['val_torch_path'] = os.path.join(args.torch_path, 'val.pth')
    args['test_torch_path'] = os.path.join(args.torch_path, 'test.pth')
    args['train_torch_path'] = os.path.join(args.torch_path, 'train.pth')

    args['train_csv'] = os.path.join(args.job_path, "train.csv")
    args['test_csv'] = os.path.join(args.job_path, "test.csv")
    args['val_csv'] = os.path.join(args.job_path, "val.csv")

    df_features = load_features(args.dataset_path, dataset=args.embeddings)

    df_features.dropna(inplace=True)

    df_val_features = df_val.merge(df_features, on='track_id')
    df_test_features = df_test.merge(df_features, on='track_id')
    df_train_features = df_train.merge(df_features, on='track_id')

    # df_train_features.to_csv(args['train_csv'], index=False)
    # df_test_features.to_csv(args['test_csv'], index=False)
    # df_val_features.to_csv(args['val_csv'], index=False)

    df_train_features = df_train_features[['track_id', 'all_binarized', 'feature']]
    df_test_features = df_test_features[['track_id', 'all_binarized', 'feature']]
    df_val_features = df_val_features[['track_id', 'all_binarized', 'feature']]

    generate_tf_record(df_val_features, args, tf_path=args['val_path'])
    generate_tf_record(df_test_features, args, tf_path=args['test_path'])
    generate_tf_record(df_train_features, args, tf_path=args['train_path'])

    # generate_torch_data(df_val_features, args, save_path=args['val_torch_path'], batch_size=1024 * 50, shuffle=True)
    # generate_torch_data(df_test_features, args, save_path=args['test_torch_path'], batch_size=1024 * 50, shuffle=True)
    # generate_torch_data(df_train_features, args, save_path=args['train_torch_path'], batch_size=1024 * 50, shuffle=True)

    args['val_len'] = df_val.shape[0]
    args['test_len'] = df_test.shape[0]
    args['train_len'] = df_train.shape[0]

    # ## Create metadata file
    create_metadata(args)

   

def run():
    args = pd.Series({
        "root_dir": "/mnt/disks/data/",
        "dataset_path": "/mnt/disks/data/fma/fma_large", 
        "embeddings": "music_style",
        "sequence_size": 1280,
        "train_id": "hierarchical_tworoots_dev",
        'sample_size': 0.1
    })


    tracks_df, args = prepare_paths(args)
    tracks_df = tracks_df[tracks_df['track_genre_top'].isin(['Rock','Electronic'])]
    tracks_df, args = prepare_labels(tracks_df,args)
    split_dataset(tracks_df,args)




