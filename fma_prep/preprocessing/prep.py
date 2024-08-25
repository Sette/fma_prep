#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import pickle
import argparse
import json
import ast
import os
from tqdm.notebook import tqdm

from fma_prep.utils.dir import create_dir
from fma_prep.dataset.labels import __create_labels__, get_all_structure, get_labels_name
from fma_prep.dataset.dataset_tensorflow import generate_tf_record
from fma_prep.dataset.dataset_torch import generate_pt_record
from fma_prep.dataset.dataset import select_dataset, create_metadata, load_features
from sklearn.preprocessing import MultiLabelBinarizer
# In[2]:

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)

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
    input_path = args.input_path
    output_path = args.output_path

    job_path = os.path.join(output_path, "trains")
    args['job_path'] = os.path.join(job_path, args.train_id)
    args['tfrecord_path'] = os.path.join(args.job_path, "tfrecrods")
    args['torch_path'] = os.path.join(args.job_path, "torch")
    args['metadata_path'] = os.path.join(input_path, 'fma_metadata')
    args['metadata_train_path'] = os.path.join(args['job_path'], "metadata.json")
    args['mlb_path'] = os.path.join(job_path, "mlb.pkl")
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

    max_depth = 0
    for idx, caminho in enumerate(estruturas):
        caminho.sort(key=len, reverse=True)
        caminho = remover_sublistas_redundantes(caminho)
        estruturas[idx] = caminho
        if len(caminho) > max_depth:
            max_depth = len(caminho)

    ## Get structure form hierarchical classification
    tracks_df.loc[:, 'y_true'] = estruturas
    all_labels = []
    lens = []
    for idx, row in enumerate(tracks_df.y_true):
        for labels in row:
            lens.append(len(labels))
            converted_labels = labels.copy()
            converted_labels.extend([0] * (max_depth - len(labels)))
            all_labels.append(converted_labels)

    labels_name = []
    for level in range(1, max_depth+1):
        labels_name.append(f'level{level}')

    categories_df = pd.DataFrame(all_labels, columns=labels_name).drop_duplicates()

    valid_labels_name = []
    for label_name in labels_name:
        unique_labels = [x for x in categories_df[label_name].unique() if x != 0]
        if len(unique_labels) > 1:
            valid_labels_name.append(label_name)

    max_depth = len(valid_labels_name)
    args['max_depth'] = max_depth
    categories_df = categories_df[valid_labels_name]

    mlbs = []
    
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
        binary_labels = mlb.fit_transform(level_labels)
        mlbs.append(mlb)

        binary_labels = [binary_labels[i] if i < len(binary_labels) else [0] * len(mlb.classes_) for i in range(len(tracks_df))]

        tracks_df.loc[:, labels_name[idx]] = binary_labels

    # Serializar a lista de mlb
    with open(args.mlb_path, 'wb') as file:
        pickle.dump(mlbs, file)

    tracks_df['all_binarized'] = tracks_df.apply(lambda row: [sublist for sublist in row[labels_name]], axis=1)

    tracks_df = tracks_df[['track_id', 'y_true', 'all_binarized']]

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

    args['val_torch_path'] = os.path.join(args.torch_path, 'val')
    args['test_torch_path'] = os.path.join(args.torch_path, 'test')
    args['train_torch_path'] = os.path.join(args.torch_path, 'train')

    args['train_csv'] = os.path.join(args.job_path, "train.csv")
    args['test_csv'] = os.path.join(args.job_path, "test.csv")
    args['val_csv'] = os.path.join(args.job_path, "val.csv")

    df_features = load_features(args.input_path)

    df_features.dropna(inplace=True)

    df_train.to_csv(args['train_csv'], index=False)
    df_test.to_csv(args['test_csv'], index=False)
    df_val.to_csv(args['val_csv'], index=False)

    df_val_features = df_val.merge(df_features, on='track_id')
    df_test_features = df_test.merge(df_features, on='track_id')
    df_train_features = df_train.merge(df_features, on='track_id')

    df_train_features = df_train_features[['track_id', 'all_binarized', 'feature']]
    df_test_features = df_test_features[['track_id', 'all_binarized', 'feature']]
    df_val_features = df_val_features[['track_id', 'all_binarized', 'feature']]

    generate_tf_record(df_val_features, tf_path=args['val_path'])
    generate_tf_record(df_test_features, tf_path=args['test_path'])
    generate_tf_record(df_train_features, tf_path=args['train_path'])

    generate_pt_record(df_val_features, pt_path=args['val_torch_path'])
    generate_pt_record(df_test_features, pt_path=args['test_torch_path'])
    generate_pt_record(df_train_features, pt_path=args['train_torch_path'])

    args['val_len'] = df_val.shape[0]
    args['test_len'] = df_test.shape[0]
    args['train_len'] = df_train.shape[0]

    # ## Create metadata file
    create_metadata(args)


def run():
    # ArgumentParser configuration
    parser = argparse.ArgumentParser(description="Music data processing.")

    parser.add_argument('--input_path', type=str, default="/mnt/disks/data/", help="Root directory of the data.")
    parser.add_argument('--output_path', type=str, default="/mnt/disks/data/", help="Path to the dataset.")
    parser.add_argument('--top_genres', type=list, default='', help="List of top Genres.")
    parser.add_argument('--sequence_size', type=int, default=1280, help="Size of the sequence.")
    parser.add_argument('--train_id', type=str, default="hierarchical_tworoots_dev", help="Training ID.")
    parser.add_argument('--sample_size', type=float, default=1, help="Size of the sample.")

    # Parse command-line arguments
    args = parser.parse_args()
    # Log an info message


    # Convert arguments to a pandas Series
    args = pd.Series(vars(args))
    print("Prepraring paths.")
    tracks_df, args = prepare_paths(args)
    if args['top_genres'] != '':
        print(f"Using top genres list. {args['top_genres']}")
        tracks_df = tracks_df[tracks_df['track_genre_top'].isin(args['top_genres'])]
    print("Crerating labels structures.")
    tracks_df, args = prepare_labels(tracks_df, args)
    print("Spliting dataset in train/test/val.")
    split_dataset(tracks_df, args)
