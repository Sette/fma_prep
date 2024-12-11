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

def transform_labels(label_lists):
    """
    Transforms a list of lists into a single string with the format `X.Y.Z@A.B.C`.

    Args:
        label_lists (list of lists): A list of lists of labels.

    Returns:
        str: A string representation in the format `X.Y.Z@A.B.C`.
    """
    # Join each sublist with "." and then join the resulting strings with "@"
    return "@".join(".".join(map(str, sublist)) for sublist in label_lists)



def max_hierarchy_depth_from_lists(label_lists):
    """
    Calculates the maximum hierarchy depth from a list of lists.

    Args:
        label_lists (list of lists): A list containing hierarchical labels.

    Returns:
        int: The maximum depth of the hierarchy.
    """
    return max(len(sublist) for sublist in label_lists)

def group_and_remove_redundant(structures):
    """
    Group sublists by their size and remove redundant sublists.
    
    Args:
        structures (list of lists): A list of lists to process.
    
    Returns:
        list of lists: Processed list with no redundant sublists.
    """
    # Step 1: Group lists by their size
    grouped = {}
    for sublist in structures:
        size = len(sublist)
        if size not in grouped:
            grouped[size] = []
        grouped[size].append(sublist)
    
    # Step 2: Sort keys in descending order (largest lists first)
    sorted_sizes = sorted(grouped.keys(), reverse=True)
    
    # Step 3: Remove redundant sublists
    unique_structures = []
    seen = set()
    
    for size in sorted_sizes:
        for sublist in grouped[size]:
            # Check if the sublist is already a subset of another added list
            sublist_set = set(sublist)
            if not any(sublist_set.issubset(set(added)) for added in unique_structures):
                unique_structures.append(sublist)
    
    return unique_structures
    

def remover_sublistas_redundantes(lista_de_listas):
    max_depth = max([len(value) for value in lista_de_listas])
    new_sublist = []
    for sublista in lista_de_listas:
        get = True
        if len(sublista) != max_depth:
            for s_list in new_sublist:
                if any(sub in s_list for sub in sublista):
                    get = False

        if get:
            new_sublist.append(sublista)

    return new_sublist


def prepare_paths(args):
    ## Define job paths
    input_path = args.input_path
    output_path = args.output_path

    args['job_path'] = os.path.join(output_path, "trains" ,args.train_id)
    args['tfrecord_path'] = os.path.join(args.job_path, "tfrecrods")
    args['torch_path'] = os.path.join(args.job_path, "torch")
    args['metadata_path'] = os.path.join(input_path, 'fma_metadata')
    args['metadata_train_path'] = os.path.join(args['job_path'], "metadata.json")
    args['mlb_path'] = os.path.join(args['job_path'], "mlb.pkl")
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



def group_labels_by_level(df, max_depth):
    # Initialize empty lists for each level based on max_depth
    levels = [[] for _ in range(max_depth)]
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Iterate over each level and append the labels to the corresponding list
        for level in range(max_depth):
            level_labels = []
            for label in row['y_true']:
                if level < len(label):
                    level_labels.append(label[level])
            levels[level].append(list(set(level_labels)))
    
    # Return the grouped labels by level
    return levels



# Função para dividir os rótulos em níveis
def split_labels(all_labels, level):
    return [label[level] for labels in all_labels for label in labels]


def create_labels(tracks_df, args):
    ##### Labels
     # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # Mapear os identificadores numéricos de gêneros para os nomes dos gêneros
    
    # Inicialize uma lista para armazenar todos os caminhos de gêneros para cada exemplo
    all_labels = []
    all_structures = []
    depths = []
    # Iterar sobre as faixas e seus gêneros associados
    for track_genres in tracks_df['track_genres_all']:
        structures = [get_all_structure(genre_id, genres_df) for genre_id in track_genres]
        structures.sort(key=len, reverse=True)
        depths.append(max_hierarchy_depth_from_lists(structures))
        structures = group_and_remove_redundant(structures)
        labels = transform_labels(structures)
        categories = labels.split('@')
        all_structures.extend(categories)
        all_labels.append(labels)

    all_categories = {"labels": list(set(all_structures))}
    # Write labels file

    with open(args.categories_labels_path, 'w+') as f:
        f.write(json.dumps(all_categories))

    ## Get structure form hierarchical classification
    tracks_df.loc[:, 'y_true'] = all_labels
    args['max_depth'] = max(depths)

    return tracks_df

    

def binarize_labels(tracks_df, args):
    ##### Labels
    mlbs = []
    
    grouped_labels = group_labels_by_level(tracks_df, args.max_depth)
    
    labels_name = []
    for level, level_labels in enumerate(grouped_labels):
        labels_name.append(f'level{level+1}')
        
        # Remove valores None
        #level_labels = [label if type(label) == list else [label] for label in level_labels if label is not None]
        
        # Cria e aplica o MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binary_labels = mlb.fit_transform(level_labels).tolist()
       
        mlbs.append(mlb)
        
        binary_labels = [binary_labels[i] if i < len(binary_labels) else [0] * len(mlb.classes_) for i in range(len(tracks_df))]

        tracks_df.loc[:, labels_name[level]] = binary_labels

    # Serializar a lista de mlb
    with open(args.mlb_path, 'wb') as file:
        pickle.dump(mlbs, file)

    tracks_df['all_binarized'] = tracks_df.apply(lambda row: [sublist for sublist in row[labels_name]], axis=1)
    tracks_df = tracks_df[['track_id', 'y_true', 'all_binarized']]


def split_dataset(tracks_df,args):
    #### Split dataset

    df_train, df_test, df_val = select_dataset(tracks_df)

    args['val_path'] = os.path.join(args.tfrecord_path, 'val')
    args['test_path'] = os.path.join(args.tfrecord_path, 'test')
    args['train_path'] = os.path.join(args.tfrecord_path, 'train')

    args['val_torch_path'] = os.path.join(args.torch_path, 'val')
    args['test_torch_path'] = os.path.join(args.torch_path, 'test')
    args['train_torch_path'] = os.path.join(args.torch_path, 'train')

    args['train_csv'] = os.path.join(args.job_path, "train.csv")
    args['test_csv'] = os.path.join(args.job_path, "test.csv")
    args['val_csv'] = os.path.join(args.job_path, "val.csv")
    
    feature_path = os.path.join(args.input_path, 'fma_large')

    df_features = load_features(feature_path)

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

    parser.add_argument('--input_path', type=str, default="/home/bruno/storage/data/fma", help="Root directory of the data.")
    parser.add_argument('--output_path', type=str, default="/home/bruno/storage/data/fma/trains", help="Path to the dataset.")
    parser.add_argument('--top_genres', type=str, nargs='+', default=[], help="List of top genres.")
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
    # Converter a string de volta para uma lista
    if args.top_genres:
        print(f"Using top genres list. {args['top_genres']}")
        tracks_df = tracks_df[tracks_df['track_genre_top'].isin(args['top_genres'])]
    print("Creating labels structures.")
    #return tracks_df, args
    create_labels(tracks_df, args)
    print("Binarizing labels structures.")
    binarize_labels(tracks_df, args)
    print("Spliting dataset in train/test/val.")
    split_dataset(tracks_df, args)