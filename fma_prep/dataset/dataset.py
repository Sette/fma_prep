import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

import os
import json
import math

from tqdm import tqdm

from sklearn.utils import shuffle
from math import ceil

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO)

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

def load_features(dataset_path):
    dataset_path = [os.path.join(dataset_path, path) for path in os.listdir(dataset_path) if path.endswith('.tfrecord')]
    dataset = get_dataset(dataset_path)

    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['features', 'track_id']
    )

    df.dropna(inplace=True)

    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print('Erro ao carregar features')
    return df


BUFFER_SIZE = 10

class HMCDatasetFeatures(Dataset):
    def __init__(self, files):
        self.files = [os.path.join(files, f) for f in os.listdir(files)]  # Lista de arquivos
        self.files.sort()  # Ordena para manter a ordem consistente
        self.buffer = []  # Buffer para armazenar dados carregados temporariamente
        self.buffer_index = 0  # Índice do buffer
    
    def load_file(self, file_path):
        """ Carrega um único arquivo e move para GPU """
        loaded_data = torch.load(file_path, map_location="cpu")  # Carregar na CPU primeiro
        for example in loaded_data:
            example["features"] = torch.tensor(example["features"], dtype=torch.float32, device="cuda")
        return loaded_data

    def __len__(self):
        return sum(len(torch.load(f, map_location="cpu")) for f in self.files)  # Tamanho total

    def __getitem__(self, idx):
        """ Retorna um item específico, carregando arquivos sob demanda """
        if self.buffer_index <= idx < self.buffer_index + len(self.buffer):
            # Se o índice estiver no buffer, apenas retorna
            example = self.buffer[idx - self.buffer_index]
        else:
            # Carregar novo buffer
            file_index = idx // BUFFER_SIZE  # Descobre qual arquivo carregar
            self.buffer = self.load_file(self.files[file_index])  # Carrega o arquivo
            self.buffer_index = file_index * BUFFER_SIZE  # Atualiza índice base
            example = self.buffer[idx - self.buffer_index]  # Pega o exemplo correto
        
        return example["track_id"], example["features"]
    def to_dataframe(self):
        """ Converte o dataset inteiro para um DataFrame Pandas """
        all_data = []
        for file in tqdm(self.files, desc="Carregando arquivos"):
            loaded_data = self.load_file(file)  # Carrega os dados do arquivo
            for example in loaded_data:
                track_id = example["track_id"]
                features = example["features"].cpu().numpy()  # Converte tensor para numpy
                all_data.append({"track_id": track_id, "features": features})

        df = pd.DataFrame(all_data)
        return df


def get_csv_dataset(files_path):
    # create the dataset
    logging.info("Carregando dataframes de features.")
    all_data = []
    total = len(files_path)
    for count,file_path in enumerate(files_path):
        logging.info(f"Carregando dataframe {total}/{count}.")
        dataset = pd.read_csv(file_path)
        all_data.append(dataset)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_csv_features(dataset_path):
    files_path = [os.path.join(dataset_path, path) for path in os.listdir(dataset_path) if path.endswith('.csv')]
    dataset = get_csv_dataset(files_path)

    dataset.dropna(inplace=True)

    try:
        dataset.feature = dataset.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print('Erro ao carregar features')
    return dataset

# Função para converter listas em strings
def convert_list_to_string(lst):
    return '_'.join([str(x) for x in lst])

def __split_data__(group, percentage=0.1):
    if len(group) == 1:
        return group, group

    shuffled = shuffle(group.values)
    finish_test = int(ceil(len(group) * percentage))

    first = pd.DataFrame(shuffled[:finish_test], columns=group.columns)
    second = pd.DataFrame(shuffled[finish_test:], columns=group.columns)

    return first, second


def select_dataset(df):
    tests = []
    trains = []
    validations = []

    df = df[['track_id','file_path','track_genre_top','y_true' ]]
    
    logging.info("Agrupando o DataFrame com base nos rótulos hierárquicos")
    # Agrupa o DataFrame com base nos rótulos hierárquicos
    groups = df.groupby(df['y_true'])
    
    count = 0
    items_count = 0
    oversampling_size = 20  # int(group_sizes.mean() + group_sizes.std() * 2)
    #print(f"oversampling_size: {oversampling_size}")
    
    for _, group in groups:
        test, train_to_split = __split_data__(group, 0.2)  # 20%
        train_to_split = train_to_split
        validation, train = __split_data__(train_to_split, 0.1)  # %10
    
        tests.append(test)
        validations.append(validation)
    
        ## this increase the numner of samples when classes has low quantity
        count_train = len(train)
        if count_train < oversampling_size:
            #print(f'Oversampling: {train.y_true.iloc[0]}')
            train = train.sample(oversampling_size, replace=True)
    
        trains.append(train)
    
        count += 1
        items_count += count_train

    logging.info("Concatenando os DataFrames de teste, validação e treino")
    df_test = pd.concat(tests, sort=False).sample(frac=1).reset_index(drop=True)
    # .to_csv(dataset_testset_path, index=False,quoting=csv.QUOTE_ALL)
    df_val = pd.concat(validations, sort=False).sample(frac=1).reset_index(drop=True)
    df_train = pd.concat(trains, sort=False).sample(frac=1).reset_index(drop=True)

    return df_train, df_test, df_val

def create_metadata(args):
    with open(args.metadata_train_path, 'w+') as f:
        f.write(json.dumps({
            'sequence_size': args.sequence_size,
            'max_depth': args.max_depth,
            'val_path': args.val_path,
            'train_path': args.train_path,
            'test_path': args.test_path,
            'val_torch_path': args.val_torch_path,
            'train_torch_path': args.train_torch_path,
            'test_torch_path': args.test_torch_path,
            'val_csv': args.val_csv,
            'train_csv': args.train_csv,
            'test_csv': args.test_csv,
            'trainset_count': args.train_len,
            'validationset_count': args.val_len,
            'testset_count': args.test_len
        }))

