import pandas as pd
import json

from tqdm import tqdm

from sklearn.utils import shuffle
from math import ceil


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


def select_dataset(df, args):
    tests = []
    trains = []
    validations = []
    
    labels_strings = df['y_true'].apply(convert_list_to_string)
    # Agrupa o DataFrame com base nos rótulos hierárquicos
    groups = df.groupby(labels_strings)
    
    count = 0
    items_count = 0
    oversampling_size = 15  # int(group_sizes.mean() + group_sizes.std() * 2)
    #print(f"oversampling_size: {oversampling_size}")
    
    for _, group in tqdm(groups):
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
            'levels_size': args.levels_size,
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

