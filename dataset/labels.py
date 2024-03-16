
import numpy as np


def convert_label_to_string(parsed_labels):
    all_labels = []
    for p in parsed_labels:
        all_labels.append['-'.join([str(value) for value in p])]

    return all_labels

def split_label(label):
    all_labels = []
    label = label.split('-')
    for i in range(len(label)):
        all_labels.appoend('-'.join(label[:i]))

    return all_labels

def parse_label(label, label_size=5):
    # label = label.split('-')
    # preencher com 0 no caso de haver menos de 5 níveis
    labels = np.zeros(label_size, dtype=int)
    for i, label in enumerate(label):
        if i == 5:
            break
        # Aqui você pode fazer a conversão do label em um índice inteiro usando um dicionário ou outro método
        # Neste exemplo, estou apenas usando a posição da label na lista como índice
        labels[i] = label
    return labels


def get_labels_name(x, genres_df):
    levels = 5
    full_name = []
    last_level = 0
    genre_root = ""
    for genre in x.split('-'):
        genre_df = genres_df[genres_df['genre_id'] == int(genre)]
        if genre_df.empty:
            genre_name = genre_root
        else:
            genre_name = genre_df.title.values.tolist()[0]
            genre_root = genre_name

        full_name.append(genre_name)
    full_name = '>'.join(full_name)

    return full_name
    # return genres_df[genres_df['genre_id'] == int(x)].title.values.tolist()[0]


def __create_labels__(categories_df):
    data = {
        "label1": {},
        "label2": {},
        "label3": {},
        "label4": {},
        "label5": {},
        "label1_inverse": [],
        "label2_inverse": [],
        "label3_inverse": [],
        "label4_inverse": [],
        "label5_inverse": [],
        "label1_name": {},
        "label2_name": {},
        "label3_name": {},
        "label4_name": {},
        "label5_name": {},
    }

    idx = 0

    for id_x, cat in enumerate(set(categories_df.level1.values.tolist())):
        data['label1'][cat] = idx
        data['label1_inverse'].append(cat)
        data['label1_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level2.values.tolist())):
        data['label2'][cat] = idx
        data['label2_inverse'].append(cat)
        data['label2_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level3.values.tolist())):
        data['label3'][cat] = idx
        data['label3_inverse'].append(cat)
        data['label3_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level4.values.tolist())):
        data['label4'][cat] = idx
        data['label4_inverse'].append(cat)
        data['label4_count'] = idx + 1
        idx += 1

    for idx, cat in enumerate(set(categories_df.level5.values.tolist())):
        data['label5'][cat] = idx
        data['label5_inverse'].append(cat)
        data['label5_count'] = idx + 1
        idx += 1

    for cat5, cat1, cat2, cat3, cat4, name5 in categories_df.values:
        name1 = '>'.join(name5.split('>')[:1])
        name2 = '>'.join(name5.split('>')[:2])
        name3 = '>'.join(name5.split('>')[:3])
        name4 = '>'.join(name5.split('>')[:4])

        data['label1_name'][cat1] = name1
        data['label2_name'][cat2] = name2
        data['label3_name'][cat3] = name3
        data['label4_name'][cat4] = name4
        data['label5_name'][cat5] = name5

    return data


## Get complete genre structure
def get_all_structure(estrutura, df_genres):
    ## Get structure from df_genres
    def get_all_structure_from_df(estrutura, df_genres, structure=[]):
        if estrutura == 0:
            return structure
        else:
            structure.append(int(estrutura))
            get_all_structure_from_df(df_genres[df_genres["genre_id"] == int(estrutura)].parent.values[0], df_genres,
                                      structure)
            return structure

    return get_all_structure_from_df(estrutura, df_genres, structure=[])