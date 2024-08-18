
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
        all_labels.append('-'.join(label[:i]))

    return all_labels

def parse_label(label, label_size=5):
    # label = label.split('-')
    # preencher com 0 no caso de haver menos de 5 níveis

    labels = np.zeros(label_size, dtype=int)
    for i, label in enumerate(label):
        if i == label_size-1:
            break
        # Aqui você pode fazer a conversão do label em um índice inteiro usando um dicionário ou outro método
        # Neste exemplo, estou apenas usando a posição da label na lista como índice
        labels[i] = label

    labels = '-'.join([str(value) for value in labels])

    return labels


def get_labels_name(x, genres_df):
    full_name = []
    genre_root = ""
    for genre in x:
        if genre != "":
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

def __create_labels__(categories_df, max_depth):
    data = {}
    data[f'levels_size'] = []
    for level in range(1, max_depth+1):
        level_name = f'level{level}'
        data[level_name] = {}
        data[f'{level_name}_name'] = {}
        data[f'{level_name}_inverse'] = []
        slice_df = categories_df[level_name].drop_duplicates()
        for idx, cat in enumerate(slice_df.values):
            if cat != 0:
                cat = int(cat)
                name = categories_df[categories_df[level_name] == cat][f'level{max_depth}_name'].values.tolist()[0]
                data[level_name][cat] = idx
                data[f'{level_name}_name'][cat] = '>'.join(name.split('>')[:level])
                data[f'{level_name}_inverse'].append(cat)
        
        data['levels_size'].append(len(data[f'{level_name}']))

    for values in categories_df.values:
        label_name = values[-1]
        labels = values[:-1]
        for level, label in zip(range(1, max_depth+1), labels):
            if label != "":
                data[f'level{level}_name'][int(label)] = '>'.join(label_name.split('>')[:level])
            
    return data
        



def get_all_structure(estrutura, df_genres):
    # Inicializar uma lista para armazenar a estrutura completa
    structure = []

    # Iterar até chegar ao topo da hierarquia (quando estrutura for 0)
    while estrutura != 0:
        # Adicionar o nó atual à estrutura
        structure.append(int(estrutura))
        # Obter o pai do nó atual
        parent = df_genres[df_genres["genre_id"] == int(estrutura)].parent.values[0]
        # Atualizar o nó atual para o pai
        estrutura = parent

    structure.reverse()
    return structure