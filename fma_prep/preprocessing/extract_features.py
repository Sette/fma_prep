# %%
import os
import pandas as pd
import numpy as np
import torch
import multiprocessing as mp

from essentia.standard import TensorflowInputMusiCNN, FrameGenerator, MonoLoader

# %%
from tqdm.notebook import tqdm

# %%
tqdm.pandas()

# %%
args = pd.Series({
    "dataset_path":"/home/bruno/storage/data/fma/fma_large",
    "output_path":"/home/bruno/storage/data/fma/fma_large",
    "embeddings":"musicnn",
    "top_genres": ["Rock", "Electronic"]
})

# %%

# In[16]:

metadata_path_fma = os.path.join(args.dataset_path, "fma_metadata")



# %%
df = pd.read_csv(os.path.join(metadata_path_fma,"tracks_valid.csv"))

# %%
if args.top_genres:
    print(f"Using top genres list. {args['top_genres']}")
    df = df[df['track_genre_top'].isin(args['top_genres'])]


df = df[['track_id','file_path']]

# %%
total = len(df) / 1024

# %%
total

# %%
model = TensorflowInputMusiCNN()
print("Modelo carregado com sucesso na GPU.")

# %%
def create_dir(path):
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True


def extract_audio(dataset_path, filename):
    ### Configuração do model para extrair a representação do aúdio
    # model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
    file_path = os.path.join(dataset_path, filename)
    audio = MonoLoader(filename=file_path, sampleRate=16000)()
    
    # Concatenar todas as ativações em um único vetor
    final_feature_vector = audio.tolist()
    return final_feature_vector


def extract_feature(dataset_path, filename):
    ### Configuração do model para extrair a representação do aúdio
    # model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
    
    model = TensorflowInputMusiCNN()
    
    file_path = os.path.join(dataset_path, filename)
    audio = MonoLoader(filename=file_path, sampleRate=16000)()
    # Criar frames de 512 samples
    frames = list(FrameGenerator(audio, frameSize=512, hopSize=256))
    activations = []
    for frame in frames:
        # Executar a inferência na GPU
        activations.append(model(frame))  # Converte de volta para numpy se necessário

    # Concatenar todas as ativações em um único vetor
    final_feature_vector = np.concatenate(activations, axis=0).tolist()
    return final_feature_vector


def extract_feature_wrapper(dataset_path, filename):
    return extract_feature(dataset_path, filename)

# %%
def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path

# In[36]:

# %%
#####
## Pt files
#####

def create_example(data):
    track_id, music = data

    example = {
        'features': music,
        'track_id': track_id
    }

    return example

def process_df_to_pt(df, pt_path, args, model):
    features = df.file_path.apply(lambda filename: extract_feature(args.dataset_path, filename))

    df.loc[:, 'features'] = features

    df.drop(columns=['file_path'], inplace=True)

    print("Extraiu as features")

    path = os.path.join(pt_path, f"{str(0).zfill(10)}.pt")

    create_dir(pt_path)

    pt_records = [create_example(data) for data in df.values]
    torch.save(pt_records, path)

def process_df_to_pt_batch(df,i,count,batch_size,pt_path, model, args):
    batch_df = df[i:i+batch_size]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        features = pool.starmap(extract_feature_wrapper, 
                            [(args.dataset_path, filename) for filename in batch_df.file_path])
    
    batch_df.loc[:, 'features'] = features
    print("Extraiu as features")
    batch_df.drop(columns=['file_path'], inplace=True)

    path = os.path.join(pt_path, f"{str(count).zfill(10)}.pt")

    create_dir(pt_path)

    pt_records = [create_example(data) for data in batch_df.values]
    torch.save(pt_records, path)

    print(f"{count} {len(pt_records)} {path}")
    count += 1

def generate_pt_files_batch(df, args, model):

    pt_path = os.path.join(args.output_path, "pt_rock_electronic", args.embeddings)

    batch_size = 1024 * 1  # 1k records from each file batch

    total = len(df) / batch_size

    for count,i in enumerate(range(0, len(df), batch_size)):
        process_df_to_pt_batch(df, i, count, batch_size, pt_path, model, args)
        print(f"{count}/{int(total)} {len(df)} {pt_path}")


def generate_pt_files(df, args, model):
    pt_path = os.path.join(args.output_path, "pt_rock_electronic", args.embeddings)
    process_df_to_pt(df, pt_path, args, model)
   

# %%
#df = df.sample(10)

# %%
generate_pt_files_batch(df, args, model)

# %%


# %%



