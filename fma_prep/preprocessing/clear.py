#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -r ../requirements.txt


# In[2]:


get_ipython().system('pip install essentia')


# In[1]:


import pandas as pd
import numpy as np
import ast
import json
import os
from essentia.standard import MonoLoader


# In[2]:


from tqdm.notebook import tqdm


# In[3]:


tqdm.pandas()


# In[4]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large"
})


# In[5]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:

metadata_path_fma = os.path.join(base_path,"fma_metadata")



# In[6]:


# Cria um dicionário que associa o ID de cada música aos IDs de seus gêneros musicais
tracks_df = pd.read_csv(os.path.join(metadata_path_fma,'tracks.csv'), header=[0,1], index_col=0)


# In[7]:


tracks_df.reset_index(inplace=True)


# In[8]:


tracks_df.columns = ['_'.join(col) for col in tracks_df.columns.values]


# In[9]:


tracks_df.columns


# In[10]:


tracks_df = tracks_df[['track_genres_all','track_genres','track_id_']]


# In[11]:


tracks_df.reset_index(inplace=True)


# In[12]:


tracks_df.dropna(inplace=True)


# In[13]:


tracks_df.track_genres_all.value_counts()


# In[14]:


tracks_df["track_genres_all"] = tracks_df.track_genres_all.apply(lambda x : ast.literal_eval(x))


# In[15]:


# Remover linhas com genre_id vazio
tracks_df = tracks_df[tracks_df['track_genres_all'].map(len) > 0]


# In[16]:


tracks_df


# In[17]:


def find_path(track_id, dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path, folder_id, track_id+'.mp3')
    return file_path
# In[36]:


tracks_df['file_path'] = tracks_df.track_id_.apply(lambda x: find_path(str(x), args.dataset_path))



# In[18]:


## Fazer o valid genre aqui


# In[19]:


def valid_music(file_path):
    try:
        # we start by instantiating the audio loader:
        loader = MonoLoader(filename=file_path)
    
        return True
    except:
        return False


# In[20]:


valid_music('/mnt/disks/data/fma/fma_large/000/000003.mp3')


# In[21]:


tracks_df.loc[:,'valid'] = tracks_df.file_path.progress_apply(lambda x: valid_music(x))


# In[22]:


tracks_df_filter = tracks_df[tracks_df['valid'] == True]


# In[23]:


def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path


# In[24]:


tracks_df_filter['file_path'] = tracks_df_filter.track_id_.apply(lambda x: find_path(str(x),args.dataset_path))


# In[25]:


tracks_df_filter.iloc[0].file_path


# In[26]:


tracks_df_filter = tracks_df_filter[['track_genres_all','track_genres','track_id_','file_path']]


# In[27]:


tracks_df_filter.to_csv(os.path.join(metadata_path_fma,'tracks_valid.csv'),index=False)


# In[ ]:





# In[ ]:




