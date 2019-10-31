import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import requests
import zipfile
import pandas as pd
import numpy as np

data_root = os.path.join(ROOT_DIR, 'resources', 'gan')


def get_data():
    url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
    r = requests.get(url)
    local_file_path = os.path.join(data_root, 'horse2zebra_.zip')
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    with open(local_file_path, 'wb') as f:
        f.write(r.content)


get_data()


with zipfile.ZipFile(os.path.join(data_root, 'horse2zebra_.zip'), 'r') as zip_ref:
    zip_ref.extractall()

source_train = os.path.join(data_root, 'horse2zebra', 'trainA')
target_train = os.path.join(data_root, 'horse2zebra', 'trainB')
source_test = os.path.join(data_root, 'horse2zebra', 'testA')
target_test = os.path.join(data_root, 'horse2zebra', 'testB')

source_img_train = np.array(os.listdir(source_train))
target_img_train = np.array(os.listdir(target_train))
source_img_test= np.array(os.listdir(source_test))
target_img_test = np.array(os.listdir(target_test))

source_paths_train = np.char.add('trainA' + os.sep, source_img_train)
target_paths_train = np.char.add('trainB' + os.sep, target_img_train)
source_paths_test = np.char.add('testA' + os.sep, source_img_test)
target_paths_test = np.char.add('testB' + os.sep, target_img_test)

df_s_train = pd.DataFrame({'image_path': source_paths_train, 'designation': 'source_train'})
df_t_train = pd.DataFrame({'image_path': target_paths_train, 'designation': 'target_train'})
df_s_test = pd.DataFrame({'image_path': source_paths_test, 'designation': 'source_test'})
df_t_test = pd.DataFrame({'image_path': target_paths_test, 'designation': 'target_test'})

df = pd.concat([df_s_train, df_t_train, df_s_test, df_t_test])

df.to_csv(os.path.join(data_root, 'dataset_info.csv'))
