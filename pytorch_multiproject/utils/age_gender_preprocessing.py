import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import scipy.io
import numpy as np
import pandas as pd
import imagesize
import requests
import tarfile

local_file_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop.tar')
# path where all the unpacked image files will be stored
unpacked_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')


def get_data():
    url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar'
    r = requests.get(url)
    with open(local_file_path, 'wb') as f:
        f.write(r.content)


get_data()

# Extract the files from the dataset archive
with tarfile.open(local_file_path) as tf:
    tf.extractall(path=os.path.join(ROOT_DIR, 'resources'))

mat_path = os.path.join(unpacked_path, 'wiki.mat')
dataset_info = scipy.io.loadmat(mat_path)
dataset_info = dataset_info['wiki'].flatten()

# get the image paths
images = np.concatenate(dataset_info[0][2].flatten())

# get gender data, 0 - female, 1 - male, NaN - unknown
gender = dataset_info[0][3].flatten()

# age = photo_taken - date_birth
# originally date_birth comes in matlab serial number format so we need to convert it to datetime64[D]
date_birth = dataset_info[0][0]
origin = np.datetime64('0000-01-01', 'D') - np.timedelta64(1, 'D')
date_birth = (date_birth * np.timedelta64(1, 'D') + origin).flatten()

photo_taken = dataset_info[0][1].flatten()
photo_taken_format = np.char.add(photo_taken.astype(str), '-07-01')
photo_taken_format = pd.to_datetime(photo_taken_format).values.astype('datetime64[D]')

age = (photo_taken_format - date_birth).astype('int')//365

dataset_df = pd.DataFrame({'image_path': images, 'gender': gender, 'age': age, })

# Drop nan values
original_len = len(dataset_df)
dataset_df = dataset_df.dropna()
print('{} data entries contaning nan values have been dropped.'.format(original_len - len(dataset_df)))

# Find image instances where size is < 100 px
idx_drop = []
for idx, value in dataset_df['image_path'].items():
    try:
        if imagesize.get(os.path.join(unpacked_path, value))[0] < 100:
            idx_drop.append(idx)
    except FileNotFoundError:
        print(value)
        idx_drop.append(idx)
    if idx % 5000 == 0:
        print('Iteration {}'.format(idx))
print('Indices to drop: {}'.format(len(idx_drop)))

# Drop instances where size is < 100 px
original_len = len(dataset_df)
dataset_df = dataset_df.drop(idx_drop)
print('{} data entries with incorrect image size have been dropped.'.format(original_len - len(dataset_df)))

# Drop entries with broken age
original_len = len(dataset_df)
mask = (dataset_df['age'] > 90) | (dataset_df['age'] < 1)
dataset_df = dataset_df.drop(labels = dataset_df[mask].index)
print('{} data entries with incorrect age have been dropped.'.format(original_len - len(dataset_df)))

dataset_df['gender'] = dataset_df['gender'].astype(int)
dataset_df['age'] = dataset_df['age'].astype(float)

print('-'*30)
print(dataset_df.head())

# Save path in session for csv file
INFO_SAVE_PATH = os.path.join(unpacked_path, 'dataset_info.csv')

dataset_df.to_csv(INFO_SAVE_PATH)
