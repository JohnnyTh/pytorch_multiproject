import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import requests
import zipfile

data_root = os.path.join(ROOT_DIR, 'resources')


def get_data():
    url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
    r = requests.get(url)
    local_file_path = os.path.join(data_root, 'horse2zebra_.zip')
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    with open(local_file_path, 'wb') as f:
        f.write(r.content)


get_data()


with zipfile.ZipFile(os.path.join(data_root, 'horse2zebra_.zip'), 'r') as zip_ref:
    zip_ref.extractall(path=data_root)
