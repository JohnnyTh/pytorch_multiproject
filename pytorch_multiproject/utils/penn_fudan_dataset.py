import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import requests
import zipfile

data_root = os.path.join(ROOT_DIR, 'resources')


def get_data():
    url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
    r = requests.get(url)
    local_file_path = os.path.join(data_root, 'PennFudanPed.zip')
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    with open(local_file_path, 'wb') as f:
        f.write(r.content)


get_data()


with zipfile.ZipFile(os.path.join(data_root, 'PennFudanPed.zip'), 'r') as zip_ref:
    zip_ref.extractall(path=data_root)
