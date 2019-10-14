import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import pytest
import shutil
from data.generic_dataset import GenericDataset
from torch.utils.data import Dataset

data = {
    'images': ['img_1.jpg', 'img_2.jpg', 'img_3.png', 'not_img.txt'],
    'txt_files': ['text_1.txt', 'not_text.rar', 'not_text_2.jpg', 'text_2.txt', 'text_3.txt'],
    'mixed_1': ['img_1.jpg', 'img_2.jpeg', 'text_1.txt', 'img_3.png', 'dword_1.doc'],
    'single_test': ['1.jpg', '2.txt', '3.png', '4.jpeg', '5.txt', '6.txt', '7.jpg']
}

# Create the temporary test files from data dict
temp_dir = os.path.join(ROOT_DIR, 'temp_test')
dirs = []
for dir, _ in data.items():
    dirs.append(os.path.join(temp_dir, dir))
    os.makedirs(dirs[-1], exist_ok=True)
    for file in data[dir]:
        with open(os.path.join(temp_dir, dir, file), "w") as f:
            f.write(' ')

test = GenericDataset(dirs, (('.jpg', '.png'), ('.txt'), ('.jpeg'), ()))
test_dataset = test.get_dataset()

single_dir_test = os.path.join(temp_dir, 'single_test')
test_single_dir = GenericDataset([single_dir_test], (('.txt'), ('.jpg')))
test_single_dir_data = test_single_dir.get_dataset()


def test_inheritance():
    assert isinstance(test, Dataset)


def test_all_folders_captured():
    captured_folders = [entry['root'] for entry in test_dataset]
    assert set(dirs) == set(captured_folders)


def test_files_captured():
    captured_img = set(test_dataset[0]['names'])
    captured_txt = set(test_dataset[1]['names'])
    captured_mixed = set(test_dataset[2]['names'])
    assert captured_img == {'img_1.jpg', 'img_2.jpg', 'img_3.png'}
    assert captured_txt == {'text_1.txt', 'text_2.txt', 'text_3.txt'}
    assert captured_mixed == {'img_2.jpeg'}


def test_single_dir_captured():
    captured_txt = set(test_single_dir_data[0]['names'])
    captured_img = set(test_single_dir_data[1]['names'])
    assert len(test_single_dir_data) == 2
    assert captured_txt == {'2.txt', '5.txt', '6.txt'}
    assert captured_img == {'1.jpg', '7.jpg'}


# Remove the temp_test directory after we're done
shutil.rmtree(temp_dir)
