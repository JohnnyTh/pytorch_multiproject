import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import mock
from data.generic_dataset import GenericDataset
from torch.utils.data import Dataset

data = {
        '/images': ['img_1.jpg', 'img_2.jpg', 'img_3.png', 'not_img.txt'],
        '/txt_files': ['text_1.txt', 'not_text.rar', 'not_text_2.jpg', 'text_2.txt', 'text_3.txt'],
        '/mixed_1': ['img_1.jpg', 'img_2.jpeg', 'text_1.txt', 'img_3.png', 'dword_1.doc'],
        '/single_test': ['1.jpg', '2.txt', '3.png', '4.jpeg', '5.txt', '6.txt', '7.jpg']
    }


def side_effect(val):
    return data[val]


def get_test_obj(self, case):
    test = None
    if case == 'multiple_fold':
        test = GenericDataset(data.keys(), (('.jpg', '.png'), ('.txt'), ('.jpeg'), ()))
    elif case == 'single_fold':
        test = GenericDataset(['/single_test'], (('.txt'), ('.jpg')))
    return test


@mock.patch('data.generic_dataset.os.listdir', side_effect=side_effect)
def test_inheritance(self):
    test = get_test_obj(self, 'multiple_fold')
    assert isinstance(test, Dataset)


@mock.patch('data.generic_dataset.os.listdir', side_effect=side_effect)
def test_all_folders_captured(self):
    test = get_test_obj(self, 'multiple_fold')
    captured_folders = {entry['root'] for entry in test._found_dataset}
    assert set(data.keys()) == captured_folders


@mock.patch('data.generic_dataset.os.listdir', side_effect=side_effect)
def test_files_captured(self):
    test = get_test_obj(self, 'multiple_fold')
    captured_img = set(test._found_dataset[0]['names'])
    captured_txt = set(test._found_dataset[1]['names'])
    captured_mixed = set(test._found_dataset[2]['names'])
    assert captured_img == {'img_1.jpg', 'img_2.jpg', 'img_3.png'}
    assert captured_txt == {'text_1.txt', 'text_2.txt', 'text_3.txt'}
    assert captured_mixed == {'img_2.jpeg'}


@mock.patch('data.generic_dataset.os.listdir', side_effect=side_effect)
def test_single_dir_captured(self):
    test = get_test_obj(self, 'single_fold')
    captured_txt = set(test._found_dataset[0]['names'])
    captured_img = set(test._found_dataset[1]['names'])
    assert len(test._found_dataset) == 2
    assert captured_txt == {'2.txt', '5.txt', '6.txt'}
    assert captured_img == {'1.jpg', '7.jpg'}
