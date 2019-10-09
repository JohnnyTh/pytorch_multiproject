import os
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
local_file_path = os.path.join(ROOT_DIR, 'resources', 'mnist')
if not os.path.exists(local_file_path):
    os.makedirs(local_file_path)

# Download the dataset from an open source
def get_mnist_data(local_file_path):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    r = requests.get(url)

    local_file_path = os.path.join(local_file_path, 'mnist.pkl.gz')
    with open(local_file_path, 'wb') as f:
        f.write(r.content)


get_mnist_data(local_file_path)
