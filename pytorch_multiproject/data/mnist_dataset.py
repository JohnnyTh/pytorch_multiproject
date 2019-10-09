from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle


class MnistDataset(Dataset):
    """Args:
            gzip_data (file object):  an object exctracted from open zip file.
            train_or_test (string): can be either 'train' 'test'. Needed to
                            specify which of the tuples to load using pickle.
            img_size (integer): size of the square images in dataset.
            transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, gzip_data, train_or_test, img_size, transform=transforms.ToTensor()):
        if train_or_test == 'train':
            data = pickle.load(gzip_data, encoding='latin-1')[0]
        elif train_or_test == 'test':
            data = pickle.load(gzip_data, encoding='latin-1')[2]
        else:
            raise Exception("'train' ot 'test' keyword not specified!")
        # reshape the image data into (N, H, W, C) if you will use ToTensor()
        self.data, self.labels = data
        self.data = self.data.reshape(-1, img_size, img_size, 1)
        self.transform = transform
        self.mode = train_or_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __str__(self):
        info = """The {} dataset consists of {} images of size {} and {} labels.
        \nImage matrix format: {}, label matrix format: {}.
               """.format(self.mode, self.data.shape[0],
                          self.data.shape[1:], self.labels.shape[0],
                          self.data.shape, self.labels.shape)

        return info