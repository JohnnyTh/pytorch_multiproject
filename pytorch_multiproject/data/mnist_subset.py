from torch.utils.data import Subset


class MnistSubset(Subset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable, optional): Optional transform to be applied
                on a subset.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img = self.dataset[self.indices[idx]][0]
        label = self.dataset[self.indices[idx]][1]

        if self.transform:
            img = self.transform(img)
        return img, label