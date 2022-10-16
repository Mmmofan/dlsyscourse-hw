import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = img[:, ::-1, :]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        h, w, c = img.shape
        new_im = np.zeros([h + self.padding * 2, w + self.padding * 2, c])
        new_im[self.padding: self.padding + h,
            self.padding: self.padding + w, :] = img
        coord_x = self.padding + shift_x
        coord_y = self.padding + shift_y
        return new_im[coord_x: coord_x + h, coord_y: coord_y + w, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.idx = 0
        self.len = len(self.dataset) // self.batch_size
        if self.shuffle:
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            self.ordering = np.array_split(tmp_range,
                range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.idx < self.len:
            data = self.dataset[self.ordering[self.idx]]
            self.idx += 1
            return [Tensor(x) for x in data]
        else:
            raise StopIteration
        ### END YOUR SOLUTION

    def __len__(self):
        return len(self.dataset)


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        import gzip, struct
        with gzip.open(image_filename) as f:
            _, num, ros, cols = struct.unpack('>4I', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        with gzip.open(label_filename) as f:
            _, num = struct.unpack('>2I', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min())
        self.images = self.images.astype(np.float32)
        assert self.images.shape[0] == self.labels.shape[0] == num

        self.transforms = transforms

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]
        if len(img.shape) > 1:
            for b in range(img.shape[0]):
                im = img[b].reshape(28, 28, 1)
                if self.transforms is not None:
                    for trans in self.transforms:
                        im = trans(im)
                    img[b] = im.reshape(-1)
            return (img, label)
        else:
            img = img.reshape(28, 28, 1)
            if self.transforms is not None:
                for trans in self.transforms:
                    img = trans(img)
            return (img.reshape(-1), label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
