import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import datasets

# from torchvision import datasets
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, 1:-1]
        self.labels = data[:, -1]
        self.label_dict = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }

        return

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature_tensor = torch.tensor(self.features[idx].astype(np.float32))
        label = torch.tensor(np.eye(3)[self.label_dict[self.labels[idx]]].astype(np.float32))

        return feature_tensor, label

class BasicImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_cifar10():
    (train_images, train_labels), (valid_images, valid_labels) = datasets.cifar10.load_data()
    train_images = train_images.transpose(0, 3, 1, 2)[:1024]
    valid_images = valid_images.transpose(0, 3, 1, 2)[:512]
    # import matplotlib.pyplot as plt
    # plt.plot(train_labels[:, 0])
    # plt.show()

    train_images = (train_images / 255.0).astype(np.float32)
    valid_images = (valid_images / 255.0).astype(np.float32)

    train_labels = np.eye(10)[train_labels[:, 0]].astype(np.float32)
    valid_labels = np.eye(10)[valid_labels[:, 0]].astype(np.float32)

    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels)
    valid_images = torch.tensor(valid_images)
    valid_labels = torch.tensor(valid_labels)

    train_dataset = BasicImageDataset(train_images, train_labels)
    valid_dataset = BasicImageDataset(valid_images, valid_labels)

    return train_dataset, valid_dataset

def get_mnist(train = True):
    (train_images, train_labels), (valid_images, valid_labels) = datasets.mnist.load_data()

    # plt.scatter(list(range(len(train_labels))), train_labels)
    # plt.show()

    train_images = train_images.reshape(-1, 1, 28, 28)[0 if train else 1000 : 1000 if train else 2000]
    valid_images = valid_images.reshape(-1, 1, 28, 28)[0 if train else 1000 : 1000 if train else 2000]

    train_images = (train_images / train_images.max()).astype(np.float32)
    valid_images = (valid_images / valid_images.max()).astype(np.float32)

    train_labels = train_labels[0 if train else 1000 : 1000 if train else 2000]
    valid_labels = valid_labels[0 if train else 1000 : 1000 if train else 2000]

    train_labels = np.eye(10)[train_labels].astype(np.float32)
    valid_labels = np.eye(10)[valid_labels].astype(np.float32)

    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels)
    valid_images = torch.tensor(valid_images)
    valid_labels = torch.tensor(valid_labels)

    train_dataset = BasicImageDataset(train_images, train_labels)
    valid_dataset = BasicImageDataset(valid_images, valid_labels)

    return train_dataset, valid_dataset


if __name__ == '__main__':
    df_iris = pd.read_csv(f'dataset/Iris.csv')
    print(df_iris.columns)

    iris_dataset = IrisDataset(df_iris.values)
    print(len(iris_dataset))
    print(iris_dataset[0])

    dummy_train_dataset, dummy_valid_dataset = get_mnist()
    plt.imshow(dummy_train_dataset[0][0][0])
    plt.title(dummy_train_dataset[0][1])
    plt.show()

    plt.imshow(dummy_valid_dataset[0][0][0])
    plt.title(dummy_valid_dataset[0][1])
    plt.show()