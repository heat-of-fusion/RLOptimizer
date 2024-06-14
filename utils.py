import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
import torch.distributions as dist

from tensorflow.keras import datasets

import numpy as np
import matplotlib.pyplot as plt

import A2CNets as net
from CustomDataset import BasicImageDataset

def compute_gae(rewards, values, next_value, gamma, lam):
    values = values + [next_value]
    gae = 0
    returns = list()
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[i])

    return returns

def total_reward(valid_loss, sqrt_lv = 4):
    episode_reward = float()
    for epoch in range(len(valid_loss)):
        episode_reward += (valid_loss[0] - valid_loss[epoch]) / ((epoch + 1) ** (1 / sqrt_lv))

    return episode_reward

def get_model(Net, INPUT_SIZE, HIDDEN_SIZE, N_CLASSES, batch_shape = (32, 1, 28, 28), device = 'cuda'):
    model = Net(INPUT_SIZE, HIDDEN_SIZE, N_CLASSES).to(device)
    model.setup(input_shape = batch_shape, device = device)

    return model

def get_accuracy(pred, label):
    pred = pred.cpu().detach().numpy().argmax(axis = 1)
    label = label.cpu().detach().numpy().argmax(axis = 1)

    accuracy = (pred == label).mean()

    return accuracy

class RandomSampleLoader():
    def __init__(self):
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_label = None
        self.valid_label = None
        self.test_label = None

        return

    def download_data(self):
        (train_data, train_label), (valid_data, valid_label) = datasets.mnist.load_data()

        train_data = train_data.reshape(-1, 1, 28, 28)
        valid_data = valid_data.reshape(-1, 1, 28, 28)

        train_data = (train_data / 255.0).astype(np.float32)
        valid_data = (valid_data / 255.0).astype(np.float32)

        train_label = np.eye(10)[train_label].astype(np.float32)
        valid_label = np.eye(10)[valid_label].astype(np.float32)

        self.train_data = torch.tensor(train_data)[:-1000]
        self.train_label = torch.tensor(train_label)[:-1000]
        self.valid_data = torch.tensor(valid_data)[:-1000]
        self.valid_label = torch.tensor(valid_label)[:-1000]

        self.retrain_data = torch.tensor(train_data)[-1000:]
        self.retrain_label = torch.tensor(train_label)[-1000:]
        self.revalid_data = torch.tensor(valid_data)[-1000:]
        self.revalid_label = torch.tensor(valid_label)[-1000:]

        # print(f'Train Data Shape: {self.train_data.shape}')
        # print(f'Train Label Shape: {self.train_label.shape}')
        # print(f'Valid Data Shape: {self.valid_data.shape}')
        # print(f'Valid Label Shape: {self.valid_label.shape}')
        # print(f'Retrain Data Shape: {self.retrain_data.shape}')
        # print(f'Retrain Label Shape: {self.retrain_label.shape}')
        # print(f'Revalid Data Shape: {self.revalid_data.shape}')
        # print(f'Revalid Label Shape: {self.revalid_label.shape}')

        return 'Dataset Download Completed!'

    def get_dataloader(self, train_num, valid_num, batch_size = 32, first = False):
        train_sample_idx = torch.randperm(self.train_data.shape[0])[:min(train_num, self.train_data.shape[0])]
        valid_sample_idx = torch.randperm(self.valid_data.shape[0])[:min(valid_num, self.valid_data.shape[0])]

        train_sample_data = self.train_data[train_sample_idx]
        train_sample_label = self.train_label[train_sample_idx]

        valid_sample_data = self.valid_data[valid_sample_idx]
        valid_sample_label = self.valid_label[valid_sample_idx]

        train_dataset = BasicImageDataset(train_sample_data, train_sample_label)
        valid_dataset = BasicImageDataset(valid_sample_data, valid_sample_label)

        train_loader = [batch for batch in DataLoader(train_dataset, batch_size = batch_size, shuffle = True)]
        valid_loader = [batch for batch in DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)]

        if first:
            retrain_dataset = BasicImageDataset(self.retrain_data, self.retrain_label)
            revalid_dataset = BasicImageDataset(self.revalid_data, self.revalid_label)

            retrain_loader = [batch for batch in DataLoader(retrain_dataset, batch_size = batch_size, shuffle = True)]
            revalid_loader = [batch for batch in DataLoader(revalid_dataset, batch_size = batch_size, shuffle = True)]

        return (train_loader, valid_loader, retrain_loader, revalid_loader) if first else (train_loader, valid_loader)

def visualize(train_loss, valid_loss, train_accuracy, valid_accuracy, rewards, episode):
    plt.figure(figsize = (6 * (len(train_loss.keys()) + 1), 6 * 2))

    benchmarks = len(train_loss.keys())

    plt.subplot(2, benchmarks + 1, 1)
    plt.plot(rewards, label = 'Episode Reward')
    plt.title(f'Episode Rewards\nCurrent Episode: {episode}')
    plt.legend(loc = 'lower right')

    for idx, key in enumerate(train_loss.keys()):

        benchmark_valid_loss = np.array(valid_loss[key])
        benchmark_valid_accuracy = np.array(valid_accuracy[key])


        plt.subplot(2, benchmarks + 1, 2 + idx)
        plt.plot(train_loss[key], label = 'Train Loss')
        plt.plot(valid_loss[key], label = 'Valid Loss')
        plt.scatter(np.where(benchmark_valid_loss == benchmark_valid_loss.min()), benchmark_valid_loss.min(), c = 'red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(-5, 105)
        plt.ylim(1.45, 2.3)
        plt.title(f'Optimizer: {key} Loss\nLowest Valid Loss: {min(valid_loss[key]):.3f}')
        plt.legend(loc = 'upper right')

        plt.subplot(2, benchmarks + 1, benchmarks + 3 + idx)
        plt.plot(train_accuracy[key], label = 'Train Accuracy')
        plt.plot(valid_accuracy[key], label = 'Valid Accuracy')
        plt.scatter(np.where(benchmark_valid_accuracy == benchmark_valid_accuracy.max()), benchmark_valid_accuracy.max(), c = 'red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xlim(-5, 105)
        plt.ylim(0.2, 1.1)
        plt.title(f'Optimizer: {key} Accuracy\n\nHighest Valid Loss: {max(valid_accuracy[key]):.3f}')
        plt.legend(loc = 'lower right')

    return 'Visualizing Completed!'

if __name__ == '__main__':
    random_sampler = RandomSampleLoader()
    random_sampler.download_data()
    train_loader, valid_loader = random_sampler.get_dataloader(1000, 1000)

    print(f'Length: {len(train_loader)}, Batch Shape: {train_loader[0][0].shape}, {train_loader[0][1].shape}')