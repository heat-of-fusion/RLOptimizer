import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm.auto import tqdm

from train import train_model

class IntegratedEnvironment():
    def __init__(self, input_size, hidden_size, n_classes, Net, Optimizer, LEARNING_RATE = 1e-3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.Net = Net
        self.Optimizer = Optimizer
        self.LEARNING_RATE = LEARNING_RATE

        self.complexity = np.log(n_classes) + np.log(input_size)

        self.train_loss = list()
        self.valid_loss = list()
        self.train_accuracy = list()
        self.valid_accuracy = list()

        self.prev_margin = 5

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loader = None
        self.valid_loader = None
        self.dl_model = None
        self.dl_criterion = nn.CrossEntropyLoss().to(self.device)

        self.loaders_flag = False
        self.model_flag = False
        self.optimizer_flag = False

        return

    def check_loaders(self):
        assert self.loaders_flag, 'DataLoader Setting is Not Completed!\nYou Need to call .set_loaders(train_dataset, valid_dataset) function to make the environment work.'

        return

    def check_model(self):
        assert self.model_flag, 'Model Setting is Not Completed!\nYou Need to call .set_model(Net) function to make the environment work.'

        return

    def check_optimizer(self):
        assert self.optimizer_flag, 'Optimizer Setting is Not Completed!\nYou Need to call .set_optimizer(Optimizer) function to make the environment work.'

        return

    def set_loaders(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loaders_flag = True

        return 'DataLoader Setting Completed!'

    def set_datasets(self, train_dataset, valid_dataset, batch_size = 32):
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
        self.loaders_flag = True

        return 'Dataset Setting Completed!'

    def set_model(self):
        self.check_loaders()

        input_shape = list(next(iter(self.train_loader))[0].shape)

        self.dl_model = self.Net(self.input_size, self.hidden_size, self.n_classes).to(self.device)
        self.dl_model.setup(input_shape = input_shape, device = self.device)
        self.model_flag = True

        return 'Model Setting Completed!'

    def set_optimizer(self):
        self.check_loaders()
        self.check_model()

        self.dl_optimizer = self.Optimizer(self.dl_model.parameters(), lr = self.LEARNING_RATE)
        self.optimizer_flag = True

        return 'Optimizer for DL Model Setting Completed!'

    def update_param(self, beta_1_action, beta_2_action, beta_3_action):
        try:
            self.dl_optimizer.update_param(beta_1_action, beta_2_action, beta_3_action)

            return 'Optimizer Parameters(beta_1, beta_2, gamma_3) Update Completed!'

        except:
            return 'Optimizer Parameters(beta_1, beta_2, gamma_3) Update Failed!'

    def replace_loader(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        return 'DataLoader Replacement Completed!'

    def load_state_dict(self, state_dict):
        self.check_model()

        return self.dl_model.load_state_dict(state_dict)

    def state_dict(self):
        self.check_model()

        return self.dl_model.state_dict()

    def init(self):
        self.check_loaders()
        # self.check_model()
        # self.check_optimizer()

        self.train_loss = list()
        self.valid_loss = list()
        self.train_accuracy = list()
        self.valid_accuracy = list()

        self.set_model()
        self.set_optimizer()

        return torch.tensor([0.0, 1.0, self.complexity, 1.0], dtype = torch.float32), False

    def reward(self, epoch, sqrt_lv = 4):
        r = (self.valid_loss[0] - self.valid_loss[epoch]) / ((epoch + 1) ** (1 / sqrt_lv))

        return r

    def step(self, epoch):
        self.check_loaders()
        self.check_model()
        self.check_optimizer()

        loss_1, accuracy_1, loss_2, accuracy_2 = train_model(self.dl_model, self.train_loader, self.valid_loader, self.dl_criterion, self.dl_optimizer)

        self.train_loss.append(loss_1)
        self.valid_loss.append(loss_2)
        self.train_accuracy.append(accuracy_1)
        self.valid_accuracy.append(accuracy_2)

        s_1 = epoch / 100
        s_2 = self.valid_loss[max(0, epoch - self.prev_margin)] - self.valid_loss[-1]
        s_3 = self.complexity
        s_4 = self.train_loss[-1] - self.valid_loss[-1]

        reward = self.reward(epoch)
        done = False

        return torch.tensor([s_1, s_2, s_3, s_4], dtype = torch.float32), reward, done

if __name__ == '__main__':
    from DLModel import CNNet
    from CustomDataset import get_mnist
    from OptimzerAgent import OptimAgentDemoV2

    train_dataset, valid_dataset = get_mnist(train = True)

    input_size = train_dataset[0][0].shape[0]
    hidden_size = 256
    n_classes = 10
    LEARNING_RATE = 3e-4

    dummy_env = IntegratedEnvironment(input_size, hidden_size, n_classes)
    dummy_env.set_loaders(train_dataset, valid_dataset)
    dummy_env.set_model(CNNet)
    dummy_env.set_optimizer(OptimAgentDemoV2, LEARNING_RATE)

    print(f'Environment Setup Completed! {dummy_env.init()}')

    print(f'Environment Stepping Check')
    for epoch in range(50):
        print(f'Epoch {epoch} | Output: {dummy_env.step(epoch = epoch)}')
