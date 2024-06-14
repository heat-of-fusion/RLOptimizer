import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(FCNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim = 1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_classes)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.setup_comp = False

        return

    def setup(self, input_shape, device):
        dummy_input = torch.zeros(input_shape).to(device)
        dummy_output = self.forward(dummy_input)
        self.setup_comp = True
        print(f'FCNet Model Setup Completed\nInput Size: {input_shape}\nOutput Shape: {dummy_output.shape}')

        return

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.softmax(self.fc4(x))

        return x

class CNNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(CNNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim = 1)

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size = (3, 3), stride = (1, 1), padding = 'same')
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size = (3, 3), stride = (1, 1), padding = 'same')

        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.bn4 = nn.BatchNorm2d(hidden_size)

        self.max_pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.max_pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.setup_comp = False

        return

    def setup(self, input_shape, device):
        dummy_input = torch.zeros(input_shape).to(device)
        dummy_output = self.forward(dummy_input)

        batch_size = input_shape[0]
        conv_out = dummy_output.view(batch_size, -1).shape[-1]

        self.fc1 = nn.Linear(conv_out, self.hidden_size, device = device)
        self.fc2 = nn.Linear(self.hidden_size, self.n_classes, device = device)

        self.bn5 = nn.BatchNorm1d(self.hidden_size, device = device)

        self.setup_comp = True
        # print(f'CNNet Model Setup Completed\nInput Size: {input_shape}\nOutput Shape: torch.Size({[batch_size, self.n_classes]})')

        return

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.max_pool2(self.leaky_relu(self.bn4(self.conv4(x))))

        if self.setup_comp:
            x = x.view(x.shape[0], -1)

            x = self.leaky_relu(self.bn5(self.fc1(x)))
            x = self.softmax(self.fc2(x))

        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Available Device: {device}')

    dummy_fcnet = FCNet(4, 256, 3).to(device)
    dummy_input = torch.zeros((64, 4)).to(device)
    dummy_fcnet.setup(dummy_input.shape, device = device)
    dummy_output = dummy_fcnet(dummy_input)
    print(f'------\nFCNet Output Shape: {dummy_output.shape}\n------')

    dummy_cnnet= CNNet(3, 256, 10).to(device)
    dummy_input = torch.zeros((64, 3, 32, 32)).to(device)
    dummy_cnnet.setup(dummy_input.shape, device = device)
    dummy_output = dummy_cnnet(dummy_input)
    print(f'------\nCNNet Output Shape: {dummy_output.shape}\n------')