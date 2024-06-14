import torch
from utils import get_accuracy

def train_model(model, train_loader, valid_loader, criterion, optimizer, device = 'cuda'):
    model.train()
    running_loss = float()
    running_accuracy = float()

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        pred = model(data)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += get_accuracy(pred, label)

    train_loss = running_loss / len(train_loader)
    train_accuracy = running_accuracy / len(train_loader)

    model.eval()
    running_loss = float()
    running_accuracy = float()

    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)

            pred = model(data)
            loss = criterion(pred, label)

            running_loss += loss.item()
            running_accuracy += get_accuracy(pred, label)

    valid_loss = running_loss / len(valid_loader)
    valid_accuracy = running_accuracy / len(valid_loader)

    return train_loss, train_accuracy, valid_loss, valid_accuracy