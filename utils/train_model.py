import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

class Mean_metric():
    def __init__(self, value=0, iteration=0):
        self.value = value
        self.iteration = iteration
    def update(self, value, num_iter=1):
        self.value += value
        self.iteration += num_iter
    def get_value(self):
        return self.value / self.iteration
    def reset(self):
        self.value = 0
        self.iteration = 0


def train(model, loss_fn,optimizer, scheduler, train_loader, val_loader, model_name, device,epoch=1, print_every=1):
    model = model.to(device)
    train_loss = Mean_metric()
    train_acc = Mean_metric()
    val_loss = Mean_metric()
    val_acc = Mean_metric()
    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    best_acc = 0
    batch_size = train_loader.batch_size
    num_train = len(train_loader.dataset)
    for e in range(epoch):
        model.train()
        for i, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            score = model(x)
            loss = loss_fn(score, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            with torch.no_grad():
                train_acc.update((y == score.argmax(dim=1)).sum().item(), len(y))
            if i % print_every == print_every - 1 or i == len(train_loader) - 1:
                print(f"Epoch:{e}\tIteration:{i}\tTrain Loss:{train_loss.get_value()}\tTrain Accuracy:{train_acc.get_value()}")
        train_loss_hist.append(train_loss.get_value())
        train_acc_hist.append(train_acc.get_value())
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                score = model(x)
                loss = loss_fn(score, y)
                val_loss.update(loss.item())
                val_acc.update((y == score.argmax(dim=1)).sum().item(), len(y))
        val_loss_hist.append(val_loss.get_value())
        val_acc_hist.append(val_acc.get_value())
        if val_acc_hist[-1] > best_acc:
            best_acc = val_acc_hist[-1]
            torch.save(model.state_dict(), model_name)
        print(f"Epoch:{e}\tVal Loss:{val_loss.get_value()}\tVal Accuracy:{val_acc.get_value()}")
        train_loss.reset()
        train_acc.reset()
        val_loss.reset()
        val_acc.reset()
        scheduler.step()

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist