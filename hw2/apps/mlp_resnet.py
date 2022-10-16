import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    module = nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    return module
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_all = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = (total - hit) / total
    return acc, loss_all / (idx + 1)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tr_im_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    tr_lb_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    te_im_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    te_lb_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    tr_dataset = ndl.data.MNISTDataset(tr_im_path, tr_lb_path)
    te_dataset = ndl.data.MNISTDataset(te_im_path, te_lb_path)

    tr_dataloader = ndl.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    te_dataloader = ndl.data.DataLoader(te_dataset)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        tr_acc, tr_loss = epoch(tr_dataloader, model, opt)
    te_acc, te_loss = epoch(te_dataloader, model)
    return (tr_acc, tr_loss, te_acc, te_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
