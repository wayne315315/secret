from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt

from data import get_cifar100

def metric(bank, ys):
    # bank : (num_classes, dim)
    # ys : (num_labels, 1)
    ys = np.squeeze(ys, -1)
    labels = []
    for l in range(100):
        indices = np.squeeze(np.argwhere(ys==l), -1)
        labels.append(indices)
    labels = np.asarray(labels) 
    ms = bank[labels].mean(axis=1)
    ms /= np.sqrt(np.sum(ms*ms, -1, keepdims=True)) # metroid (#num_labels, dim)
    
    inners = bank @ ms.T # (num_classes, num_labels)
    inners = inners[labels].mean(axis=1) # (num_labels, num_labels)

    pos = inners.diagonal()
    neg = (np.sum(inners, -1) - pos) / 99
    score = np.mean(pos - neg)
    return score


def bank_load(e):
    p = Path("bank", "epoch-%d.npy" % e)
    bank = np.load(p)
    return bank


def img_retrieval(i, k, bank, x_train, y_train):
    img_dir = "img"
    if Path(img_dir).exists():
        shutil.rmtree(img_dir)
    Path(img_dir).mkdir()

    v = bank[[i]]
    ids = np.argsort(v @ bank.T).squeeze()[::-1][:k]
    print("query:", i, "->", y_train[i])
    plt.imshow(x_train[i])
    plt.savefig(Path(img_dir, "query.png"))
    for j in ids:
        print(j, "->", y_train[j])
        plt.imshow(x_train[j])
        plt.savefig(Path(img_dir, "%d.png" % j))



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_cifar100()
    """
    for epoch in range(200):
        bank = bank_load(epoch)
        score = metric(bank, y_train)
        print("%d:\t%.4f" % (epoch, score))
    """
    import random
    i = random.randint(0, 49999)
    k = 10
    bank = bank_load(200)
    img_retrieval(i, k, bank, x_train, y_train)
