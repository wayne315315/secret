from pathlib import Path
import shutil

import numpy as np


def foo(i, n):
    bank_dir = "bank"
    tsv_dir = "tsv"
    p1 = Path(bank_dir, "epoch-%d.npy" % i)
    p2 = Path(tsv_dir, "epoch-%d.tsv" % i)
    p3 = Path(tsv_dir, "epoch-%d-%d.tsv" % (i, n))

    x = np.load(p1)
    y = x[:n]
    np.savetxt(p2, x, delimiter="\t")
    np.savetxt(p3, y, delimiter="\t")


if __name__ == "__main__":
    epoch = 170
    n = 10000
    foo(epoch, n)
