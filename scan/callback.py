from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf


class BankCallback(tf.keras.callbacks.Callback):
    def __init__(self, bank_dir):
        super().__init__()
        self.bank_dir = bank_dir
        if Path(bank_dir).exists():
            shutil.rmtree(bank_dir)
        Path(bank_dir).mkdir()

    def on_epoch_end(self, epoch, logs=None):
        bank = self.model.bank.read_value().numpy()
        name = "epoch-%d.npy" % epoch
        p = Path(self.bank_dir, name)
        np.save(p, bank) 
