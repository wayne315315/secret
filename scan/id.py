"""Instance Discrimination"""
from pathlib import Path

import tensorflow as tf

from model import id_model, efficientb0, efficientb7, efficientv2l, resnet18, resnet34, resnet50, resnet152
from callback import BankCallback
from data import get_cifar100
from augment import augment_fn, lookup_closure

# set set_jit = False
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)

# prepare data
(x_train, y_train), (x_test, y_test) = get_cifar100()
z_train = tf.range(y_train.shape[0], dtype=tf.int64)
z_train = tf.expand_dims(z_train, -1)


# parameter setting
num_classes, *input_shape = x_train.shape
num_sampled = 9999
tau = 0.07
lmda = 50
dim = 128
bank_dir = "bank"
model_dir = "model"
epochs = 1
batch_size = 1024


# callbacks
bankCallback = BankCallback(bank_dir)
lrCallback = tf.keras.callbacks.ReduceLROnPlateau(monitor='id_loss', factor=0.1**0.5, patience=5, verbose=1, mode='min', min_delta=0.0001, cooldown=10, min_lr=1e-10)
backupCallback = tf.keras.callbacks.BackupAndRestore("backup", save_freq='epoch', delete_checkpoint=True, save_before_preemption=False)

callbacks = [bankCallback, lrCallback, backupCallback]

# id_model
base_model = resnet18(input_shape)
#base_model = efficientb7(input_shape)
#base_model = efficientv2l(input_shape)
#base_model = resnet50(input_shape)
model = id_model(base_model, num_classes, num_sampled, tau, lmda, dim)
# compile
model.compile(optimizer="adam")

# dataset
lookup_fn = lookup_closure(x_train)
ds_train = tf.data.Dataset.from_tensor_slices(z_train).shuffle(num_classes).map(lookup_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(batch_size).prefetch(2)
ds_test = tf.data.Dataset.from_tensor_slices(z_train).map(lookup_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(batch_size).prefetch(2)

# start training
model.fit(ds_train, epochs=epochs, callbacks=callbacks, validation_data=ds_test)
model.save(model_dir)
print(model.bank)
#tf.saved_model.save(model, model_dir)
