import tensorflow as tf

from data import get_cifar100


# load data
(x_train, y_train), (x_test, y_test) = get_cifar100()

# load model
model_dir = "model"
model = tf.keras.models.load_model(model_dir)

# predict 
z_train_pred = model.predict(x_train, batch_size=1024, verbose=1)
print(z_train_pred.shape)
print(model.bank)
