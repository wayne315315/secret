from pathlib import Path

from tqdm import tqdm
import tensorflow as tf


def get_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    return ((x_train, y_train), (x_test, y_test))


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import matplotlib.pyplot as plt

    (x_train, y_train), (x_test, y_test) = get_cifar100()
    data_dir = "data"
    if not Path(data_dir).exists():
        Path(data_dir).mkdir()
    
    def save_image(x):
        image, label = x
        plt.imshow(image)
        p = Path(data_dir, "%s.png" % label)
        plt.savefig(p)
    
    with ProcessPoolExecutor() as executor:
        for _ in tqdm(executor.map(save_image, zip(x_train, y_train))):
            pass
