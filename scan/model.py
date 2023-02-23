import tensorflow as tf

from loss import id_loss
from resnet import resnet


class ID_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k = 100
        self.num_classes = None
        self.num_sampled = None
        self.tau = None
        self.lmda = None
        self.dim = None
        self.bank = None
        self.loss_tracker = tf.keras.metrics.Mean("id_loss")
        self.acc_tracker = tf.keras.metrics.Accuracy(name="acc")
        self.acc_topk_tracker = tf.keras.metrics.TopKCategoricalAccuracy(k=k, name="acc_top_%d" % k)
        self.lr_tracker = tf.keras.metrics.Mean(name="lr")

    def init_ID(self, num_classes, num_sampled, tau, lmda):
        dim = self.output_shape[-1]
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.tau = tau
        self.lmda = lmda
        self.dim = dim
        self.bank = self.bank_init(num_classes, dim)
    
    def bank_init(self, num_classes, dim):
        bank = tf.random.normal([num_classes, dim])
        norm = tf.sqrt(tf.reduce_sum(bank*bank, -1, keepdims=True))
        bank /= norm
        bank = tf.Variable(bank, trainable=False)
        return bank

    @tf.function(jit_compile=True)
    def bank_update(self, xs, ys):
        # xs : (#batch, dim)
        # ys : (#batch, 1)
        self.bank.scatter_nd_update(ys, xs)
    
    @tf.function(jit_compile=False)
    def train_step(self, data):
        xs, ys, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        xs, ys = data
        with tf.GradientTape() as tape:
            xs = self(xs, training=True)
            loss = id_loss(xs, ys, self.bank, self.num_classes, num_sampled=self.num_sampled, tau=self.tau, lmda=self.lmda)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(tf.reduce_sum(loss), trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update bank
        self.bank_update(xs, ys)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        return {self.loss_tracker.name: self.loss_tracker.result()}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        xs, ys, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        xs = self(xs, training=False) # (batch, dim)
        ys = tf.squeeze(ys, axis=-1)
        logits = tf.matmul(xs, self.bank, transpose_b=True) / self.tau # (batch, num_classes)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits)
        ys_pred = tf.argmax(logits, axis=-1)
        ys_onehot = tf.one_hot(ys, logits.shape[1])

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(ys, ys_pred)
        self.acc_topk_tracker.update_state(ys_onehot, logits)
        self.lr_tracker.update_state(self.optimizer.lr)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=True)
    def predict_step(self, data):
        xs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        xs = self(xs, training=False) # (batch, dim)
        logits = tf.matmul(xs, self.bank, transpose_b=True) / self.tau # (batch, num_classes)
        pbs = tf.nn.softmax(logits)
        return pbs

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker, self.acc_topk_tracker, self.lr_tracker]

def id_model(base_model, num_classes, num_sampled, tau, lmda, dim):
    # representation learning w/ instance discrimination
    i = o = tf.keras.Input(shape=(None, None, 3))
    o = base_model(o)
    o = tf.keras.layers.Dense(dim)(o)
    o = tf.keras.layers.UnitNormalization()(o)
    model = ID_Model(i, o, name="id_model")
    model.init_ID(num_classes, num_sampled, tau, lmda)
    return model

def efficientb0(input_shape):
    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg"
    )
    return model

def efficientb7(input_shape):
    model = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg"
    )
    return model

def efficientv2l(input_shape):
    model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg",
        include_preprocessing=False
    )
    return model

def resnet18(input_shape):
    i = o = tf.keras.Input(shape=input_shape)
    o = resnet(o, [2, 2, 2, 2])
    model = tf.keras.Model(i, o, name="resnet_18")
    return model

def resnet34(input_shape):
    i = o = tf.keras.Input(shape=input_shape)
    o = resnet(o, [3, 4, 6, 3])
    model = tf.keras.Model(i, o, name="resnet_34")
    return model

def resnet50(input_shape):
    model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg"
    )
    return model

def resnet152(input_shape):
    model = tf.keras.applications.resnet_v2.ResNet152V2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg"
    )
    return model

if __name__ == "__main__":
    input_shape = (32,32,3)
    num_classes = 50000
    num_sampled = 10
    tau = 0.07
    lmda = 50
    dim = 128

    # efficient b0
    base_model = efficientb0(input_shape)
    model = id_model(base_model, num_classes, num_sampled, tau, lmda, dim)
    model.summary()

    # resnet 18
    base_model = resnet18(input_shape)
    model = id_model(base_model, num_classes, num_sampled, tau, lmda, dim)
    model.summary()
