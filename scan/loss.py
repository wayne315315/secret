import tensorflow as tf


@tf.function(jit_compile=False)
def nce_loss(xs, ys, bank, num_classes, num_true=1, num_sampled=4092, tau=0.07):
  """Negative contrastive estimation"""
  # xs : (#batch, dim)
  # ys : (#batch, 1)
  # bank : (#classes, dim)
  # num_class
  # num_true
  # num_sampled
  # tau

  bank = tf.identity(bank)
  bank /= tau

  sampled_values = tf.random.uniform_candidate_sampler(
      ys,
      num_true,
      num_sampled,
      unique=False,
      range_max=num_classes
  )

  biases = tf.zeros([num_classes], tf.float32)

  loss = tf.nn.nce_loss(
    bank,
    biases,
    ys,
    xs,
    num_sampled,
    num_classes,
    num_true=num_true,
    sampled_values=sampled_values,
    remove_accidental_hits=False,
    name='nce_loss'
  )

  return loss

@tf.function(jit_compile=True)
def pr_loss(xs, ys, bank, lmda=50):
  """proximal regularization"""
  # xs : (#batch, dim)
  # ys : (#batch, 1)
  # bank : (#classes, dim)
  ys = tf.squeeze(ys, 1)
  xs_pre = tf.gather(bank, ys)
  xs_delta = xs - xs_pre
  loss = lmda * tf.reduce_sum(xs_delta * xs_delta, -1)

  return loss

@tf.function(jit_compile=False)
def id_loss(xs, ys, bank, num_classes, num_true=1, num_sampled=4092, tau=0.07, lmda=50):
  nce = nce_loss(xs, ys, bank, num_classes, num_true=num_true, num_sampled=num_sampled, tau=tau)
  pr = pr_loss(xs, ys, bank, lmda=lmda)
  loss = nce + pr
  loss /= (num_sampled + 1)

  return loss


if __name__ == "__main__":
  bank = tf.constant([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], dtype=tf.float32)
  xs = tf.constant([[1,1,1,1],[1,1,1,1],[3,3,3,3]], dtype=tf.float32)
  ys = tf.expand_dims(tf.constant([0,0,3], dtype=tf.int64), -1)
  num_classes = 4
  num_sampled = 10

  nce_loss = nce_loss(xs, ys, bank, num_classes, num_true=1, num_sampled=num_sampled, tau=0.07)
  pr_loss = pr_loss(xs, ys, bank)
  id_loss = id_loss(xs, ys, bank, num_classes, num_true=1, num_sampled=4092, tau=0.07, lmda=50)
  print(nce_loss, pr_loss)