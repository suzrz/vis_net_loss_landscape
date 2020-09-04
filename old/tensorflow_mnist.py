import tensorflow_datasets as tfds
import tensorflow as tf


# load data
mnist_bldr = tfds.builder("mnist")
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)  # shuffle disabled to split to be able to create validation set

train_orig = datasets["train"]
test_orig = datasets["test"]

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 2  # 20

train = train_orig.map(lambda item: (tf.cast(item["image"], tf.float32) / 255.0,
                                     tf.cast(item["label"], tf.int32)))
test = test_orig.map(lambda item: (tf.cast(item["image"], tf.float32 / 255.0,
                                           tf.cast(item["label"], tf.int32))))
tf.random.set_seed(1)
train = train.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
valid = train.take(10000).batch(BATCH_SIZE)  # validation set
train = train.skip(10000).batch(BATCH_SIZE)
