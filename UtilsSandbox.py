import tensorflow as tf
a = tf.keras.utils.to_categorical([2.3, 0.8, 0, 11, 15], num_classes=16)
print(a)