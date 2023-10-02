import tensorflow as tf

def __preprocessFeatures(ds):
	return ds / 255.0

# returns preprocessed dataset
def loadData():
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  train_images = __preprocessFeatures(train_images)
  test_images = __preprocessFeatures(test_images)

  return (train_images, train_labels), (test_images, test_labels);