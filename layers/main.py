import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers

# (batch, steps, features)
# (sentences, words, embeddings)


# 2 sentences, 3 words each, 4 emb dim per word
tensor = tf.constant([
    [ #sentence 1
    	[3, 3, 3, 3], # word1
     	[5,5,5,5], # word2
        [6,6,6,6] # word3
    ],
    [ #sentence 2
        [76,76,76,76], # word1
          [34,34,34,34],
            [77,77,77,77]
    ]
 ])


tensor = tf.constant([[4, 4, 4, 4], [62, 62, 62, 62]])

print('original tensor', tensor)
print('original tensor shape', tensor)
print('==============')
# df = tf.data.Dataset.from_tensor_slices(tensor);
# df.batch(2)

#print(df)

globalAveragePooling1DLayer = layers.GlobalAveragePooling1D()

pooled = globalAveragePooling1DLayer(tensor)
print('pooled tensor', pooled)
