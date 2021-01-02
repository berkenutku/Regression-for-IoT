import numpy as np
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers, metrics
from tensorflow.keras.datasets import cifar10, cifar100
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import time, imageio
import os
import matplotlib.pyplot as plt
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print(type(x_train))
x_train = tf.image.resize(x_train, (224,224)).numpy()

#print(type(x_train))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#image = tf.cast(x_train[0], np.uint8).eval()
#print(x_train.shape)
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

'''
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(x_train.shape, y_test.shape)
'''
plt.imshow(x_train[1])

plt.show()
