

#THIS SEARCH SPACE WILL WORK FOR CIFAR10, CIFAR100, AND IMAGENET for VGG16-like networks
#You can directly call get_values(x) with the appropriate parameters in order to get the original performance of VGG-16
#__________________________________________________
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import time, imageio
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_values(x):
    # Model Architecture Configuration
    # Conv Layer 1
    #print(x)
    arch_param = {}
    i = 0
    while i<5:
        arch_param['f%s' %(i+1)] = int(x[i*3])
        arch_param['k%s' %(i+1)] = int(x[i*3 + 1])
        arch_param['l%s' %(i+1)] = int(x[i*3 + 2])
        i = i+1
    arch_param['p'] = int(x[15])
    arch_param['fc1'] = int(x[16])
    arch_param['fc2'] = int(x[17])
    #arch_param['fc3'] = int(x[18]) 

    # Model Training Configuration
    num_classes = 10 #10 or 100 depending on type of CIFAR
    batch_size = 32
    no_epochs = 300
    learning_rate = 0.01
    validation_split = 0.1
    input_shape = (32, 32, 3) #RETURN TO 32,32,3

    # Define overall score containers

    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Download and prepare the CIFAR10 dataset
    
    x_train, y_train = shuffle(x_train, y_train, random_state=0)


    #x_train = x_train.astype('float32')
    #x_test = x_test astype('float32')

    #print(x_train.shape)
    datagen = ImageDataGenerator(
          featurewise_center=False,           # set input mean to 0 over the dataset
          samplewise_center=False,            # set each sample mean to 0
          featurewise_std_normalization=False,# divide inputs by std of the dataset
          samplewise_std_normalization=False, # divide each input by its std
          zca_whitening=False,                # apply ZCA whitening
          rotation_range=0,                   # randomly rotate images in the range (degrees, 0 to 180)
          width_shift_range=0.1,              # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,             # randomly shift images vertically (fraction of total height)
          horizontal_flip=True,               # randomly flip images
          vertical_flip=False )               # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    tf.keras.backend.clear_session()

    # Defining the model architecture
    input = keras.Input(shape=input_shape, dtype='float32')  # Edit shape depending on whether it's cifar or imagenet

    convA1 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(input)
    #convA1 = layers.BatchNormalization()(convA1)
    convA1 = layers.Activation('relu')(convA1)
    convA1 = layers.BatchNormalization()(convA1)
    if arch_param['l1'] == 3:
        convA2 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convA1)
        #convA2 = layers.BatchNormalization(3)(convA2)
        convA2 = layers.Activation('relu')(convA2)
        convA2 = layers.BatchNormalization()(convA2)
        convA3 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convA2)
        #convA3 = layers.BatchNormalization(3)(convA3)
        convA3 = layers.Activation('relu')(convA3)
        convA3 = layers.BatchNormalization()(convA3)
    elif arch_param['l1'] == 2:
        convA2 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convA1)
        #convA2 = layers.BatchNormalization(3)(convA2)
        convA2 = layers.Activation('relu')(convA2)
        convA3 = convA2
        convA3 = layers.BatchNormalization()(convA3)
    else:
        convA3 = convA1
    if arch_param['p'] != 1:
        poolA = layers.MaxPooling2D((2, 2), (2, 2), data_format="channels_last")(convA3)
    else:
        poolA = convA3
    
    convB1 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(poolA)
    #convB1 = layers.BatchNormalization(3)(convB1)
    convB1 = layers.Activation('relu')(convB1)
    convB1 = layers.BatchNormalization()(convB1)
    if arch_param['l2'] == 3:
        convB2 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convB1)
        #convB2 = layers.BatchNormalization(3)(convB2)
        convB2 = layers.Activation('relu')(convB2)
        convB2 = layers.BatchNormalization()(convB2)
        convB3 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convB2)
        #convB3 = layers.BatchNormalization(3)(convB3)
        convB3 = layers.Activation('relu')(convB3)
        convB3 = layers.BatchNormalization()(convB3)
    elif arch_param['l2'] == 2:
        convB2 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convB1)
        #convB2 = layers.BatchNormalization(3)(convB2)
        convB2 = layers.Activation('relu')(convB2)
        convB3 = convB2
        convB3 = layers.BatchNormalization()(convB3)
    else:
        convB3 = convB1
    if arch_param['p'] != 2:
        poolB = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convB3)
    else:
        poolB = convB3

    convC1 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(poolB)
    #convC1 = layers.BatchNormalization(3)(convC1)
    convC1 = layers.Activation('relu')(convC1)
    convC1 = layers.BatchNormalization()(convC1)
    if arch_param['l3'] == 3:
        convC2 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convC1)
        #convC2 = layers.BatchNormalization(3)(convC2)
        convC2 = layers.Activation('relu')(convC2)
        convC2 = layers.BatchNormalization()(convC2)
        convC3 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convC2)
        #convC3 = layers.BatchNormalization(3)(convC3)
        convC3 = layers.Activation('relu')(convC3)
        convC3 = layers.BatchNormalization()(convC3)
    elif arch_param['l3'] == 2:
        convC2 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convC1)
        #convC2 = layers.BatchNormalization(3)(convC2)
        convC2 = layers.Activation('relu')(convC2)
        convC3 = convC2
        convC3 = layers.BatchNormalization()(convC3)
    else:
        convC3 = convC1
    if arch_param['p'] != 3:
        poolC = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convC3)
    else:
        poolC = convC3

    convD1 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(poolC)
    #convD1 = layers.BatchNormalization(3)(convD1)
    convD1 = layers.Activation('relu')(convD1)
    convD1 = layers.BatchNormalization()(convD1)
    if arch_param['l4'] == 3:
        convD2 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convD1)
        #convD2 = layers.BatchNormalization(3)(convD2)
        convD2 = layers.Activation('relu')(convD2)
        convD2 = layers.BatchNormalization()(convD2)
        convD3 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), kernel_initializer='he_uniform', padding='same', ata_format="channels_last")(convD2)
        #convD3 = layers.BatchNormalization(3)(convD3)
        convD3 = layers.Activation('relu')(convD3)
        convD3 = layers.BatchNormalization()(convD3)
    elif arch_param['l4'] == 2:
        convD2 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convD1)
        #convD2 = layers.BatchNormalization(3)(convD2)
        convD2 = layers.Activation('relu')(convD2)
        convD2 = layers.BatchNormalization()(convD2)
        convD3 = convD2
    else:
        convD3 = convD1
    if arch_param['p'] != 4:
        poolD = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convD3)
    else:
        poolD = convD3

    convE1 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(poolD)
    #convE1 = layers.BatchNormalization(3)(convE1)
    convE1 = layers.Activation('relu')(convE1)
    convE1 = layers.BatchNormalization()(convE1)
    if arch_param['l5'] == 3:
        convE2 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convE1)
        #convE2 = layers.BatchNormalization(3)(convE2)
        convE2 = layers.Activation('relu')(convE2)
        convE2 = layers.BatchNormalization()(convE2)
        convE3 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convE2)
        #convE3 = layers.BatchNormalization(3)(convE3)
        convE3 = layers.Activation('relu')(convE3)
        convE3 = layers.BatchNormalization()(convE3)
    elif arch_param['l5'] == 2:
        convE2 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), kernel_initializer='he_uniform', padding='same', data_format="channels_last")(convE1)
        #convE2 = layers.BatchNormalization(3)(convE2)
        convE2 = layers.Activation('relu')(convE2)
        convE3 = convE2
        convE3 = layers.BatchNormalization()(convE3)
    else:
        convE3 = convE1
    if arch_param['p'] != 5:
        poolE = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convE3)
    else:
        poolE = convE3

    flatten = layers.Flatten()(poolE)
    #flatten = layers.Flatten()(poolC)
    if arch_param['fc1'] != 0:
        fc1 = layers.Dense(int(arch_param['fc1']))(flatten)
        #fc1 = layers.BatchNormalization(3)(fc1)
        fc1 = layers.Activation('relu')(fc1)
    else:
        fc1 = flatten
    if arch_param['fc2'] != 0:
        fc2 = layers.Dense(int(arch_param['fc2']), kernel_initializer='he_uniform')(fc1)
        #fc2 = layers.BatchNormalization(3)(fc2)
        fc2 = layers.Activation('relu')(fc2)
    else:
        fc2 = fc1

    checkpoint = ModelCheckpoint("model_weights.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=100)
    '''    fc3 = layers.Dense(int(arch_param['fc3']), kernel_initializer='he_uniform')(fc2)
        #fc2 = layers.BatchNormalization(3)(fc2)
        fc3 = layers.Activation('relu')(fc3)
    else:
        fc3 = fc2'''

    output = layers.Dense(num_classes, kernel_initializer='he_uniform')(fc2)    #Change to fc3 if you're using another dense layer
    output = layers.Activation('softmax')(output)


    model = keras.Model(inputs=input, outputs=output)

    model.summary()


    # compile model
    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    #history = model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size, steps_per_epoch = x_train.shape[0] // batch_size, verbose=1, validation_split=validation_split)
    #If data augmentation is needed
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = x_train.shape[0] // batch_size, epochs=no_epochs, verbose=2, validation_data=(x_test, y_test), callbacks=[checkpoint])

    #If you want to save the weights
    #model.save_weights('vgg_16.h5')

    # Generate generalization metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)   #Verbosity
    print(test_acc*100)

    return test_acc*100
