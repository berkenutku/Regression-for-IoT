

#THIS SEARCH SPACE WILL WORK FOR IMAGENET, CIFAR-10 and CIFAR-100 for alexnet variants. will require RESIZE to 224*224*3
#__________________________________________________


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, callbacks
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def normalize(X_train, X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        #print(mean)
        #print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test


def get_values(x):
    # Model Architecture Configuration
    # Conv Layer 1
    #print(x)
    arch_param = {}
    i = 0
    arch_param['f1'] = x[0]
    arch_param['k1'] = x[1]
    arch_param['p1'] = x[2]
    while i<2:
        arch_param['f%s' %(i+2)] = x[i*4 + 3]
        arch_param['k%s' %(i+2)] = x[i*4 + 4]
        arch_param['l%s' %(i+2)] = x[i*4 + 5]
        arch_param['p%s' %(i+2)] = x[i*4 + 6]
        i = i+1
    arch_param['fc1'] = x[11]
    arch_param['fc2'] = x[12] 

    # Model Training Configuration
    num_classes = 100 #10 or 100 depending on type of CIFAR
    batch_size = 128
    no_epochs = 20
    learning_rate = 0.1
    input_shape = (32, 32, 3) #depending on CIFAR or imagenet
    validation_split = 0.1

    file = open(f"CNN.txt","w")

    # Define overall score containers

    # Load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(tf.shape(x_train), tf.shape(x_test))

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Defining the model architecture
    input = keras.Input(shape=input_shape, dtype='float32')  # Edit shape depending on whether it's cifar or imagenet

    convA = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (4, 4), padding='valid', data_format="channels_last")(input)
    convA = layers.Activation('relu')(convA)
    convA = layers.BatchNormalization(3)(convA)
    if arch_param['p1'] == 0:
        poolA = convA
    elif arch_param['p1'] == 1:
        poolA = layers.MaxPooling2D((2, 2), (2, 2), data_format="channels_last")(convA)
    else:
        poolA = layers.MaxPooling2D((3, 3), (2, 2), data_format="channels_last")(convA)
    
    convB1 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), padding='same', data_format="channels_last")(poolA)
    convB1 = layers.Activation('relu')(convB1)
    if arch_param['l2'] == 3:
        convB2 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), padding='same', data_format="channels_last")(convB1)
        #convB2 = layers.BatchNormalization(3)(convB2)
        convB2 = layers.Activation('relu')(convB2)
        convB3 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), padding='same', data_format="channels_last")(convB2)
        #convB3 = layers.BatchNormalization(3)(convB3)
        convB3 = layers.Activation('relu')(convB3)
    elif arch_param['l2'] == 2:
        convB2 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), padding='same', data_format="channels_last")(convB1)
        #convB2 = layers.BatchNormalization(3)(convB2)
        convB2 = layers.Activation('relu')(convB2)
        convB3 = convB2
    else:
        convB3 = convB1
    convB3 = layers.BatchNormalization(3)(convB3)
    if arch_param['p2'] == 0:
        poolB = convB3
    elif arch_param['p2'] == 1:
        poolB = layers.MaxPooling2D((2, 2), (2, 2), data_format="channels_last")(convB3)
    else:
        poolB = layers.MaxPooling2D((3, 3), (2, 2), data_format="channels_last")(convB3)

    convC1 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), padding='same', data_format="channels_last")(poolB)
    convC1 = layers.Activation('relu')(convC1)
    if arch_param['l3'] == 3:
        convC2 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), padding='same', data_format="channels_last")(convC1)
        #convC2 = layers.BatchNormalization(3)(convC2)
        convC2 = layers.Activation('relu')(convC2)
        convC3 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), padding='same', data_format="channels_last")(convC2)
        #convC3 = layers.BatchNormalization(3)(convC3)
        convC3 = layers.Activation('relu')(convC3)
    elif arch_param['l3'] == 2:
        convC2 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), padding='same', data_format="channels_last")(convC1)
        #convC2 = layers.BatchNormalization(3)(convC2)
        convC2 = layers.Activation('relu')(convC2)
        convC3 = convC2
    else:
        convC3 = convC1
    convC3 = layers.BatchNormalization(3)(convC3)
    if arch_param['p3'] == 0:
        poolC = convC3
    elif arch_param['p3'] == 1:
        poolC = layers.MaxPooling2D((2, 2), (2, 2), data_format="channels_last")(convC3)
    else:
        poolC = layers.MaxPooling2D((3, 3), (2, 2), data_format="channels_last")(convC3)

    flatten = layers.Flatten()(poolC)
    if arch_param['fc1'] != 0:
        fc1 = layers.Dense(int(arch_param['fc1']))(flatten)
        #fc1 = layers.BatchNormalization(3)(fc1)
        fc1 = layers.Activation('relu')(fc1)
        #should I add dropout here ?
    else:
        fc1 = flatten
    if arch_param['fc2'] != 0:
        fc2 = layers.Dense(int(arch_param['fc2']))(fc1)
        #fc2 = layers.BatchNormalization(3)(fc2)
        fc2 = layers.Activation('relu')(fc2)
        #should I add dropout here ?
    else:
        fc2 = fc1

    output = layers.Dense(num_classes)(fc2)
    output = layers.Activation('softmax')(output)


    model = keras.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=no_epochs, verbose=2, validation_data=(x_test, y_test))
    #If data augmentation is needed
    #history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = x_train.shape[0] // batch_size, epochs=no_epochs, verbose=2, validation_data=(x_test, y_test))

    #If you want to save the weights
    #model.save_weights('vgg_16.h5')


    # Generate generalization metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)   #Verbosity
    print(test_acc)

    return test_acc

    #Need top 5 accuracy, test this code with arbitratry architecture