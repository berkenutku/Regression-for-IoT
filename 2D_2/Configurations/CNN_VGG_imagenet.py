

#THIS SEARCH SPACE WILL WORK FOR CIFAR10, CIFAR100, AND IMAGENET for VGG16-like networks
#You can directly call get_values(x) with the appropriate parameters in order to get the original performance of VGG-16
#__________________________________________________


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import time, imageio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = '../tiny-imagenet-200/'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [imageio.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        #print(train_labels_.shape)
        train_labels += train_labels_.tolist()
        #print(np.array(train_labels)[-1])

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(imageio.imread( path + 'val/images/{}'.format(img_name) ,pilmode='RGB'))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def shuffle_data(train_data, train_labels ):
    size = len(train_data)
    train_idx = np.arange(size)
    np.random.shuffle(train_idx)

    return train_data[train_idx], train_labels[train_idx]

#train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())



def normalize(X_train, X_test): #Not clear whether I should use mean and std or not
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=0)
        #std = np.std(X_train, axis=0)
        #print(mean)
        #print(std)
        #X_train = (X_train-mean)/(std+1e-7)
        #mean = 0
        X_train = (X_train-mean)/255.0
        X_test = (X_test-mean)/255.0
        return X_train, X_test


def get_values(x):
    # Model Architecture Configuration
    # Conv Layer 1
    #print(x)
    arch_param = {}
    i = 0
    while i<5:
        arch_param['f%s' %(i+1)] = x[i*4]
        arch_param['k%s' %(i+1)] = x[i*4 + 1]
        arch_param['l%s' %(i+1)] = x[i*4 + 2]
        arch_param['p%s' %(i+1)] = x[i*4 + 3]
        i = i+1
    arch_param['fc1'] = x[20]
    arch_param['fc2'] = x[21] 

    # Model Training Configuration
    num_classes = 200 #10 or 100 depending on type of CIFAR
    batch_size = 250
    no_epochs = 1
    learning_rate = 0.1
    input_shape = (64, 64, 3) #depending on CIFAR or imagenet

    # Define overall score containers

    # Load data
    '''(x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)'''
    train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())
    train_data, train_labels = shuffle_data(train_data, train_labels)

    # The data, shuffled and split between train and test sets:
    X_train = train_data
    Y_train = train_labels
    X_test = test_data
    Y_test = test_labels

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train, X_test = normalize(X_train, X_test)


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
    datagen.fit(X_train)

    # Defining the model architecture
    input = keras.Input(shape=input_shape, dtype='float32')  # Edit shape depending on whether it's cifar or imagenet

    convA1 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), padding='same', data_format="channels_last")(input)
    #convA1 = layers.BatchNormalization(3)(convA1)
    convA1 = layers.Activation('relu')(convA1)
    if arch_param['l1'] == 3:
        convA2 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), padding='same', data_format="channels_last")(convA1)
        #convA2 = layers.BatchNormalization(3)(convA2)
        convA2 = layers.Activation('relu')(convA2)
        convA3 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), padding='same', data_format="channels_last")(convA2)
        #convA3 = layers.BatchNormalization(3)(convA3)
        convA3 = layers.Activation('relu')(convA3)
    elif arch_param['l1'] == 2:
        convA2 = layers.Conv2D(arch_param['f1'], (arch_param['k1'], arch_param['k1']), (1, 1), padding='same', data_format="channels_last")(convA1)
        #convA2 = layers.BatchNormalization(3)(convA2)
        convA2 = layers.Activation('relu')(convA2)
        convA3 = convA2
    else:
        convA3 = convA1
    convA3 = layers.BatchNormalization(3)(convA3)
    if arch_param['p1'] == 1:
        poolA = layers.MaxPooling2D((2, 2), (2, 2), data_format="channels_last")(convA3)
    else:
        poolA = convA3
    
    convB1 = layers.Conv2D(arch_param['f2'], (arch_param['k2'], arch_param['k2']), (1, 1), padding='same', data_format="channels_last")(poolA)
    #convB1 = layers.BatchNormalization(3)(convB1)
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
    if arch_param['p2'] == 1:
        poolB = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convB3)
    else:
        poolB = convB3

    convC1 = layers.Conv2D(arch_param['f3'], (arch_param['k3'], arch_param['k3']), (1, 1), padding='same', data_format="channels_last")(poolB)
    #convC1 = layers.BatchNormalization(3)(convC1)
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
    if arch_param['p3'] == 1:
        poolC = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convC3)
    else:
        poolC = convC3

    convD1 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), padding='same', data_format="channels_last")(poolC)
    #convD1 = layers.BatchNormalization(3)(convD1)
    convD1 = layers.Activation('relu')(convD1)
    if arch_param['l4'] == 3:
        convD2 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), padding='same', data_format="channels_last")(convD1)
        #convD2 = layers.BatchNormalization(3)(convD2)
        convD2 = layers.Activation('relu')(convD2)
        convD3 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), padding='same', ata_format="channels_last")(convD2)
        #convD3 = layers.BatchNormalization(3)(convD3)
        convD3 = layers.Activation('relu')(convD3)
    elif arch_param['l4'] == 2:
        convD2 = layers.Conv2D(arch_param['f4'], (arch_param['k4'], arch_param['k4']), (1, 1), padding='same', data_format="channels_last")(convD1)
        #convD2 = layers.BatchNormalization(3)(convD2)
        convD2 = layers.Activation('relu')(convD2)
        convD3 = convD2
    else:
        convD3 = convD1
    convD3 = layers.BatchNormalization(3)(convD3)
    if arch_param['p4'] == 1:
        poolD = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convD3)
    else:
        poolD = convD3

    convE1 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), padding='same', data_format="channels_last")(poolD)
    #convE1 = layers.BatchNormalization(3)(convE1)
    convE1 = layers.Activation('relu')(convE1)
    if arch_param['l5'] == 3:
        convE2 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), padding='same', data_format="channels_last")(convE1)
        #convE2 = layers.BatchNormalization(3)(convE2)
        convE2 = layers.Activation('relu')(convE2)
        convE3 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), padding='same', data_format="channels_last")(convE2)
        #convE3 = layers.BatchNormalization(3)(convE3)
        convE3 = layers.Activation('relu')(convE3)
    elif arch_param['l5'] == 2:
        convE2 = layers.Conv2D(arch_param['f5'], (arch_param['k5'], arch_param['k5']), (1, 1), padding='same', data_format="channels_last")(convE1)
        #convE2 = layers.BatchNormalization(3)(convE2)
        convE2 = layers.Activation('relu')(convE2)
        convE3 = convE2
    else:
        convE3 = convE1
    convE3 = layers.BatchNormalization(3)(convE3)
    if arch_param['p5'] == 1:
        poolE = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format="channels_last")(convE3)
    else:
        poolE = convE3

    flatten = layers.Flatten()(poolE)
    if arch_param['fc1'] != 0:
        fc1 = layers.Dense(int(arch_param['fc1']))(flatten)
        #fc1 = layers.BatchNormalization(3)(fc1)
        fc1 = layers.Activation('relu')(fc1)
    else:
        fc1 = flatten
    if arch_param['fc2'] != 0:
        fc2 = layers.Dense(int(arch_param['fc2']))(fc1)
        #fc2 = layers.BatchNormalization(3)(fc2)
        fc2 = layers.Activation('relu')(fc2)
    else:
        fc2 = fc1

    output = layers.Dense(num_classes)(fc2)
    output = layers.Activation('softmax')(output)


    model = keras.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=no_epochs, steps_per_epoch = X_train.shape[0] // batch_size, verbose=1, validation_data=(X_test, Y_test))
    #If data augmentation is needed
    #history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch = X_train.shape[0] // batch_size, epochs=no_epochs, verbose=1, validation_data=(X_test, Y_test))

    #If you want to save the weights
    #model.save_weights('vgg_16.h5')


    # Generate generalization metrics
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)   #Verbosity
    print(test_acc)

    return test_acc