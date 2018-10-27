import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models import load_model
from keras.losses import mean_squared_error

def create_baseline_model(activation = 'relu', input_shape=(64, 64)):
    model = Sequential()
    model.add(Conv2D(100, (11,11), padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_larger(activation = 'relu', input_shape=(64, 64)):
    """
    Larger (more filters) convnet model : one convolution, one average pooling and one fully connected layer:
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc. 
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(200, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 16200]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_model_deeper(activation = 'relu', input_shape=(64, 64)):
    """
    Deeper convnet model : two convolutions, two average pooling and one fully connected layer:
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(64, (11,11), activation=activation, padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(128, (10, 10), activation=activation, padding='valid', strides=(1, 1)))
    model.add(AveragePooling2D((2,2)))
    model.add(Reshape([-1, 128*9*9]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def create_maxpooling_model(activation = 'relu', input_shape = (64,64)):
    """
    Simple convnet model with max pooling: one convolution, one max pooling and one fully connected layer
    :param activation: None if nothing passed, e.g : ReLu, tanh, etc.
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(100, (11,11), activation='relu', padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(MaxPooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

def print_model(model):
    print('Size for each layer :\nLayer, Input Size, Output Size')
    for p in model.layers:
        print(p.name.title(), p.input_shape, p.output_shape)


def run_cnn(data, train = False):
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    if train:
        model = create_maxpooling_model()
        print_model(model)
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        h = training(model, X_train, Y_train, batch_size=16, epochs= 10, data_augm=False)
        metrics = 'loss'
        plt.plot(range(len(h.history[metric])), h.history[metric])
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.title("Learning curve")
        model.save('cnn_model_saved.h5')

        y_pred = model.predict(X_test, batch_size = 16)
    else:
        try:
            model = load_model('cnn_model_saved.h5')
        except IOError as e:
            print "I/O Error ({0}): {1}".format(e.errno, e.strerror)
        y_pred = model.predict(X_test, batch_size = 16)
    del model
    return y_pred

def run(X, Y, model, X_to_pred=None, history=False, verbose=0, activation=None, epochs=20, data_augm=False):
    if model == 'simple':
        m = create_baseline_model(activation = activation)
    elif model == 'larger':
        m = create_model_larger(activation=activation)
    elif model == 'deeper':
        m = create_model_deeper(activation=activation)
    elif model == 'maxpooling':
        m = create_model_maxpooling(activation=activation)

    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    if verbose > 0:
        print('Size for each layer :\nLayer, Input Size, Output Size')
        for p in m.layers:
            print(p.name.title(), p.input_shape, p.output_shape)
    h = training(m, X, Y, batch_size=16, epochs=epochs, data_augm=data_augm)

    if not X_to_pred:
        X_to_pred = X
    y_pred = m.predict(X_to_pred, batch_size=16)
    
    if history:
        return h, m
    else:
        return m


def training(model, X, Y, batch_size=16, epochs= 10, data_augm=False):
    """
    Training CNN with the possibility to use data augmentation
    :param m: Keras model
    :param X: training pictures
    :param Y: training binary ROI mask
    :return: history
    """
    if data_augm:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) 
        datagen.fit(X)
        history = model.fit_generator(datagen.flow(X, Y,
                                    batch_size=batch_size),
                                    steps_per_epoch=X.shape[0] // batch_size,
                                    epochs=epochs)         
    else:
        history = model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    return history

