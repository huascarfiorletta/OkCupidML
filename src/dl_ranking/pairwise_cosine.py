import os

from keras import Model
from keras.layers import Dense
from keras.layers import Dot, Input
import numpy as np
from data_processing.one_hot_data_provider import OneHotDataProvider


#https://keras.io/getting-started/functional-api-guide/
#https://machinelearningmastery.com/keras-functional-api-deep-learning/
def get_pairwise_model(input_dimensions, embedding_dimensions=50):
    input_layer_user_1 = Input(shape=(input_dimensions,))
    input_layer_user_2 = Input(shape=(input_dimensions,))
    dense_layer = Dense(embedding_dimensions, activation='relu')
    embedding_layer_1 = dense_layer(input_layer_user_1)  # embedding_layer
    embedding_layer_2 = dense_layer(input_layer_user_2)  # embedding_layer
    cosine_similarity = Dot(axes=1, normalize=True)([embedding_layer_1, embedding_layer_2])  # cosine sim
    model = Model([input_layer_user_1, input_layer_user_2], cosine_similarity)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == '__main__':
    filename = 'pairwise_cosine.cache.npz'
    batch_size = 4096
    one_hot_data = OneHotDataProvider()
    if not os.path.isfile(filename):
        trainX, trainY = one_hot_data.get_training_data(100000)
        np.savez(filename, trainX=trainX, trainY=trainY)
    else:
        data = np.load(filename)
        trainX = data['trainX']
        trainX = [trainX[0], trainX[1]]
        trainY = data['trainY']

    input_dimensions = len(trainX[0][0])
    model = get_pairwise_model(input_dimensions=input_dimensions)
    # model.summary()
    print(trainX[0].shape, trainY.shape)
    model.fit(
        trainX, trainY,
        epochs=5000,
        batch_size=batch_size,
        validation_split=0.1
    )