from keras import Model
from keras.layers import Dense, Concatenate
from keras.layers import Dot, Input
import numpy as np
from dl_ranking.pairwise_cosine import get_pairwise_model


def train_debug_model():
    batch_size = 1024
    epochs = 400
    data_samples = batch_size * epochs
    embedding_dimensions = 4
    input_dimensions = 128

    trainX = [np.random.random_sample((data_samples, input_dimensions)),
              np.random.random_sample((data_samples, input_dimensions))]
    # output is equal
    trainY = trainX[0][:, 0] * .8
    model = get_debug_model(input_dimensions, embedding_dimensions=embedding_dimensions)

    # trainX = np.random.random_sample((data_samples, embedding_dimensions))
    # trainY = trainX[:,0]*.8
    # model = get_debug_model(input_dimensions, embedding_dimensions=embedding_dimensions)

    model.fit(
        trainX, trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

def get_debug_model(input_dimensions, embedding_dimensions=50):
    input_layer_user_1 = Input(shape=(input_dimensions,))
    input_layer_user_2 = Input(shape=(input_dimensions,))
    dense_layer = Dense(embedding_dimensions, activation='relu',kernel_initializer='normal')
    embedding_layer_1 = dense_layer(input_layer_user_1)  # embedding_layer
    embedding_layer_2 = dense_layer(input_layer_user_2)  # embedding_layer

    # cosine_similarity = Dot(axes=1, normalize=True)([embedding_layer_1, embedding_layer_2])  # cosine sim

    concat = Concatenate()([embedding_layer_1,embedding_layer_2])
    cosine_similarity = Dense(1, activation='relu',kernel_initializer='normal')(concat)  # cosine sim

    model = Model([input_layer_user_1, input_layer_user_2], cosine_similarity)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_simple_linear_model():
    batch_size = 1024
    epochs = 400
    data_samples = batch_size * epochs
    embedding_dimensions = 4
    input_dimensions = 128

    trainX = np.random.random_sample((data_samples, input_dimensions))
    trainY = trainX[:, 0] * .8

    input_layer_user_1 = Input(shape=(input_dimensions,))
    dense_layer = Dense(1, activation='relu', kernel_initializer='normal')(input_layer_user_1)
    model = Model(input_layer_user_1, dense_layer)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(
        trainX, trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )



if __name__ == '__main__':
    # train_simple_linear_model()
    train_debug_model()

