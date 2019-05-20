from keras import Model
from keras.layers import Dense
from keras.layers import Dot, Input

from data_processing.one_hot_data_provider import OneHotDataProvider


def get_pairwise_model(input_dimensions, embedding_dimensions=50):
    input_layer_user_1 = Input(shape=(input_dimensions,))
    input_layer_user_2 = Input(shape=(input_dimensions,))
    embedding_layer_1 = Dense(50, activation='sigmoid')(input_layer_user_1)  # embedding_layer
    embedding_layer_2 = Dense(50, activation='sigmoid')(input_layer_user_2)  # embedding_layer
    cosine_similarity = Dot(axes=0, normalize=True)([embedding_layer_1, embedding_layer_2])  # cosine sim
    model = Model([input_layer_user_1, input_layer_user_2], cosine_similarity)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    one_hot_data = OneHotDataProvider()
    trainX, trainY = one_hot_data.get_trianing_data(10)
    validationX, validationY = one_hot_data.get_trianing_data(10)
    input_dimensions = len(trainX[0][0])
    model = get_pairwise_model(input_dimensions=input_dimensions)
    model.fit(
        trainX, trainY,
        validation_data=(validationX, validationY),
        epochs=32, batch_size=8)