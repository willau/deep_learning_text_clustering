# -*- coding: utf-8 -*-


from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, GlobalMaxPooling1D


def create_reference_model(input_dim, output_dim, embedding_layer):

    # Input
    sequence_input = Input(shape=(input_dim, ), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # 1st Layer
    x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)

    # Output
    x = Dropout(0.5)(x)
    predictions = Dense(output_dim, activation='sigmoid')(x)
    model = Model(sequence_input, predictions)

    # Fine-tune embeddings or not
    model.layers[1].trainable = False

    # Loss and Optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['mae'])

    return model


def behead(model):
    return Model(
        model.layers[0].input,
        model.layers[-2].output
    )
