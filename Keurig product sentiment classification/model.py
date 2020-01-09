import keras 
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, Activation
from keras.models import Model

def DNN_model(input_size, embedding_input_dim, num_class):
    '''
    Deep neural network model running on input from embedding layer
    '''
    inputs = Input(shape=(input_size, ))
    embedding_layer = Embedding(embedding_input_dim,
                            128,
                            input_length=input_size)(inputs)
    x = Flatten()(embedding_layer)
    x = Dense(32, activation='relu')(x)

    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=predictions)
    return model