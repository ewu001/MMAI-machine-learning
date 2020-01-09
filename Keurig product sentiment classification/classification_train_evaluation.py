
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score

import keras 
from keras.models import Sequential


from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import preprocessor
import model as md

MAX_LENGTH = 500

#read in datafile
df = pd.read_csv('data store/training_data.csv', names=['comment', 'label'], header=None)

# pre process
# converts label into integer values
df['label'] = df.label.astype('category').cat.codes
#counts the number of classes
num_class = len(np.unique(df.label.values))
y = df['label'].values
print("\nThere are a total of " + str(num_class) + " classes.")

padded_seq, tokenizer = preprocessor.tokenization(MAX_LENGTH, df.comment.values)


#train test split
X_train, X_test, y_train, y_test = train_test_split(padded_seq, y, test_size=0.3)
vocab_size = len(tokenizer.word_index) + 1

dnn_model = md.DNN_model(MAX_LENGTH, vocab_size, num_class)

dnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

#prints out summary of model
dnn_model.summary()
#saves the model weights
filepath="weights/weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = dnn_model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=10, callbacks=[checkpointer])


#plot model
df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')

#get prediction accuarcy
predicted = dnn_model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
print(accuracy_score(y_test, predicted))