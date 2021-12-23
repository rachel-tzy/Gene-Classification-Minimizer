#coding=utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D


# Set the most frequently used 64 words(texts_to_matrix will takes first MAX_NB_WORDS columns)
MAX_NB_WORDS = 64
# max length for each sequence
MAX_SEQUENCE_LENGTH = 196

# Define Model
model = Sequential()
model.add(LSTM(256, input_shape=(196*2, 64), return_sequences=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.save('5genus-prediction-model_extremer.hdf5')
