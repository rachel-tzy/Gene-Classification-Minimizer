#coding=utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


# 设置最频繁使用的4个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)
MAX_NB_WORDS = 64
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 196
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100

#定义模型
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
