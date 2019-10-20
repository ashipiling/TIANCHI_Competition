from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils
#数据处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784)/255
x_test = x_test.reshape(x_test.shape[0], 784)/255
print(y_train[0])
#转为多维
y_train = np_utils.to_categorical(y_train, 10)
print(y_train[0])
y_test = np_utils.to_categorical(y_test, 10)
print(x_train.shape)

modle = Sequential()

modle.add(Dense(input_dim = 784, units= 40))
modle.add(Activation('relu'))
modle.add(Dense(10))
modle.add(Activation('softmax'))

modle.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

#batch_size小包训练
#epoch每个数据利用次数
loss = modle.fit(x_train, y_train, batch_size=100,epochs = 3)
print(loss)
print(modle.evaluate(x_test, y_test))
y_pre = modle.predict(x_test)
print(y_pre[0], y_test[0])



