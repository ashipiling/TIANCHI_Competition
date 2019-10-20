from keras.datasets import mnist
from keras.layers import Dense,Activation,Convolution2D, MaxPooling2D,Flatten
from keras.models import Sequential
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28,28, 1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255
print(y_train[0])
#转为多维
y_train = np_utils.to_categorical(y_train, 10)
print(y_train[0])
y_test = np_utils.to_categorical(y_test, 10)
print(x_train.shape)

model = Sequential()
model.add(Convolution2D(10,3,3,input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size= 100, epochs=10, verbose = 2)

print(model.evaluate(x_test, y_test))

