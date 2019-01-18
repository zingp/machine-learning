from keras.models import Sequential  
from keras.layers import Dense
from keras.datasets import mnist


# 导入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_test.shape)
print(X_train[0])
# model = Sequential()

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])