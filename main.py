import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
                          Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
                          AveragePooling2D((2,2)),
                          Conv2D(64, (3,3), padding='same', activation='relu'),
                          AveragePooling2D((2,2)),
                          Conv2D(128, (3,3), padding='same', activation='relu'),
                          AveragePooling2D((2,2)),
                          Conv2D(128, (3,3), padding='same', activation='relu'),
                          AveragePooling2D((2,2)),
                          Flatten(),
                          Dense(2048, activation='relu'),
                          BatchNormalization(),
                          Dense(512, activation='relu'),
                          BatchNormalization(),
                          Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
his = model.fit(x_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()

for i in range(10):
    n = np.random.randint(0, 10000)
    x = np.expand_dims(x_test[n], axis=0)
    res_ind = np.argmax(model.predict(x))
    orig_ind = np.argmax(y_test_cat[n])

    dataset_classes = {0: 'Футболка/Топ', 1: 'Брюки', 2: 'Пуловер', 3: 'Платье', 4: 'Пальто', 5: 'Сандалия',
                       6: 'Рубашка', 7: 'Кроссовок', 8: 'Сумка', 9: 'Ботинок'}

    print('Ожидаемое значение: ', dataset_classes[orig_ind], '\nПрогнозируемое значение: ', dataset_classes[res_ind])

    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()