import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_images(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        print(label, " is ready to load")
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (64, 64))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)

train_dir = "data/asl_alphabet_train"

uniq_labels = sorted(os.listdir(train_dir))
images, labels = load_images(directory=train_dir)
print("Data has been loaded")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)

n = len(uniq_labels)
train_n = len(X_train)
test_n = len(X_test)

print("Total number of symbols: ", n)
print("Number of training images: ", train_n)
print("Number of testing images: ", test_n)

def print_images(image_list):
    n = int(len(image_list) / len(uniq_labels))
    cols = 8
    rows = 4

    for i in range(len(uniq_labels)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    plt.show()

y_train_in = y_train.argsort()
y_train = y_train[y_train_in]
X_train = X_train[y_train_in]

print("Training Images: ")
print_images(image_list=X_train)

y_test_in = y_test.argsort()
y_test = y_test[y_test_in]
X_test = X_test[y_test_in]

print("Testing images: ")
print_images(image_list=X_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(29, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=101, batch_size=64)

score = model.evaluate(x=X_test, y=y_test, verbose=0)
print('Accuracy for test images:', round(score[1]*100, 3), '%')

print(hist)
model.save('SignLanguageV2.0.h5')
