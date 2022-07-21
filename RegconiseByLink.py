from keras.models import load_model
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khai báo model
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y', 'Z', 'del', 'nothing', 'space']

model = load_model('SignLanguage.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

image = cv2.imread("data/asl_alphabet_test/B_test.jpg")
image = cv2.resize(image, (64, 64))
image = image.reshape(1, 64, 64, 3)

predict = model.predict(image)
for i in range(len(predict[0])):
    if predict[0][i] == 1:
        print(class_names[i])


