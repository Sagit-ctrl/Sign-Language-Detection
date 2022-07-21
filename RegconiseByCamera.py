from keras.models import load_model
import cv2
import numpy as np
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

camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

score = 0
top, right, bottom, left = 170, 150, 425, 450
predictThreshold = 95

def predict_model(image):
    predict = model.predict(image)
    result = class_names[np.argmax(predict)]
    print(max(predict[0]))
    score = float("%0.2f" % (max(predict[0]) * 100))
    for i in range(len(predict[0])):
        if predict[0][i] == 1.0:
            print(class_names[i])
    return result, score

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]

    cv2.imshow("Area Detection", roi)

    img = cv2.resize(roi, (64, 64))
    img = img.reshape(1, 64, 64, 3)
    prediction, score = predict_model(img)

    print(score, prediction)
    if (score >= predictThreshold):
        cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
    thresh = None

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

    cv2.imshow("Sign Language Detection", cv2.resize(clone, dsize=None, fx=0.5, fy=0.5))
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)


cv2.destroyAllWindows()
camera.release()
