import cv2
import os

IMG_SIZE = 64
top, right, bottom, left = 100, 150, 400, 450
exitCondition = '**'
labelName = ''
check = "1234"
dirFolder = input('Nhập tên thư mục : ')

try:
    os.mkdir(dirFolder)
except:
    print('Đã tồn tại thư mục tương tự')

camera = cv2.VideoCapture(0)

while True:
    labelName = input('Gõ ** để thoát hoặc gõ tên nhãn : ')
    if labelName == exitCondition:
        break
    dirLabel = str(dirFolder) + '/' + str(labelName)
    print(dirLabel)
    try:
        os.mkdir(dirLabel)
    except:
        print('Thư mục đã chứa')
    i = 0
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.imshow("Video origin", frame)

        if i == 0:
            check = input("Nhấn Enter để bắt đầu")
        if check == "":
            cv2.imwrite("%s/%s/%d.jpg" % (dirFolder, labelName, i), img)
            i += 1
        print(i)
        if i > 500:
            check = "1234"
            break

        keypress = cv2.waitKey(1)
        if keypress == 27:
            break

camera.release()
cv2.destroyAllWindows()




