from keras.models import load_model
import cv2
import numpy as np
import os, sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y', 'Z', 'del', 'nothing', 'space']

model = load_model('SignLanguage.h5')

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

def predict_model(image):
    predict = model.predict(image)
    result = class_names[np.argmax(predict)]
    for i in range(len(predict[0])):
        if predict[0][i] == 1.0:
            print(class_names[i])
    return result

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)

        MainWindow.resize(761, 560)
        MainWindow.setWindowIcon(QtGui.QIcon("img/logo.jpg"))
        self.Main = QtWidgets.QWidget(MainWindow)

        self.lSLS = QtWidgets.QLabel(self.Main)
        self.lSLS.setGeometry(QtCore.QRect(370, 0, 381, 511))
        self.lSLS.setText("")
        self.lSLS.setPixmap(QtGui.QPixmap("img/SLS.jpg"))

        self.bChoosefile = QtWidgets.QPushButton(self.Main)
        self.bChoosefile.setGeometry(QtCore.QRect(40, 30, 131, 51))
        self.bChoosefile.setFont(font)
        self.bChoosefile.setDefault(True)
        self.bChoosefile.clicked.connect(lambda: self.getfile())

        self.bOpencamera = QtWidgets.QPushButton(self.Main)
        self.bOpencamera.setGeometry(QtCore.QRect(200, 30, 131, 51))
        self.bOpencamera.setFont(font)
        self.bOpencamera.setDefault(True)
        self.bOpencamera.clicked.connect(lambda: self.startCam())

        self.lPlaceholder = QtWidgets.QLabel(self.Main)
        self.lPlaceholder.setGeometry(QtCore.QRect(40, 100, 290, 311))
        self.lPlaceholder.setText("")
        self.lPlaceholder.setPixmap(QtGui.QPixmap("img/logo.jpg"))
        self.lPlaceholder.setAlignment(QtCore.Qt.AlignCenter)
        self.lPlaceholder.setScaledContents(True)

        self.lDectectionarea = QtWidgets.QLabel(self.Main)
        self.lDectectionarea.setGeometry(QtCore.QRect(40, 440, 290, 51))
        font.setPointSize(30)
        self.lDectectionarea.setFont(font)
        self.lDectectionarea.setAlignment(QtCore.Qt.AlignCenter)

        MainWindow.setCentralWidget(self.Main)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.cam = Camera()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sign Detection App"))
        self.bChoosefile.setText(_translate("MainWindow", "Chọn file"))
        self.bOpencamera.setText(_translate("MainWindow", "Mở camera"))
        self.lDectectionarea.setText(_translate("MainWindow", "Xin chào !!"))

    def getfile(self):
        self.cam.stop()
        try:
            fname = QFileDialog.getOpenFileName(self.Main, 'Open file', 'data/asl_alphabet_test/', "Image files (*.jpg *.gif *.png)")
            img = cv2.imread(fname[0])
            img = cv2.resize(img, (64, 64))
            img = img.reshape(1, 64, 64, 3)
            print(model)
            prediction = predict_model(img)
            self.lPlaceholder.setPixmap(QtGui.QPixmap(fname[0]))
            self.lDectectionarea.setText(prediction)
        except:
            pass

    def startCam(self):
        self.cam.start()
        self.cam.ImageUpdate.connect(self.camera)

    def camera(self, source):
        self.lDectectionarea.setText(self.cam.prediction)
        self.lPlaceholder.setPixmap(QtGui.QPixmap.fromImage(source))

class Camera(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.prediction = None

    def run(self):
        self.ThreadActive = True
        capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = capture.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                flip = cv2.flip(image, 1)
                Convert2Qt = QImage(flip.data, flip.shape[1], flip.shape[0], QImage.Format_RGB888)
                pic = Convert2Qt.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                self.ImageUpdate.emit(pic)

                img = cv2.resize(frame, (64, 64))
                img = img.reshape(1, 64, 64, 3)
                self.prediction = predict_model(img)

    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
