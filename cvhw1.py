import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal
from sklearn.preprocessing import normalize

from PyQt5.QtWidgets import QApplication, QMainWindow
from Ui_hw1_gui import *

class mainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_1_1.clicked.connect(q1_1)
        self.pushButton_1_2.clicked.connect(q1_2)
        self.pushButton_1_3.clicked.connect(q1_3)
        self.pushButton_1_4.clicked.connect(q1_4)
        self.pushButton_2_1.clicked.connect(q2_1)
        self.pushButton_2_2.clicked.connect(q2_2)
        self.pushButton_2_3.clicked.connect(q2_3)
        self.pushButton_3_1.clicked.connect(q3_1)
        self.pushButton_3_2.clicked.connect(q3_2)
        self.pushButton_3_3.clicked.connect(q3_3)
        self.pushButton_3_4.clicked.connect(q3_4)
        self.pushButton_4_1.clicked.connect(q4_1)
        self.pushButton_4_2.clicked.connect(q4_2)
        self.pushButton_4_3.clicked.connect(q4_3)
        self.pushButton_4_4.clicked.connect(q4_4)
        
        
    
def q1_1():
    img = cv2.imread("Q1_Image/Sun.jpg")
    print("Height :", img.shape[0])
    print("Width :", img.shape[1])
    pass

def q1_2():
    img = cv2.imread("Q1_Image/Sun.jpg")
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    (b, g, r) = cv2.split(img)
    img_concat = np.concatenate((cv2.merge([b, zeros, zeros]), cv2.merge([zeros, g, zeros]), cv2.merge([zeros, zeros, r])), axis = 1)
    cv2.imshow("Q1_2", img_concat)
    cv2.waitKey(0)
    pass

def q1_3():
    img = cv2.imread("Q1_Image/Sun.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j]) * 0.33
    print(img)
    cv2.imshow("Q1_3_1", img_gray)
    cv2.waitKey(0)
    cv2.imshow("Q1_3_2", img)
    cv2.waitKey(0)
    
    pass

def nothing(x):
    pass

def q1_4():
    img1 = cv2.imread("C:\\Users\\User\\Desktop\\Hw1\\cvhw1\\Q1_Image\\Dog_Strong.jpg")
    img2 = cv2.imread("C:\\Users\\User\\Desktop\\Hw1\\cvhw1\\Q1_Image\\Dog_Weak.jpg")
    cv2.namedWindow("Q1_4")
    cv2.createTrackbar("Blend", "Q1_4", 0, 255, nothing)
    while True:
        t = cv2.getTrackbarPos("Blend", "Q1_4")
        img = cv2.addWeighted(img1, t/256, img2, (255-t)/256, 0)
        cv2.imshow("Q1_4", img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        pass
    cv2.destroyAllWindows()


def q2_1():
    img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
    gaussian_filter = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])/273
    img_filted = cv2.filter2D(img, -1, gaussian_filter)
    out = np.concatenate((img, img_filted), axis = 1)
    cv2.imshow("Q2_1", out)
    cv2.waitKey(0)
    pass

def q2_2():
    img = cv2.imread("Q2_Image/Lenna_whiteNoise.jpg")
    bilateral = cv2.bilateralFilter(img, 9, 90, 90)
    out = np.concatenate((img, bilateral), axis = 1)
    cv2.imshow("Q2_2", out)
    cv2.waitKey(0)
    pass

def q2_3():
    img = cv2.imread("Q2_Image/Lenna_pepperSalt.jpg")
    median_3 = cv2.medianBlur(img, 3)
    median_5 = cv2.medianBlur(img, 5)
    out = np.concatenate((img, median_3, median_5), axis = 1)
    cv2.imshow("Q2_3", out)
    cv2.waitKey(0)

    pass

def q3_1():
    img = cv2.imread("Q3_Image/House.jpg")
    dev = 3
    gaussian_pos = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]], dtype="float32")
    gaussian_matrix = np.zeros((3, 3), dtype="float32")
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] = (1/(2*np.pi*(dev**2)))*(np.exp(-(j[0]**2+j[1]**2)/(2*(dev**2))))
            pass
    s = np.sum(gaussian_matrix)
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] /= s
            pass

    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j]) * 0.33
    out = scipy.signal.convolve2d(img[:, :, 0], gaussian_matrix)


    out = cv2.merge([out, out, out])
    cv2.imshow("Q3_1", np.uint8(out))
    cv2.waitKey(0)

def q3_2():
    img = cv2.imread("Q3_Image/House.jpg")
    dev = 3
    gaussian_pos = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]], dtype="float32")
    gaussian_matrix = np.zeros((3, 3), dtype="float32")
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] = (1/(2*np.pi*(dev**2)))*(np.exp(-(j[0]**2+j[1]**2)/(2*(dev**2))))
            pass
    s = np.sum(gaussian_matrix)
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] /= s
            pass

    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j]) * 0.33
    out = scipy.signal.convolve2d(img[:, :, 0], gaussian_matrix)
    out = cv2.merge([out, out, out])

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='int16')
    x_sol = cv2.filter2D(np.int16(out), -1, sobel_x)
    x_sol[x_sol<0]=0
    cv2.imshow("Q3_2", np.uint8(x_sol))
    cv2.waitKey(0)

    pass

def q3_3():
    img = cv2.imread("Q3_Image/House.jpg")
    dev = 3
    gaussian_pos = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]], dtype="float32")
    gaussian_matrix = np.zeros((3, 3), dtype="float32")
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] = (1/(2*np.pi*(dev**2)))*(np.exp(-(j[0]**2+j[1]**2)/(2*(dev**2))))
            pass
    s = np.sum(gaussian_matrix)
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] /= s
            pass

    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j]) * 0.33
    out = scipy.signal.convolve2d(img[:, :, 0], gaussian_matrix)
    out = cv2.merge([out, out, out])

    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='int16')
    y_sol = cv2.filter2D(np.int16(out), -1, sobel_y)
    y_sol[y_sol<0]=0
    cv2.imshow("Q3_3", np.uint8(y_sol))
    cv2.waitKey(0)

def q3_4():
    img = cv2.imread("Q3_Image/House.jpg")
    dev = 3
    gaussian_pos = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]], dtype="float32")
    gaussian_matrix = np.zeros((3, 3), dtype="float32")
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] = (1/(2*np.pi*(dev**2)))*(np.exp(-(j[0]**2+j[1]**2)/(2*(dev**2))))
            pass
    s = np.sum(gaussian_matrix)
    for i_num, i in enumerate(gaussian_pos):
        for j_num, j in enumerate(i):
            gaussian_matrix[i_num][j_num] /= s
            pass

    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j]) * 0.33
    out = scipy.signal.convolve2d(img[:, :, 0], gaussian_matrix)
    out = cv2.merge([out, out, out])

    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='int16')
    x_sol = cv2.filter2D(np.int16(out), -1, sobel_x)
    x_sol[x_sol<0]=0
    
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='int16')
    y_sol = cv2.filter2D(np.int16(out), -1, sobel_y)
    y_sol[y_sol<0]=0

    sobel_sol = (x_sol**2 + y_sol**2)**(1/2)
    cv2.imshow("Q3_4", np.uint8(sobel_sol))
    cv2.waitKey(0)

def q4_1():
    img = cv2.imread("Q4_Image/SQUARE-01.png")
    img1 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Q4_1_1", img)
    cv2.imshow("Q4_1_2", img1)
    cv2.waitKey(0)
    pass


def q4_2():
    img = cv2.imread("Q4_Image/SQUARE-01.png")
    (row, col) = img.shape[0:2]
    H = np.float32([[1, 0, 0], [0, 1, 60]])

    out = cv2.warpAffine(img, H, (row, col))

    cv2.imshow("Q4_2_1", img)
    cv2.imshow("Q4_2_2", out)
    cv2.waitKey(0)
    pass


def q4_3():
    img = cv2.imread("Q4_Image/SQUARE-01.png")
    (row, col) = img.shape[0:2]
    H = np.float32([[1, 0, 0], [0, 1, 60]])
    sol_1 = cv2.warpAffine(img, H, (row, col))

    scale = 0.5

    M = cv2.getRotationMatrix2D((128, 188), 10, scale)
    out_2 = cv2.warpAffine(sol_1, M, (int(scale*row), int(scale*col)))
    cv2.imshow("Q4_3_1", img)
    cv2.imshow("Q4_3_2", out_2)
    cv2.waitKey(0)
    pass

def q4_4():
    img = cv2.imread("Q4_Image/SQUARE-01.png")
    (row, col) = img.shape[0:2]
    H = np.float32([[1, 0, 0], [0, 1, 60]])
    sol_1 = cv2.warpAffine(img, H, (row, col))

    scale = 0.5

    M = cv2.getRotationMatrix2D((128, 188), 10, scale)
    out_2 = cv2.warpAffine(sol_1, M, (int(scale*row), int(scale*col)))


    ps1 = np.float32([[50, 50], [200, 50], [50, 200]])
    ps2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(ps1, ps2)
    out_3 = cv2.warpAffine(out_2, M, (row, col))

    cv2.imshow("Q4_3_1", img)
    cv2.imshow("Q4_3_2", out_3)
    cv2.waitKey(0)
    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())