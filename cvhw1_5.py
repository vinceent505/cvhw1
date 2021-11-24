import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import pickle
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Ui_hw1_5_gui import *
import cv2
import numpy as np
import os
import random 
from torchvision import models

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class mainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_5_1.clicked.connect(q5_1)
        self.pushButton_5_2.clicked.connect(q5_2)
        self.pushButton_5_3.clicked.connect(q5_3)
        self.pushButton_5_4.clicked.connect(q5_4)
        self.pushButton_5_5.clicked.connect(lambda: q5_5(self.spinBox.value()))


bach_size = 256
learning_rate = 1e-3
num_epoches = 100

#讀資料
train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def q5_1():
    plt.figure(figsize=(12, 4))
    res = [random.randrange(1, 50000, 1) for i in range(10)]
    for i, index in enumerate(res):
        show = plt.subplot(2, 5, i+1)
        img, label = train_dataset[index]
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        show.title.set_text(classes[label])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def q5_2():
    print("hyperparameters:")
    print("batch size: 512")
    print("learning rate: 0.001")
    print("optimizer: SGD")

def q5_3():
    model = VGG16().to(device)
    summary(model, (3, 32, 32))

def q5_4():
    
    record = pickle.load( open( "VGG16_record", "rb" ) )

    training_loss = []
    testing_loss = []
    training_accuracy = []
    testing_accuracy = []
    for i in range(len(record)):
        training_loss.append(record[i][0])
        testing_loss.append(record[i][2])
        training_accuracy.append(record[i][1]*100)
        testing_accuracy.append(record[i][3]*100)

    # #### Accuracy

    plt.figure(figsize=(9,5))
    best_acc_idx = np.argmax(testing_accuracy)
    show_best_acc = str(testing_accuracy[best_acc_idx])+'%' 

    plt.plot(training_accuracy, label="Training")
    plt.plot(testing_accuracy,  label="Testing")
    plt.annotate(show_best_acc, xy = (best_acc_idx, testing_accuracy[best_acc_idx]), color='black', fontsize=11,)
    plt.plot(best_acc_idx, testing_accuracy[best_acc_idx], 'x',color='red')
    plt.xlabel("epoch")
    plt.ylabel("%")
    plt.title("Accuracy")
    plt.legend(loc = 4)
    plt.savefig('Accuracy.png')
    plt.show()

def q5_5(test_num):
    models = torch.load("VGG16_new_model.pkl", map_location=torch.device('cpu'))
    test_index = test_num
    data, label = test_dataset[int(test_index)]
    img = data.numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    data = torch.unsqueeze(data, 0)

    models.eval()
    out = models(data)
    
    softmax = nn.Softmax(dim = 1)
    softmax = softmax(out).tolist()[0]
    _, pred = torch.max(out.data, 1)
    ans = classes[label]
    pred = classes[pred.item()]

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('target=' + ans + ', predict=' + pred, fontsize=12)

    plt.subplot(1, 2, 2)
    plt.bar([i for i in range(10)], softmax, align='center')
    plt.xticks(range(10), classes, fontsize=10, rotation = 0)
    plt.show()

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        #參照VGG16的架構
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            #2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            #4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            #14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            #16
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        #相當於np的resize的用法，-1代表不固定，交由資料量和前一個parameter推斷
        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return out

def train():
    model = VGG16().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    record = []
    for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch+1), '*' * 25)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in tqdm(enumerate(train_loader, 1)):
            img, label = data[0].to(device), data[1].to(device)
            #轉成torch方便讀取的資料
            img = Variable(img)
            label = Variable(label)

            out = model(img)
            #計算loss
            loss = criterion(out, label)
            running_loss += loss.item()*label.size(0)

            _, pred = torch.max(out, 1)
            num_correct = (pred==label).sum()
            accuracy = (pred==label).float().mean()
            running_acc += num_correct.item()

            #梯度歸零
            optimizer.zero_grad()
            #根據loss進行back propagation並計算gradient
            loss.backward()
            #更新梯度
            optimizer.step()
        training_loss = running_loss/len(train_dataset)
        training_acc = running_acc/len(train_dataset)
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, running_loss/(len(train_dataset)), running_acc/len(train_dataset)))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for data in test_loader:
                img, label = data[0].to(device), data[1].to(device)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred==label).sum()
                eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss/(len(test_dataset)), eval_acc/(len(test_dataset))))
        print(" ")

        testing_acc = eval_acc/len(test_dataset)
        testing_loss = eval_loss/len(test_dataset)

        record.append((training_loss, training_acc, testing_loss, testing_acc))
    with open('VGG16_record', 'wb') as f:
        pickle.dump(np.array(record), f)
    torch.save(model,'VGG16_new_model.pkl', _use_new_zipfile_serialization=False)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())