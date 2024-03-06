# -*- coding = utf-8 -*-
# @Time : 2023/12/29 21:34
# @Author : 邓建森
# @Software: PyCharm

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

# 类别
class_dit = {
    0:"扁藻",
    1:"小球藻",
    2:"杜氏盐藻",
    3:"虫黄藻",
    4:"紫球藻",
    5:"雨生红球藻"
}

# 处理数据集
class YoloDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir  # 图像目录
        self.annotation_dir = annotation_dir   # 标签目录
        self.transform = transform  # transform从参数获得
        self.images = sorted(list(filter(lambda x: x.endswith('.jpg'), os.listdir(image_dir))))  # 加载图像路径
        self.annotations = [open(os.path.join(annotation_dir, image.split('.')[0] + '.txt')).read() for image in self.images] # 加载标签路径

        self.imgs, self.labels = [], [] # 存放图像和标签
        # 预处理图像和标签
        print("加载数据集")
        t0 = time.time() # 计算事件
        for i in range(len(self.images)):
            image_name = self.images[i]
            image = Image.open(os.path.join(image_dir, image_name))
            annotation = self.annotations[i].split('\n')
            for label in annotation:
                label = label.replace('\n', '')
                ret = label.split(' ')
                width, height = image.size
                # 将归一化的标签恢复为原始坐标
                x, y, w, h = float(ret[1]) * width, float(ret[2]) * height, float(ret[3]) * width, float(ret[4]) * height
                # 根据坐标裁剪图像
                box = (x - w, y - h, x + w, y + h)
                waterBacteria = image.crop(box)
                # 将图像转为tensor，加入数据集
                if self.transform:
                    waterBacteria = self.transform(waterBacteria)
                self.imgs.append(waterBacteria)
                # 将对应的标签加入
                self.labels.append(int(ret[0]))
        t1 = time.time()
        print(f"处理数据集完成,用时:{t1 - t0}秒")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

# 测试集
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 获取测试集目录下的图片
        test_img_names = os.listdir(test_dir)
        # 遍历图像
        for img_name in test_img_names:
            img_path = os.path.join(test_dir, img_name)
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


# 定义神经网络结构
class WaterBacteriaClassifier(nn.Module):
    # 参数为分类数量
    def __init__(self, num_classes):
        super(WaterBacteriaClassifier, self).__init__()
        # 输入图像的尺寸为(1, 32, 32)，为灰度图，大小是32 * 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU() # 激活函数
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 池化
        # 卷积
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 以下为全连接层
        self.fc1 = nn.Linear(in_features=32 * 8 * 8,out_features=512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(64, num_classes)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# 超参数设置, 学习率和batch size
batch_size = 16
learning_rate = 0.05

# 图像路径
image_dir = 'images/train'
annotation_dir = 'labels/train'
test_dir = "labels/test"

# 数据预处理和加载
# 定义transform, 将图像resize到32 * 32后转为灰度图
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),])  # 将图像转换为张量

train_dataset = YoloDataset(image_dir, annotation_dir, transform=transform) # 数据集
test_dataset = TestDataset(test_dir, transform=transform) # 测试集

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 数据集加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # 测试集加载器

# 实例化神经网络和优化器
model = WaterBacteriaClassifier(6)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练时的损失和准确率
train_loss = []
train_accuracy = []

# 训练，参数为训练轮数
def train(num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0 # 训练时的总损失
        total = 0  # 样本总数
        sum_correct = 0  # 预测正确的数量
        for i, (images, labels) in enumerate(train_loader):  # 对每一个批次的数据进行迭代训练
            outputs = model(images)  # 前向传播
            loss = F.cross_entropy(outputs, labels) # 计算损失函数值(交叉熵)
            optimizer.zero_grad()  # 清零梯度缓存，为反向传播做准备
            loss.backward()  # 反向传播，计算梯度值
            optimizer.step()  # 更新权重参数值，进行一次参数更新操作
            running_loss += loss.item() # 累加损失
            _, predict = torch.max(outputs.data, dim=1) # 将概率最大的作为预测结果
            total += predict.size(0)  # 计算样本总数
            sum_correct += (predict == labels).sum().item()  # 计算样本分类正确总数

        # 将训练损失和准确率添加到列表，方便绘制图像
        train_loss.append(running_loss / len(train_loader))
        train_accuracy.append(100 * sum_correct / total)
        print('Epoch %s accuracy: %.3f' % (epoch + 1, 100 * sum_correct / total))
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# 测试
def test():
    for image, label in test_loader:
        outputs = model(image)
        _, predict = torch.max(outputs.data, dim=1)
        # 打印预测结果
        for i in range(predict.size(0)):
            print(f'预测值:{predict[i].item()}, 类别:{class_dit[predict[i].item()]}')

if __name__ == '__main__':
    train(50)
    plt.title("Train Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss)
    plt.show()

    plt.title("Train Accuracy")
    plt.ylabel("accuracy")
    plt.plot(train_accuracy)
    plt.show()

    # 测试集
    test()