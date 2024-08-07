import struct
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class MinistDataset(Dataset):

    def __init__(self, img_path, mask_path):

        with open(mask_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            self.all_mask = np.fromfile(lbpath, dtype=np.uint8)
        with open(img_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            self.all_image = np.fromfile(imgpath, dtype=np.uint8).reshape(len(self.all_mask), 784)

        self.img_list = []
        self.mask_list = []
        self.individual_row_image = []

        for i in range(self.all_image.shape[0]):
            self.individual_row_image.append([self.all_image[i]])

        self.img_num = self.all_image.shape[0]

        pbar = tqdm(enumerate(self.individual_row_image), unit='preprocessing',  # 放入可迭代对象
                    total=len(self.individual_row_image))

        for idx, img_label in pbar:
            self.img_list.append(np.reshape(self.individual_row_image[idx][0], [1, 28, 28]))
            self.mask_list.append(self.all_mask[idx])

    def __len__(self) -> int:
        return self.img_num

    def __getitem__(self, idx: int) -> Tuple:
        img = self.img_list[idx]
        mask = self.mask_list[idx]
        img = torch.tensor(img).float()  # todo: 好像得转成float不然类型不一致会报错
        return img, mask


class Convd(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.convd = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5),
                                   nn.Sigmoid(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, img):
        return self.convd(img)


class fc(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=in_feature, out_features=out_feature), nn.Sigmoid())

    def forward(self, img):
        return self.fc(img)


class LLnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Convd(1, 6)
        self.conv_2 = Convd(6, 16)
        self.fc_1 = fc(256, 120)
        self.fc_2 = fc(120, 84)
        self.fc_3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, img: torch.tensor) -> torch.tensor:  # init的参数传给init,forward的参数传给forward
        out_1 = self.conv_1(img)
        out_2 = self.conv_2(out_1)
        out_3 = self.fc_1(out_2.view(batch_size, -1))
        out_4 = self.fc_2(out_3)
        out_5 = self.fc_3(out_4)
        return out_5


# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
#             nn.Sigmoid(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
#             nn.Sigmoid(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=256, out_features=120),
#             nn.Sigmoid(),
#             nn.Linear(in_features=120, out_features=84),
#             nn.Sigmoid(),
#             nn.Linear(in_features=84, out_features=10),
#         )
#
#     def forward(self, img: torch.tensor) -> torch.tensor:
#         feature = self.conv(img)
#         output = self.fc(feature)
#         return output


# def evaluate(model,device,dataloder)-> float:


if __name__ == '__main__':
    train_img_path = 'data/minist/train_img/train-images.idx3-ubyte'
    train_mask_path = 'data/minist/train_mask/train-labels.idx1-ubyte'
    test_img_path = 'data/minist/test_img/t10k-images.idx3-ubyte'
    test_mask_path = 'data/minist/test_mask/t10k-labels.idx1-ubyte'
    model_save_dir = 'result/models/minist'
    is_train = 1
    batch_size = 16
    num_of_epoch = 50

    from torch.utils.data import DataLoader

    if is_train == 1:

        # Dataloader的对象
        train_loader = DataLoader(dataset=MinistDataset(train_img_path,
                                                        train_mask_path),
                                  batch_size=batch_size,
                                  shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = LLnet().to(device)  # 模型的对象
        num_of_batch = DataLoader.__len__(train_loader)  # todo:返回对象的长度(注意，这里是num/batch_size=batch)返回的是batch的数量
        n_train = num_of_batch * batch_size

        loss_function = nn.CrossEntropyLoss()
        loss_list = []
        acc_list = []
        epoch_list = []  # 以上三个都是用于画图
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        for epoch in range(num_of_epoch):
            model.train()
            correct = 0
            running_loss = 0

            pbar = tqdm(enumerate(train_loader), unit='img', total=num_of_batch)  # 注意,total指定batch而不是图片数量
            for idx, data in pbar:
                img, mask = data  # 输入为[16,1,28,28],为[batch_size,C,H,W],16张图片放入一起训练,第一维16即为16张图片的信息
                img = img.to(device)
                mask = mask.to(device)
                output = model(img)  # 输出为[16,10],即为16张图片,每张图片是[0~9]的概率预测

                for i, label in enumerate(mask):
                    if torch.argmax(output[i]) == mask[i]:
                        correct = correct + 1

                loss = loss_function(output, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_list.append(running_loss / num_of_batch)
            epoch_list.append(epoch + 1)
            acc_list.append(correct / n_train)

            print('\n epoch = %d轮迭代，loss = %.3f，acc = %.3f\n' % (epoch, loss_list[-1], correct / n_train))
            savepath = model_save_dir + '/' + "Epoch{}_LS-{:.3f}_DC-{:.3f}.pth".format(epoch, loss_list[-1],
                                                                                       correct / n_train)
            torch.save(model.state_dict(), savepath)

            fig_loss_acc = plt.figure()
            fig_featuremap = plt.figure()

            axes_loss = fig_loss_acc.add_axes([0.1, 0.1, 0.8, 0.8])
            axes_acc = fig_loss_acc.add_axes([0.4, 0.5, 0.5, 0.4])

            axes_acc.plot(epoch_list, acc_list, 'r')
            axes_acc.set_xlabel('epoch')
            axes_acc.set_ylabel('accuracy')

            axes_loss.plot(epoch_list, loss_list, 'g')
            axes_loss.set_xlabel('epoch')
            axes_loss.set_ylabel('loss')
            fig_loss_acc.savefig('result/loss/loss.png', dqi=600)
            plt.show()


    else:

        test_loader = DataLoader(dataset=MinistDataset(test_img_path,
                                                       test_mask_path),
                                 batch_size=1,  # todo:注意，测试时batch_size要设为1
                                 shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LLnet().to(device)  # 模型的对象
        num_of_batch = DataLoader.__len__(test_loader)  # todo:返回对象的长度(注意，这里是num/batch_size=batch)返回的是batch的数量
        n_test = num_of_batch * 1

        state_dict = torch.load('result/models/minist/Epoch10_LS-0.001_DC-0.991.pth')
        model.load_state_dict(state_dict)  # 恢复net模型的参数，net为自定义的和载入模型相同结构的网络

        model.eval()  # todo:将Dropout层和batch normalization层设置成预测模式

        loss_function = nn.CrossEntropyLoss()

        correct = 0
        with torch.no_grad():  # todo:来关闭梯度的计算
            pbar = tqdm(enumerate(test_loader), unit='img', total=n_test)
            for idx, data in pbar:
                img, mask = data
                img = img.to(device)
                mask = mask.to(device)
                output = model(img)

                if torch.argmax(output) == mask:
                    correct = correct + 1

            print('\n evaluation  acc = %.3f\n' % (correct / n_test))
