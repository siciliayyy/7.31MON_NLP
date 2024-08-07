import torch
import yaml
from torch.utils.data import DataLoader
from dataset.preprocess import PatchDataset
from model.vnet import VNet
from train.train import train

if __name__ == '__main__':
    with open("./config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    train_loader = DataLoader(dataset=PatchDataset(config['pathing']['train_img_dirs'],
                                                   config['pathing']['train_mask_dirs'], config, True),
                              batch_size=config['dataset']['batch_size'],
                              shuffle=True)

    validation_loader = DataLoader(dataset=PatchDataset(config['pathing']['val_img_dirs'],
                                                        config['pathing']['val_mask_dirs'], config, True),
                                   batch_size=1,
                                   shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VNet().to(device)  # 创建一个类的实例方法(切patch)
    num_train_batch = DataLoader.__len__(train_loader)
    num_validation_batch = DataLoader.__len__(validation_loader)
    train(config, model, device, train_loader, validation_loader, num_train_batch, num_validation_batch)
