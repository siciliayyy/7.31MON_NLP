import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from evaluation.dice_score import dice_score


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def get_optimizer(model, config):
    optimizer = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]

    if optimizer == "SGD":
        momentum = config["training"]["momentum"]
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )
    # TODO: implement first and second moments
    elif optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    return optimizer


def get_criterion(config):
    criterion = config["training"]["criterion"]

    class FusionLoss(nn.Module):
        def __init__(self):
            super(FusionLoss, self).__init__()

        def forward(self, logits, targets):
            num = targets.size(0)
            smooth = 1

            probs = torch.sigmoid(logits)
            m1 = probs.view(num, -1)
            m2 = targets.view(num, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = 1 - score.sum() / num
            CE = torch.nn.BCELoss()
            score2 = CE(logits, targets) * 0.3
            score += score2
            return score

    if criterion == 'FusionLoss':
        criterion = FusionLoss()
    else:
        criterion = nn.BCELoss()
    # TODO: implement DiceLoss
    return criterion


def get_scheduler(optimizer, config):
    if not config["training"]["use_scheduler"]:
        return

    scheduler = config["training"]["scheduler"]
    if scheduler == "StepLR":
        step_size = config["training"]["scheduler_step"]
        gamma = config["training"]["scheduler_gamma"]
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    # TODO: implement other schedulers

    return scheduler


def train(config, model, device, dataloader, valdataloader, num_train_batch, num_val_batch):
    run_name = config["pathing"]["run_date"]
    model_save_dir = config["pathing"]["model_save_dir"]
    batch_size = config['dataset']['batch_size']
    if not os.path.exists(model_save_dir + str(run_name)):
        os.makedirs(model_save_dir + str(run_name))

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config).to(device)
    scheduler = get_scheduler(optimizer, config)

    total_epochs = config["training"]["epochs"]
    x = list()
    y = list()
    b = list()
    losses = []

    train_num = num_train_batch*batch_size
    val_num = num_val_batch*batch_size

    for epoch in range(0, total_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        pbar = tqdm(enumerate(dataloader), unit='img',
                    total=num_train_batch)

        for i, data in pbar:

            ID, volume, segmentation = data
            volume = volume.to(device, dtype=torch.float)
            segmentation = segmentation.to(device, dtype=torch.float)

            optimizer.zero_grad()
            output = model(volume)
            output = torch.sigmoid(output)

            loss = criterion(output, segmentation)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()

        # evaluation
        model.eval()
        losses.append(running_loss / (i + 1))

        tot = 0
        with torch.no_grad():
            for j, data in enumerate(valdataloader):
                ID, volume, segmentation = data
                volume = volume.to(device, dtype=torch.float)
                segmentation = segmentation.to(device, dtype=torch.float)
                val_pred = model(volume)

                pred = (torch.sigmoid(val_pred) > 0.5).float()
                tot += dice_score(pred, segmentation)


        print('\nLearning rate: {}'.format(get_learning_rate(optimizer)[0]))
        print('Training Loss: {:.4f}'.format(running_loss / num_train_batch))
        print('Dice Coeff: {:.4f}'.format(tot / (num_val_batch*batch_size)))
        x.append(epoch + 1)
        y.append(running_loss / num_train_batch)
        b.append(tot / (num_val_batch*batch_size))
        plt.figure(1)
        plt.plot(x, y, "g", label="loss")
        plt.xlabel("epoch")
        plt.ylabel("Training loss")
        plt.title('loss')
        plt.savefig("Train loss.jpg")
        plt.figure(2)
        plt.plot(x, b, "r", label="coeff")
        plt.xlabel("epoch")
        plt.ylabel("Validation Coeff")
        plt.title('dice')
        plt.savefig("Validation Coeff.jpg")
        if epoch % 5 == 4:
            train_loss = running_loss / ((num_train_batch - 0.01) // config["dataset"]["batch_size"] + 1)
            dice = tot / (num_val_batch*batch_size)
            savepath = model_save_dir + str(run_name) + '/' + "Epoch{}_LS-{:.3f}_DC-{:.3f}.pth".format(epoch, train_loss, dice)
            torch.save(model.state_dict(), savepath)

            print('Checkpoint {} saved !'.format(epoch + 1))
        print()

        # train_loss = running_loss / ((num_train_batch - 0.01) // config["dataset"]["batch_size"] + 1)
        # dice = tot / num_val_batch
        # savepath = model_save_dir + str(run_name) + '/' + "Epoch{}_LS-{:.3f}_DC-{:.3f}.pth".format(epoch, train_loss,
        #                                                                                       dice)
        # torch.save(model.state_dict(), savepath)
        #
        # print('Checkpoint {} saved !'.format(epoch + 1))
        # print()
    return losses
