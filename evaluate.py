import logging
from typing import Optional
from ShowingOutput import show_slices
import SimpleITK as sitk
import numpy as np
import yaml
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.preprocess import PatchDataset
from evaluation.dice_score import dice_score
from model.vnet import *


def calculate_auc_func1(y_labels: ndarray, y_scores: ndarray):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]
    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    auc = sum_indicator_value / (len(pos_sample_ids) * len(neg_sample_ids))
    print('AUC calculated by function1 is {:.2f}'.format(auc))
    return auc


def calculate_auc_func2(y_labels, y_scores):
    samples = list(zip(y_scores, y_labels))
    rank = [(values2, values1) for values1, values2 in sorted(samples, key=lambda x: x[0])]
    pos_rank = [i + 1 for i in range(len(rank)) if rank[i][0] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(pos_rank) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt + 1e-6)
    print('AUC calculated by function2 is {:.2f}'.format(auc))
    return auc


def evaluate(
        model: VNet,
        device: torch.device,
        dataloader: DataLoader,
        img_size: ndarray,
        ID_list: list,
        dim: Optional[int] = 1
) -> float:
    mask = img = [[np.zeros([x[1][0], x[1][1], x[1][2]]), np.zeros([x[1][0], x[1][1], x[1][2]])] for x in img_size]

    dice_scores = []
    auc_scores = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            ID, direction, volume, segmentation = data
            direction = [int(x) for x in direction]
            ID = str(ID.item())
            idx = ID_list.index(int(ID))

            volume = volume.to(device, dtype=torch.float)
            if dim == 1:
                segmentation = torch.clamp(segmentation, 0, 1)
            segmentation = segmentation.to(device, dtype=torch.uint8)

            output = model(volume)

            bbb = (torch.sigmoid(output) > 0.5).float().cpu().detach().numpy()
            bbb = np.squeeze(bbb)
            bbb = np.squeeze(bbb)
            img[idx][0][direction[0]:direction[0] + 64, direction[1]:direction[1] + 64,
            direction[2]:direction[2] + 64] = bbb.transpose([2, 0, 1])

            # ass = sitk.GetImageFromArray(img[idx][0])
            # sitk.WriteImage(ass, 'result/AB/+{},{}_{}_{}.nii.gz'.format(ID_list[idx], direction[0], direction[1],
            #                                                             direction[2]))

            ccc = output.float().cpu().detach().numpy()
            ccc = np.squeeze(ccc)
            ccc = np.squeeze(ccc)
            sss = ccc.reshape(1, -1)
            sss = np.squeeze(sss)

            ccc = segmentation.float().cpu().detach().numpy()
            ccc = np.squeeze(ccc)
            ccc = np.squeeze(ccc)
            mask[idx][0][direction[0]:direction[0] + 64, direction[1]:direction[1] + 64,
            direction[2]:direction[2] + 64] = ccc.transpose([2, 0, 1])
            xxx = ccc.reshape(1, -1)
            xxx = np.squeeze(xxx)

            dice_scores.append(dice_score((torch.sigmoid(output) > 0.5).float(), segmentation.float()))

            try:
                sk_learn = roc_auc_score(xxx, sss)
            except ValueError:
                sk_learn = -1

            print('skl_auc = {:.2f}'.format(sk_learn))
            auc_scores.append(sk_learn)

            show_slices(ID, direction, bbb, ccc, dice_scores[-1], auc_scores[-1], False)

            for i in range(len(ID_list)):
                img_ = sitk.GetImageFromArray(img[i][0])
                sitk.WriteImage(img_, 'result/nii/' + 'img915_{}.nii.gz'.format(str(ID_list[i])))
                mask_ = sitk.GetImageFromArray(mask[i][0])
                sitk.WriteImage(mask_, 'result/nii/' + 'mask915_{}.nii.gz'.format(str(ID_list[i])))
    return sum(dice_scores) / len(dice_scores)


if __name__ == '__main__':
    with open("./config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info('Using device {}'.format(device))

    net = VNet()
    net.to(device=device)
    s = torch.load('result/models/913/Epoch9_LS-0.984_DC-0.060.pth')
    net.load_state_dict(s)
    # torch.backends.cudnn.benchmark = True

    dataset = PatchDataset(config['pathing']['test_img_dirs'],
                           config['pathing']['test_mask_dirs'], config, True)
    img_size = dataset.img_size
    ID_list = [img_size[i][0] for i in range(len(img_size))]

    testloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False)
    n_test = DataLoader.__len__(testloader)
    dice_coeff = evaluate(model=net, device=device, dataloader=testloader, img_size=img_size, ID_list=ID_list, dim=1)
    print(dice_coeff)
