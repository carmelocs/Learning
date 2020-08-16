import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from dataset import get_data_files, ModelNetDataset, S3DISDataset, S3DISDatasetLite
from model import PointNetCls, PointNetSeg, GAC_Net
import argparse
import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize',
                    type=int,
                    default=4,
                    help='input batch size')
parser.add_argument('--workers',
                    type=int,
                    default=0,
                    help='number of data loading workers')
parser.add_argument('--nepoch',
                    type=int,
                    default=120,
                    help='number of epochs to train for')
# parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset',
                    type=str,
                    default='S3DIS',
                    help="dataset name")
parser.add_argument('--feature_transform',
                    action='store_true',
                    help="use feature transform")

opt = parser.parse_args()
print(opt)

DATASET = opt.dataset
BATCH_SIZE = opt.batchSize
WORKERS = opt.workers
NUM_EPOCH = opt.nepoch
FEATURE_TRANSFORM = opt.feature_transform

if DATASET == 'S3DIS':
    all_files = get_data_files(
        os.path.join(BASE_DIR, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))
    # print(self.all_files)
    random.shuffle(all_files)
    # print(self.all_files)
    print('Loading data...')
    dataset = S3DISDatasetLite(all_files=all_files)
    test_dataset = S3DISDatasetLite(all_files=all_files, train=False)
else:
    print('Loading data...')
    dataset = ModelNetDataset(root=BASE_DIR)
    test_dataset = ModelNetDataset(root=BASE_DIR, train=False)

dataloader = data.DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=WORKERS)

NUM_CLASSES = dataset.num_classes
NUM_POINTS = dataset.num_points

test_dataloader = data.DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=WORKERS)

# print(f'Dataset: {DATASET}\nTest_Area: {dataset.test_area}')
print(f'Length of training dataset: {len(dataset)}')
print(f'Length of testing dataset: {len(test_dataset)}')
num_classes = dataset.num_classes
print(f'Number of classes: {num_classes}')

if DATASET == 'S3DIS':
    net = GAC_Net(NUM_CLASSES)
    # net = PointNetSeg(num_class=NUM_CLASSES, num_point=NUM_POINTS)
else:
    net = PointNetCls(num_class=NUM_CLASSES, num_point=NUM_POINTS)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
elif torch.cuda.device_count() == 1:
    print('Using one GPU.')
else:
    print('Using CPU.')

net.to(device)

optim = optim.Adam(net.parameters(), lr=0.001)

num_batch = int(len(dataset) / BATCH_SIZE) \
    if len(dataset) % BATCH_SIZE == 0 \
    else int(len(dataset) / BATCH_SIZE)+1
print(f'Number of batches: {num_batch}')

for epoch in range(NUM_EPOCH):
    for i, (point_cloud, label) in enumerate(dataloader):
        label = label.view(-1).long()
        point_cloud, label = point_cloud.to(device), label.to(device)
        optim.zero_grad()
        net = net.train()
        pred = net(point_cloud)
        pred = pred.view(-1, NUM_CLASSES)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optim.step()
        pred_num = pred.argmax(dim=1)
        correct = (pred_num == label).sum().item()
        if DATASET == 'S3DIS':
            accuracy = 100 * correct / (BATCH_SIZE * NUM_POINTS)
        else:
            accuracy = 100 * correct / BATCH_SIZE
        print(f'Epoch: {epoch}/{NUM_EPOCH-1}, iter: {i}/{num_batch-1}, train loss: {loss.item():.3f}, \
accuracy:{accuracy:.3f}%')

        if i % 10 == 9:
            j, (test_points, test_labels) = next(enumerate(test_dataloader))
            test_labels = test_labels.view(-1).long()
            test_points, test_labels = test_points.to(device), test_labels.to(
                device)
            net = net.eval()
            pred = net(test_points)
            pred = pred.view(-1, NUM_CLASSES)
            loss = F.nll_loss(pred, test_labels)
            pred_num = pred.argmax(dim=1)
            correct = (pred_num == test_labels).sum().item()
            if DATASET == 'S3DIS':
                accuracy = 100 * correct / (BATCH_SIZE * NUM_POINTS)
            else:
                accuracy = 100 * correct / BATCH_SIZE
            print(
                f'\nEpoch: {epoch}, iter: {i}, test loss: {loss.item():.3f}, \
accuracy:{accuracy:.3f}%\n')
