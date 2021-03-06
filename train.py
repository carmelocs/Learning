import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from dataset import load_all_data, ModelNetDataset, S3DISDataset
from model import PointNetCls, GAC_Net
import argparse
import os
import sys
from pathlib import Path
import datetime
import logging
# import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize',
                    type=int,
                    default=4,
                    help='input batch size [default: 4]')
parser.add_argument('--workers',
                    type=int,
                    default=0,
                    help='number of data loading workers [default: 0]')
parser.add_argument('--nepoch',
                    type=int,
                    default=120,
                    help='number of epochs to train for [default: 120]')
# parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset',
                    type=str,
                    default='S3DIS',
                    help="dataset [default: S3DIS]")
parser.add_argument('--feature_transform',
                    action='store_true',
                    help="use feature transform")
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help='learning rate for training [default: 0.01]')
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-4,
                    help='weight decay for Adam')
parser.add_argument('--optimizer',
                    type=str,
                    default='Adam',
                    help='type of optimizer [default: Adam]')
parser.add_argument('--dropout',
                    type=float,
                    default=0,
                    help='dropout [defautl: 0]')
parser.add_argument('--alpha',
                    type=float,
                    default=0.2,
                    help='alpha for leakyRelu [default: 0.2]')

opt = parser.parse_args()
print(opt)

experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(
    str(experiment_dir) + '/' +
    str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)
checkpoints_dir = file_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)
'''LOG'''
logger = logging.getLogger('GACNet')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(str(log_dir) + '/train_GACNet.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('PARAMETER ...')
logger.info(opt)

DATASET = opt.dataset
BATCH_SIZE = opt.batchSize
WORKERS = opt.workers
NUM_EPOCH = opt.nepoch
FEATURE_TRANSFORM = opt.feature_transform
LR = opt.learning_rate
WEIGHT_DECAY = opt.weight_decay
OPTIMIZER = opt.optimizer
DROPOUT = opt.dropout
ALPHA = opt.alpha

if DATASET == 'S3DIS':
    print('Loading data...')
    train_data, train_label, test_data, test_label = load_all_data(test_area=5)
    # print(self.all_files)
    # random.shuffle(all_files)
    # print(self.all_files)
    train_dataset = S3DISDataset(data=train_data, label=train_label)
    test_dataset = S3DISDataset(data=test_data, label=test_label)
else:
    print('Loading data...')
    dataset = ModelNetDataset(root=BASE_DIR)
    test_dataset = ModelNetDataset(root=BASE_DIR, train=False)

train_dataloader = data.DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=WORKERS)

NUM_CLASSES = train_dataset.num_classes
NUM_POINTS = train_dataset.num_points

test_dataloader = data.DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=WORKERS)

# print(f'Dataset: {DATASET}\nTest_Area: {dataset.test_area}')
print(f'Length of training dataset: {len(train_dataset)}')
print(f'Length of testing dataset: {len(test_dataset)}')
print(f'Number of classes: {NUM_CLASSES}')

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


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed
    by 30 every 20000 steps
    """
    lr = LR * (0.3**(step // 20000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

num_batch = int(len(train_dataset) / BATCH_SIZE) \
    if len(train_dataset) % BATCH_SIZE == 0 \
    else int(len(train_dataset) / BATCH_SIZE)+1
print(f'Number of batches: {num_batch}')

step = 0

for epoch in range(NUM_EPOCH):
    for i, (point_cloud, label) in enumerate(train_dataloader):
        label = label.view(-1).long()
        point_cloud, label = point_cloud.to(device), label.to(device)
        optimizer.zero_grad()
        net = net.train()
        pred = net(point_cloud)
        pred = pred.view(-1, NUM_CLASSES)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        step += 1
        adjust_learning_rate(optimizer, step)

        pred_num = pred.argmax(dim=1)
        correct = (pred_num == label).sum().item()
        if DATASET == 'S3DIS':
            accuracy = 100 * correct / (BATCH_SIZE * NUM_POINTS)
        else:
            accuracy = 100 * correct / BATCH_SIZE
        print(f'Epoch: {epoch}/{NUM_EPOCH-1}, iter: {i}/{num_batch-1}, \
train loss: {loss.item():.3f}, accuracy:{accuracy:.3f}%')

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
