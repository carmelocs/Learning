import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from dataset import ModelNetDataset
from model import RandLANet
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import datetime
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=4,
                    help='input batch size [default: 4]')
parser.add_argument('--num_workers',
                    type=int,
                    default=0,
                    help='number of data loading workers [default: 0]')
parser.add_argument('--nepoch',
                    type=int,
                    default=120,
                    help='number of epochs to train for [default: 120]')
parser.add_argument('--log_dir',
                    type=str,
                    default='logs/',
                    help='log directory [default: logs]')
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='whether use pretrain model [default: None]')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help='learning rate for training [default: 0.01]')
parser.add_argument('--optimizer',
                    type=str,
                    default='Adam',
                    help='type of optimizer [default: Adam]')
parser.add_argument('--dropout',
                    type=float,
                    default=0,
                    help='dropout [defautl: 0]')


opt = parser.parse_args()
print(opt)

'''EXPERIMENT DIR '''
experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(
    str(experiment_dir) + '/' +
    str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)
checkpoints_dir = file_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)

'''LOG'''
log_dir = file_dir.joinpath(opt.log_dir)
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger('RandLA_Net')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(str(log_dir) + '/train_RandLANet.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('PARAMETER ...')
logger.info(opt)

BATCH_SIZE = opt.batch_size
NUM_WORKERS = opt.num_workers
NUM_EPOCH = opt.nepoch
LR = opt.learning_rate
OPTIMIZER = opt.optimizer
DROPOUT = opt.dropout

print('Loading data...')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dataset = ModelNetDataset(root=BASE_DIR, train=True)
test_dataset = ModelNetDataset(root=BASE_DIR, train=False)

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = data.DataLoader(test_dataset, batch_size=2*BATCH_SIZE)
NUM_CLASSES = train_dataset.num_classes
NUM_POINTS = train_dataset.num_points

print(f'Length of training dataset: {len(train_dataset)}')
logger.info(f'Length of training dataset: {len(train_dataset)}')
print(f'Length of testing dataset: {len(test_dataset)}')
logger.info(f'Length of testing dataset: {len(test_dataset)}')
print(f'Number of classes: {NUM_CLASSES}')
logger.info(f'Number of classes: {NUM_CLASSES}')

def blue(x):
    return '\033[94m' + x + '\033[0m'

best_acc = 0
TIME = 0

net = RandLANet(d_in=3, num_classes=NUM_CLASSES)

if opt.pretrain is not None:
    net.load_state_dict(torch.load(opt.pretrain))
    print(f'load model {opt.pretrain}')
    logger.info(f'load model {opt.pretrain}')

else:
    print('Training from scratch')
    logger.info('Training from scratch')

pretrain = opt.pretrain
init_epoch = int(pretrain[-14:-11]) if opt.pretrain is not None else 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs.')
    net = nn.DataParallel(net)
elif torch.cuda.device_count() == 1:
    print('Using one GPU.')
else:
    print('Using CPU.')

net.to(device)

optimizer = optim.Adam(net.parameters(), lr=LR)

for epoch in range(init_epoch, NUM_EPOCH):
    # torch.cuda.synchronize()
    start = time.time()
    train_correct = 0
    for i, (point_cloud, label) in tqdm(enumerate(train_dataloader),
                                        total=len(train_dataloader)):
        label = label.repeat(1, NUM_POINTS).view(-1).long()
        point_cloud, label = point_cloud.to(device), label.to(device)
        optimizer.zero_grad()
        net = net.train()
        pred = net(point_cloud).contiguous()
        pred = pred.view(-1, NUM_CLASSES)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        pred_num = pred.argmax(dim=-1)
        train_correct += (pred_num == label).sum().item()

    # torch.cuda.synchronize()
    end = time.time()
    TIME += end - start
    train_acc = 100 * train_correct / (len(train_dataset) * NUM_POINTS)
    print(
        f'Epoch: {epoch}/{NUM_EPOCH-1}, train accuracy: {train_acc:.3f}%, \
            Time: {end-start}s')
    logger.info(
        f'Epoch: {epoch}/{NUM_EPOCH-1}, train accuracy: {train_acc:.3f}%, \
            Time: {end-start}s')

    test_correct = 0
    for i, (point_cloud, label) in tqdm(enumerate(test_dataloader),
                                        total=len(test_dataloader)):
        label = label.repeat(1, NUM_POINTS).view(-1).long()
        point_cloud, label = point_cloud.to(device), label.to(device)
        net = net.eval()
        pred = net(point_cloud)
        pred = pred.view(-1, NUM_CLASSES)
        pred_num = pred.argmax(dim=-1)
        test_correct += (pred_num == label).sum().item()

    test_acc = 100 * test_correct / (len(test_dataset) * NUM_POINTS)
    blue_test = blue('test accuracy:')
    print(f'Epoch: {epoch}/{NUM_EPOCH-1}, {blue_test} {test_acc:.3f}%')
    logger.info(
        f'Epoch: {epoch}/{NUM_EPOCH-1}, test accuracy: {test_acc:.3f}%')

    if test_acc > best_acc:
        best_acc = test_acc
        print('Saving model...')
        logger.info('Save model')
        torch.save(
            net.state_dict(),
            f'{checkpoints_dir}/RandLANet_{epoch}_{best_acc:.4f}%.pth')