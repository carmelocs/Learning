import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from dataset import load_all_data, S3DISDataset
from model import GAC_Net
import argparse
from pathlib import Path
from tqdm import tqdm
import datetime
import logging
from collections import defaultdict

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
log_dir = file_dir.joinpath(opt.log_dir)
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

BATCH_SIZE = opt.batchSize
WORKERS = opt.workers
NUM_EPOCH = opt.nepoch
LR = opt.learning_rate
WEIGHT_DECAY = opt.weight_decay
OPTIMIZER = opt.optimizer
DROPOUT = opt.dropout
ALPHA = opt.alpha

print('Loading data...')
train_data, train_label, test_data, test_label = load_all_data(test_area=5)
train_dataset = S3DISDataset(data=train_data, label=train_label)
test_dataset = S3DISDataset(data=test_data, label=test_label)

train_dataloader = data.DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=WORKERS)

NUM_CLASSES = train_dataset.num_classes
NUM_POINTS = train_dataset.num_points

test_dataloader = data.DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE * 2,
                                  shuffle=True,
                                  num_workers=WORKERS)

print(f'Length of training dataset: {len(train_dataset)}')
logger.info(f'Length of training dataset: {len(train_dataset)}')
print(f'Length of testing dataset: {len(test_dataset)}')
logger.info(f'Length of testing dataset: {len(test_dataset)}')
print(f'Number of classes: {NUM_CLASSES}')
logger.info(f'Number of classes: {NUM_CLASSES}')


def blue(x):
    return '\033[94m' + x + '\033[0m'


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed
    by 30 every 20000 steps
    """
    lr = LR * (0.3**(step // 20000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


num_batch = int(len(train_dataset) / BATCH_SIZE) \
    if len(train_dataset) % BATCH_SIZE == 0 \
    else int(len(train_dataset) / BATCH_SIZE)+1
print(f'Number of batches: {num_batch}')

history = defaultdict(lambda: list())
best_acc = 0
step = 0
correct = 0

k_radius = range(1, 5)
k_score = []

for k in k_radius:
    net = GAC_Net(num_classes=NUM_CLASSES,
                  radius=0.1 * k,
                  dropout=DROPOUT,
                  alpha=ALPHA)

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

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(init_epoch, NUM_EPOCH):
        for i, (point_cloud, label) in tqdm(enumerate(train_dataloader),
                                            total=len(train_dataloader)):
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
            pred_num = pred.argmax(dim=-1)
            correct += (pred_num == label).sum().item()

        accuracy = 100 * correct / (len(train_dataset) * NUM_POINTS)
        correct = 0
        print(f'Epoch: {epoch}/{NUM_EPOCH-1}, accuracy:{accuracy:.3f}%')
        logger.info(f'Epoch: {epoch}/{NUM_EPOCH-1}, accuracy:{accuracy:.3f}%')

        if accuracy > best_acc:
            best_acc = accuracy
            print('Saving model...')
            logger.info('Save model')
            torch.save(
                net.state_dict(),
                f'{checkpoints_dir}/GACNet_{epoch}_{best_acc:.4f}%.pth')

    k_score.append(best_acc)

print(f'Best radius: {k_score.index(max(k_score))*0.1}')
logger.info(f'Best radius: {k_score.index(max(k_score))*0.1}')
