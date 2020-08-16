from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import h5py
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_data_files(filenames):
    return [line.rstrip() for line in open(filenames)]


def load_h5_file(h5_filename):
    f = h5py.File(h5_filename, 'r')
    # data_batch: (num_sample, num_point, data),
    # label_batch: (num_sample, num_point)
    # data dim：e.g., [XYZ, RGB, Normalized coordinates]
    data = f['data']
    label = f['label']
    return data, label


def load_data(all_files, step=1):
    data_batch_list = []
    label_batch_list = []
    for h5_filename in all_files[::step]:
        # print(h5_filename)
        data_batch, label_batch = load_h5_file(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    # put all data batches together w.r.t point clouds
    data_batches = np.concatenate(data_batch_list, 0)
    # e.g., (23585, 4096, 9) for S3DIS
    label_batches = np.concatenate(label_batch_list, 0)
    # e.g., (23585, 4096) for S3DIS
    return data_batches, label_batches


class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=2048, train=True):

        self.root = root
        self.num_points = num_points
        self.train = train

        if self.train:
            self.all_files = get_data_files(
                os.path.join(self.root,
                             'data/modelnet40_ply_hdf5_2048/train_files.txt'))
        else:
            self.all_files = get_data_files(
                os.path.join(self.root,
                             'data/modelnet40_ply_hdf5_2048/test_files.txt'))

        self.class_names = get_data_files(
            os.path.join(self.root,
                         'data/modelnet40_ply_hdf5_2048/shape_names.txt'))

        self.num_classes = len(self.class_names)  # 40 classes
        # data:(9840, 2048, 3) label:(9840, 1) for training
        # data:(2468, 2048, 3) label:(2468, 1) for testing
        self.data_batch, self.label_batch = load_data(self.all_files)

        assert len(self.data_batch) == len(self.label_batch)

    def __getitem__(self, index):
        point_cloud = self.data_batch[index]
        point_labels = self.label_batch[index]
        return point_cloud, point_labels

    def __len__(self):
        return len(self.label_batch)


class S3DISDataset(Dataset):
    def __init__(self,
                 root,
                 num_points=4096,
                 step=1,
                 train=True,
                 test_area='Area_5'):

        self.root = root
        self.num_points = num_points
        self.train = train
        self.test_area = test_area
        self.step = step

        # put all the file pathes into a list
        self.all_files = get_data_files(
            os.path.join(self.root,
                         'indoor3d_sem_seg_hdf5_data/all_files.txt'))
        # get a list of 23585 annotations(point clouds),
        # each annotation has 4096 points
        room_filelist = get_data_files(
            os.path.join(self.root,
                         'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))
        self.class_names = [
            'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
            'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
        ]
        self.num_classes = len(self.class_names)
        self.data_batch, self.label_batch = load_data(self.all_files,
                                                      step=self.step)

        assert len(self.data_batch) == len(self.label_batch)

        self.room_filelist = []
        self.train_idxs = []
        self.test_idxs = []
        for j in range(0, len(self.all_files), self.step):
            self.room_filelist.extend(room_filelist[j * 1000:(j + 1) * 1000])
        for i, room_name in enumerate(self.room_filelist):
            if self.test_area in room_name:
                self.test_idxs.append(i)
            else:
                self.train_idxs.append(i)
        self.train_data = self.data_batch[self.train_idxs]
        self.train_label = self.label_batch[self.train_idxs]
        self.test_data = self.data_batch[self.test_idxs]
        self.test_label = self.label_batch[self.test_idxs]

    def __getitem__(self, index):
        if self.train:
            point_cloud = self.train_data[index]
            point_labels = self.train_label[index]
        else:
            point_cloud = self.test_data[index]
            point_labels = self.test_label[index]
        return point_cloud, point_labels

    def __len__(self):
        return (len(self.train_idxs) if self.train else len(self.test_idxs))


class S3DISDatasetLite(Dataset):
    def __init__(self, all_files, num_points=4096, train=True, split=18):
        self.num_points = num_points
        self.train = train
        self.all_files = all_files

        if self.train:
            self.all_files = self.all_files[:split]
        else:
            self.all_files = self.all_files[split:]

        self.class_names = [
            'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
            'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
        ]
        self.num_classes = len(self.class_names)
        self.data_batch, self.label_batch = load_data(self.all_files)

        assert len(self.data_batch) == len(self.label_batch)

    def __getitem__(self, index):
        point_cloud = self.data_batch[index]
        point_labels = self.label_batch[index]
        return point_cloud, point_labels

    def __len__(self):
        return len(self.label_batch)


if __name__ == '__main__':
    # BATCH_SIZE = 64
    # modelnet = ModelNetDataset(root=BASE_DIR)
    # test_modelnet = ModelNetDataset(root=BASE_DIR, train=False)
    # print(f'Length of modelnet train dataset:\n{len(modelnet)}')
    # print(f'Length of modelnet test dataset:\n{len(test_modelnet)}')
    # modelnet_dataloader = DataLoader(modelnet, batch_size=BATCH_SIZE)
    # for i, (point_cloud, label) in enumerate(modelnet_dataloader):
    #     if i % 10 == 9:
    #         print(f'{i+1}\n{point_cloud.shape}\n{label.shape}')
    # TEST_AREA = 'Area_2'
    # train_dataset = S3DISDataset(root=BASE_DIR, step=2)
    # test_dataset = S3DISDataset(root=BASE_DIR,
    #                             train=False, step=2)
    # print(f'Test Area: {train_dataset.test_area}')
    # print(f'Length of S3DIS train dataset:\n{len(train_dataset)}')
    # print(f'Length of S3DIS test dataset:\n{len(test_dataset)}')

    all_files = get_data_files(
            os.path.join(BASE_DIR,
                         'indoor3d_sem_seg_hdf5_data/all_files.txt'))
    # print(self.all_files)
    random.shuffle(all_files)
    # print(self.all_files)
    train_dataset = S3DISDatasetLite(all_files=all_files)
    test_dataset = S3DISDatasetLite(all_files=all_files, train=False)
    print(f'Length of S3DIS train dataset:\n{len(train_dataset)}')
    print(f'Length of S3DIS test dataset:\n{len(test_dataset)}')

    dataloader = DataLoader(train_dataset, batch_size=64)
    for i, (point_cloud, label) in enumerate(dataloader):
        if i % 100 == 99:
            print(f'{i+1}\n{point_cloud.shape}\n{label.shape}')
