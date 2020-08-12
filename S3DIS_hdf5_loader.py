import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_data_files(filenames):
    return [line.rstrip() for line in open(filenames)]


def load_h5_file(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data']
    label = f['label']
    return data, label


def load_data(all_files):
    data_batch_list = []
    label_batch_list = []
    for h5_filename in all_files:
        # data_batch: (1000, 4096, 9), label_batch: (1000, 4096)
        # except for the last batch: (585, 4096, 9) and (585, 4096)
        # data dim：[XYZ, RGB, Normalized coordinates] (9 dims)
        data_batch, label_batch = load_h5_file(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    # put all 24 data batches together w.r.t point clouds
    data_batches = np.concatenate(data_batch_list, 0)  # (23585, 4096, 9)
    label_batches = np.concatenate(label_batch_list, 0)  # (23585, 4096)
    return data_batches, label_batches


# put all the file pathes into a list
all_files = get_data_files('indoor3d_sem_seg_hdf5_data/all_files.txt')

# get a list of 23585 annotations(point clouds),
# each annotation has 4096 points
room_filelist = get_data_files('indoor3d_sem_seg_hdf5_data/room_filelist.txt')

data_batches, label_batches = load_data(all_files)

class_names = get_data_files(
    '/home/shuai_cheng/Documents/Repositories/pointnet/sem_seg/meta \
    /class_names.txt'
)

test_area = 'Area_1'  # take Area_1 as the test dataset
train_idxs = []
test_idxs = []
for i, room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
