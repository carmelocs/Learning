from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

def get_data_files(filenames):
    return [line.rstrip() for line in open(filenames)]

def load_h5_file(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data']
    label = f['label']

    return data, label

def load_data(all_files):
    data_list = []
    label_list = []
    
    for h5_filename in all_files:
        data_batch, label_batch = load_h5_file(h5_filename)
        data_list.append(data_batch)
        label_list.append(label_batch)
        
    data = np.concatenate(data_list, 0)
    label = np.concatenate(label_list, 0)

    return data, label

class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=2048, train=True):

        self.root = root
        self.num_points = num_points
        self.train = train

        if self.train:
            self.all_files = get_data_files('./data/modelnet40_ply_hdf5_2048/train_files.txt')
        else:
            self.all_files = get_data_files('./data/modelnet40_ply_hdf5_2048/test_files.txt')
        
        self.class_names = get_data_files('./data/modelnet40_ply_hdf5_2048/shape_names.txt')

        self.num_classes = len(self.class_names)

        self.data, self.label = load_data(self.all_files)

        assert len(self.data) == len(self.label)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        point_label = self.label[idx]

        return point_cloud, point_label

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    import os

    BATCH_SIZE = 8
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(f'Base dir: {BASE_DIR}')

    train_dataset = ModelNetDataset(root=BASE_DIR)
    test_dataset = ModelNetDataset(root=BASE_DIR, train=False)
    print(f'Length of modelnet train dataset:\n{len(train_dataset)}')
    print(f'Length of modelnet test dataset:\n{len(test_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    for i, (point_cloud, label) in enumerate(train_dataloader):
        if i % 100 == 0:
            print(f'iteration {i}:\ndata: {point_cloud.shape}\nlabel: {label.shape}')

    # all_files = get_data_files('data/modelnet40_ply_hdf5_2048/train_files.txt')
    # print(all_files)
    # for h5_filename in all_files:
    #     data, label = load_h5_file(h5_filename)
    # print(data)
    