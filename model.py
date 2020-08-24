import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
import torch.nn.functional as F
# from dataset import ModelNetDataset, S3DISDataset
# import os


class PointFeature(nn.Module):
    def __init__(self, num_point, global_feature=True):
        super(PointFeature, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool1d(num_point)
        self.global_feature = global_feature
        self.num_point = num_point

    def forward(self, x):
        # input of Conv1d:(batch, in_channel, num_point)
        # e,g,. (64, 3, 2048) for ModelNet40
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        local_feature = x
        # (batch, channel, num_point) e.g., (64, 64, 2048) for ModelNet40

        x = F.relu(self.bn3(self.conv3(x)))  # (64, 128, 2048)
        x = F.relu(self.bn4(self.conv4(x)))  # (64, 1024, 2048)
        x = self.pool(x)  # (64, 1024, 1)
        if self.global_feature:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1,
                                           self.num_point)  # (64, 1024, 2048)
            return torch.cat((x, local_feature), dim=1)  # (64, 1088, 2048)


class PointNetCls(nn.Module):
    def __init__(self, num_class, num_point):
        super(PointNetCls, self).__init__()
        self.point_feature = PointFeature(num_point=num_point)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)  # keep ratio = 0.7

    def forward(self, x):
        x = self.point_feature(x)  # (64, 1024, 1)
        x = F.relu(self.bn1(self.fc1(x.transpose(2,
                                                 1)).squeeze(1)))  # (64, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # (64, 256)
        x = self.fc3(x)  # (64, num_class)
        return x.log_softmax(dim=-1)  # (64, num_class)


class PointNetSegFC(nn.Module):
    def __init__(self, num_class, num_point):
        super(PointNetSegFC, self).__init__()
        self.point_feature = PointFeature(num_point=num_point,
                                          global_feature=False)
        self.fc1 = nn.Linear(1088, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)  # keep ratio = 0.7

    def forward(self, x):
        x = self.point_feature(x).transpose(2, 1)  # (64, 2048, 1088)
        x = F.relu(self.bn1(self.fc1(x).transpose(2, 1)))  # (64, 512, 2048)
        x = x.transpose(2, 1)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)).transpose(
            2, 1)))  # (64, 256, 2048)
        x = x.transpose(2, 1)
        x = self.fc3(x)  # (64, 2048, num_class)
        return x.log_softmax(dim=-1)


class PointNetSeg(nn.Module):
    def __init__(self, num_class, num_point):
        super(PointNetSeg, self).__init__()
        self.point_feature = PointFeature(num_point=num_point,
                                          global_feature=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, num_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.point_feature(x)  # (64, 1088, 2048)
        x = F.relu(self.bn1(self.conv1(x)))  # (64, 512, 2048)
        x = F.relu(self.bn2(self.conv2(x)))  # (64, 256, 2048)
        x = F.relu(self.bn3(self.conv3(x)))  # (64, 128, 2048)
        x = F.relu(self.bn4(self.conv4(x)))  # (64, 128, 2048)
        x = self.conv5(x)  # (64, num_class, 2048)
        return x.transpose(2, 1).log_softmax(dim=-1)  # (64, 2048, num_class)


def square_distance(src, dst):
    '''
    Input:
        src: [B, N, C]
        dst: [B, M, C]
    Retrun:
        dist: [B, N, M]
    '''

    B, N, C = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # print(dist.is_contiguous())
    dist += torch.sum(src**2, dim=-1).view(B, N, 1)
    # print(dist.is_contiguous())
    dist += torch.sum(dst**2, dim=-1).view(B, 1, M)
    # print(dist.is_contiguous())
    return dist


def farthest_point_sample(xyz, npoint):
    '''
    Input:
        xyz: [B, N, C]
        npoint: Number of point for centered points
    Return:
        centered_idx: [B, npoint]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    # [B, npoint]
    centered_idx = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # [B]
    # initialize the first point of each batch
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    # [B, N]
    # distance matrix keeps the distance from all points to the current
    # selected point set
    distance = torch.ones(B, N).to(device) * 1e10
    # [B]
    batch_indices = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(npoint):
        centered_idx[:, i] = farthest
        centered_xyz = xyz[batch_indices, farthest, :].view(B, 1, C)
        # print(centered_xyz.shape)
        # print(centered_xyz.is_contiguous())
        # compute the distance from the current farthest points to all N points
        # [B, N]
        dist = torch.sum((xyz - centered_xyz)**2, dim=-1)
        # take the shortest distance between the remaining points and the
        # selected points as the distance from the remaining points
        # to the selected point set
        # (if the point has been selected, the distance=0(itself)
        # will not be updated)
        mask = dist < distance
        distance[mask] = dist[mask]
        # update the farthest point from the remaining point set
        # by taking the index of the longest distance
        farthest = distance.max(-1)[1]
    # print(centered_idx.is_contiguous())
    return centered_idx


def index_points(points, idx):
    '''
    Input:
        points: [B, N, D]
        idx: [B, n_1, n_2,..., n_m]
    Return:
        new_points: [B, n_1, n_2,..., n_m, D]
        can use gather()
    '''
    # new_point = points.gather()
    device = points.device
    B, N, D = points.shape
    view_list = list(idx.shape)
    view_list[1:] = [1] * (len(view_list) - 1)
    repeat_list = list(idx.shape)
    repeat_list[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_list).repeat(
        repeat_list)
    # print(batch_indices.is_contiguous())
    new_points = points[batch_indices, idx, :]
    # print(new_points.is_contiguous())
    return new_points


def query_ball_point(nsample, radius, xyz, centered_xyz):
    '''
    Input:
        nsample: Number of point for each centered point/ball query
        radius: Radius of the local query ball region
        xyz: position data [B, N, C]
        centered_xyz: position data of query points [B, npoint, C]
    Return:
        grouped_idx: indices of neighbouring points [B, npoint, nsample]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    _, npoint, _ = centered_xyz.shape
    # [B, npoint, nsample]
    grouped_idx = torch.zeros(B, npoint, nsample, dtype=torch.long).to(device)
    # [B, npiont, N]
    distance = square_distance(centered_xyz, xyz)
    # [B, npoint, N]
    mask_N = distance > radius**2
    # [B, npoint, N], assign a very large number (regarded as infinity) to all
    # points outside the local ball region
    distance[mask_N] = 1e10
    # [B, npoint, nsample]
    # take the first nsample closest points of each query(centered) point
    # the numnber of local points might be less than nsample
    grouped_idx = distance.sort(dim=-1)[1][..., :nsample]
    # [B, npoint, nsample]
    # duplicate the first point index
    grouped_first = grouped_idx[..., 0].view(B, npoint,
                                             1).repeat(1, 1, nsample)
    # print(grouped_first.is_contiguous())
    # [B, npoint, nsample]
    # mask the indices of points outside the local ball region
    mask_nsample = distance.sort(dim=-1)[0][..., :nsample] == 1e10
    # print(mask_nsample.is_contiguous(), mask_N.is_contiguous())
    # [B, npoint, nsample]
    # assign the first point index to all the outside points
    grouped_idx[mask_nsample] = grouped_first[mask_nsample]
    grouped_idx = grouped_idx.contiguous()
    # print(grouped_idx.is_contiguous())
    return grouped_idx


def sample_and_group(npoint, radius, nsample, points):
    '''
    Input:
        npoint: Number of point for centered points (by FPS sampling)
        radius: Radius of the local query ball region
        nsample: Number of point for each centered point/ball query
        points: position data and features [B, N, C+D]
    Return:
        centered_points: [B, npoint, C+D]
        grouped_points: [B, npoint, nsample, C+D]
    '''
    device = points.device
    # [B, N, C]
    xyz = points[..., :3]
    # [B, N, D]
    features = points[..., 3:]
    # [B, npoint]
    centered_idx = farthest_point_sample(xyz, npoint).to(device)
    # [B, npoint, C]
    centered_xyz = index_points(xyz, centered_idx).to(device)
    # [B, npoint, D]
    centered_features = index_points(features, centered_idx).to(device)
    # [B, npoint, C+D]
    centered_points = torch.cat((centered_xyz, centered_features),
                                dim=-1).to(device)

    # [B, npoint, nsample]
    grouped_idx = query_ball_point(nsample, radius, xyz, centered_xyz)
    # [B, npoint, nsample, C]
    grouped_xyz = index_points(xyz, grouped_idx)
    # [B, npoint, nsample, D]
    grouped_features = index_points(features, grouped_idx)
    # [B, npoint, nsample, C+D]
    grouped_points = torch.cat((grouped_xyz, grouped_features), dim=-1)
    # print(centered_points.is_contiguous(), grouped_points.is_contiguous())
    return centered_points, grouped_points


def sample_and_group_all(points):
    '''
    This function takes all points as a group,
    i.e., npoint = 1 and nsample = N.

    Input:
        points: [B, N, C+D]
    Return:
        centered_points: [B, 1, C+D]
        grouped_points: [B, 1, N, C+D]
    '''
    B, N, _ = points.shape
    # [B]
    centered_idx = torch.randint(0, N, (B, ), dtype=torch.long)
    # [B, 1, C+D]
    centered_points = index_points(points, centered_idx).view(B, 1, -1)
    # print(centered_points.is_contiguous())
    # [B, 1, N, C+D]
    grouped_points = points.view(B, 1, N, -1)
    # print(centered_points.is_contiguous(), grouped_points.is_contiguous())
    return centered_points, grouped_points


class GraphAttention(nn.Module):
    def __init__(self, all_channel, feature_channel, dropout, alpha):
        super(GraphAttention, self).__init__()
        # initialize a matrix to compute attention scores of
        # each centered point and their neighbouring points
        self.alpha = alpha
        self.a = nn.Parameter(torch.ones(all_channel, feature_channel))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, centered_points, grouped_points):
        '''
        Input:
            centered_points: [B, npoint, C+D]
            grouped_points: [B, npoint, nsample, C+D]
        Return:
            centered_features: [B, npoint, D]
        '''
        B, npoint, nsample, dim = grouped_points.shape
        # [B, npoint, nsample, D]
        grouped_features = grouped_points[..., 3:]
        # [B, npoint, nsample, C+D]
        # compute the position and feature difference between
        # the centered point and its neighbouring points
        delta_p_concat_delta_h = grouped_points - centered_points.view(
            B, npoint, 1, dim)
        # print(delta_p_concat_delta_h.is_contiguous())
        # [B, npoint, nsample, D]
        attention = self.leakyrelu(torch.matmul(delta_p_concat_delta_h,
                                                self.a))
        # [B, npoint, nsample, D]
        # normalize along each centered point's nsample neighbouring points
        attention = attention.softmax(dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # [B, npoint, nsample, D]
        # return weighted sum of each centered points by element-wise product
        # of the computed attention scores and its responding neighbouring
        # points' features, the output works as the centered_features for the
        # next layer in GAC_Conv_layer
        centered_features = torch.sum(torch.mul(attention, grouped_features),
                                      dim=2)
        # print(centered_features.is_contiguous())
        return centered_features


class GraphAttentionConvLayer(nn.Module):
    def __init__(self,
                 npoint,
                 radius,
                 nsample,
                 mlp,
                 feature_channel,
                 dropout=0.6,
                 alpha=0.2):
        '''
        Input:
                npoint: Number of point for centered points (by FPS sampling)
                radius: Radius of the local query ball region
                nsample: Number of point for each centered point/ball query
                mlp: A list for mlp input-output channel, e.g., [64, 64, 128]
                feature_channel: the dimention of the input feature channel
        '''
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.dropout = dropout
        self.alpha = alpha
        self.conv2ds = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = feature_channel
        for out_channel in mlp:
            self.conv2ds.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.GAC = GraphAttention(3 + last_channel, last_channel, self.dropout,
                                  self.alpha)

    def forward(self, points):
        '''
        Input:
            points: [B, N, C+D]
        Return:
            new_points: the centered points with position data and
                        aggregated features from input points [B, npoint, C+D']
        '''
        B, N, _ = points.shape
        # [B, npoint, C+D]
        # [B, npoint, nsample, C+D]
        centered_points, grouped_points = sample_and_group(
            self.npoint, self.radius, self.nsample, points)
        # [B, npoint, C]
        centered_xyz = centered_points[..., :3]
        # [B, D, 1, npoint]
        # permute the dimensions for conv2d operation
        centered_features = centered_points[..., 3:].view(
            B, self.npoint, 1, -1).permute(0, 3, 2, 1).contiguous()
        # print(centered_features.is_contiguous())
        # [B, npoint, nsample, C]
        grouped_xyz = grouped_points[..., :3]
        # [B, D, nsample, npoint]
        grouped_features = grouped_points[..., 3:].permute(0, 3, 2, 1)

        # map centered and grouped features into high-level ones,
        # which have D' feature channels after the operation
        for i, conv2d in enumerate(self.conv2ds):
            bn = self.bns[i]
            centered_features = F.relu(bn(conv2d(centered_features)))
            grouped_features = F.relu(bn(conv2d(grouped_features)))

        # [B, npoint, D']
        centered_features = centered_features.permute(
            0, 3, 2, 1).squeeze(2).contiguous()
        # print(centered_features.is_contiguous())
        # [B, npoint, nsample, D']
        grouped_features = grouped_features.permute(0, 3, 2, 1).contiguous()
        # print(grouped_features.is_contiguous())
        # [B, npoint, C+D']
        centered_points = torch.cat((centered_xyz, centered_features), dim=-1)
        # [B, npoint, nsample, C+D']
        grouped_points = torch.cat((grouped_xyz, grouped_features), dim=-1)
        # [B, npoint, D']
        # compute the aggregated features of each centered point
        gac_features = self.GAC(centered_points, grouped_points)
        # [B, npoint, C+D']
        new_points = torch.cat((centered_xyz, gac_features), dim=-1)
        # print(new_points.is_contiguous())
        return new_points


class FeaturePropagationLayer(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagationLayer, self).__init__()
        self.conv1ds = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.conv1ds.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, layer1_points, layer2_points):
        '''
        Input:
            points_1: input of a GAC_Layer [B, N, C + D_1]
            points_2: output of the corresponding GAC_Layer
                        [B, npoint, C + D_2]

        Return:
            new_points: [B, N, C + D_3]
        '''
        B, N, dim = layer1_points.shape
        _, npoint, _ = layer2_points.shape
        # [B, N, C]
        layer1_xyz = layer1_points[..., :3]
        # [B, N, D_1]
        layer1_features = layer1_points[..., 3:]
        # [B, npoint, C]
        layer2_xyz = layer2_points[..., :3]
        # [B, npoint, D_2]
        layer2_features = layer2_points[..., 3:]
        # if there is only one sampled point, copy the features of this point
        # as the interpolated features
        if npoint == 1:
            # [B, N, D_2]
            interpolated_features = layer2_features.repeat(1, N, 1)
        # [B, N, npoint]
        distance = square_distance(layer1_xyz, layer2_xyz)
        # [B, N, npoint]
        dist, idx = distance.sort(-1)
        # [B, N, 3]
        # take three nearest neighbours
        dist_knn = dist[..., :3]
        idx_knn = idx[..., :3]
        # avoid dividing by 0
        dist_knn[dist_knn < 1e10] = 1e-10
        # [B, N, 3]
        weight = 1 / dist_knn
        # [B, N, 3, D_2]
        layer2_features_knn = index_points(layer2_features, idx_knn)
        # [B, N, D_2]
        interpolated_features = torch.sum(
            torch.mul(weight.view(B, N, 3, 1), layer2_features_knn),
            dim=-2) / torch.sum(weight, dim=-1).view(B, N, 1)
        # print(interpolated_features.is_contiguous())
        # if layer1 points only have position(xyz) data, i.e., D_1 = 0,
        # we just use interpolated features as its new features,
        # otherwise, concatenate the layer1 features with interpolated
        # features as new features
        if dim == 3:
            # [B, N, D_2]
            new_features = interpolated_features
        else:
            # [B, N, D_1 + D_2]
            new_features = torch.cat((layer1_features, interpolated_features),
                                     dim=-1)
        new_features = new_features.permute(0, 2, 1).contiguous()
        # print(new_features.is_contiguous())
        for i, conv1d in enumerate(self.conv1ds):
            bn = self.bns[i]
            new_features = F.relu(bn(conv1d(new_features)))

        # [B, N, C + D_3]
        new_points = torch.cat((layer1_xyz, new_features.permute(0, 2, 1)),
                               dim=-1)
        # print(new_points.is_contiguous())
        return new_points


class GAC_Net(nn.Module):
    def __init__(self, num_classes, radius, dropout=0, alpha=0.2):
        super(GAC_Net, self).__init__()
        self.num_classes = num_classes
        self.radius = radius
        self.dropout = dropout
        self.alpha = alpha
        # GraphAttentionConvLayer(npoint, radius, nsample, mlp,
        #                          feature_channel, dropout, alpha)
        # [B, 1024, 64+3]
        self.graph_pooling1 = GraphAttentionConvLayer(1024, self.radius, 32, [32, 64],
                                                      6, self.dropout,
                                                      self.alpha)
        # [B, 256, 128+3]
        self.graph_pooling2 = GraphAttentionConvLayer(256, self.radius*2, 32, [64, 128],
                                                      64, self.dropout,
                                                      self.alpha)
        # [B, 64, 256+3]
        self.graph_pooling3 = GraphAttentionConvLayer(64, self.radius*4, 32, [128, 256],
                                                      128, self.dropout,
                                                      self.alpha)
        # [B, 16, 512+3]
        self.graph_pooling4 = GraphAttentionConvLayer(16, self.radius*8, 32, [256, 512],
                                                      256, self.dropout,
                                                      self.alpha)

        # FeaturePropagationLayer(in_channel, mlp)
        # in_channel = 256 + 512
        self.fp4 = FeaturePropagationLayer(256 + 512, [256, 256])
        # in_channel = 128 + 256
        self.fp3 = FeaturePropagationLayer(128 + 256, [256, 256])
        # in_channel = 64 + 256
        self.fp2 = FeaturePropagationLayer(64 + 256, [128, 128])
        # in_channel = 6 + 128
        self.fp1 = FeaturePropagationLayer(6 + 128, [128, 128])

        # GraphAttentionConvLayer(npoint, radius, nsample, mlp,
        #                          feature_channel)
        self.GAC_Layer = GraphAttentionConvLayer(4096, 0.2, 32, [128, 64], 128)

        self.conv1d = nn.Conv1d(64 + 3, 64, 1)
        self.bn = nn.BatchNorm1d(64)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, points):
        # [B, 1024, 32+3]
        layer1_points = self.graph_pooling1(points)
        # [B, 256, 128+3]
        layer2_points = self.graph_pooling2(layer1_points)
        # [B, 64, 256+3]
        layer3_points = self.graph_pooling3(layer2_points)
        # [B, 16, 512+3]
        layer4_points = self.graph_pooling4(layer3_points)

        # [B, 64, 256+3]
        layer3_points = self.fp4(layer3_points, layer4_points)
        # [B, 256, 128+3]
        layer2_points = self.fp3(layer2_points, layer3_points)
        # [B, 1024, 32+3]
        layer1_points = self.fp2(layer1_points, layer2_points)
        # [B, 4096, 32+3]
        layer0_points = self.fp1(points, layer1_points)
        # [B, 4096, 64+3]
        layer0_points = self.GAC_Layer(layer0_points)
        # [B, 32, 4096]
        layer0_points = F.relu(
            self.bn(self.conv1d(layer0_points.permute(0, 2, 1))))
        # print(layer0_points.is_contiguous())
        # [B, 4096, num_classes]
        layer0_points = self.fc(layer0_points.permute(0, 2, 1))
        # print(layer0_points.is_contiguous())
        layer0_points = layer0_points.log_softmax(-1)

        return layer0_points


if __name__ == '__main__':
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # BATCH_SIZE = 64
    # modelnet_dataset = ModelNetDataset(root=BASE_DIR)
    # # s3dis_dataset = S3DISDataset(root=base)
    # point_feature = PointFeature(num_point=modelnet_dataset.num_points,
    #                              global_feature=False)
    # cls_net = PointNetCls(num_class=modelnet_dataset.num_classes,
    #                       num_point=modelnet_dataset.num_points)
    # seg_net = PointNetSeg(num_class=modelnet_dataset.num_classes,
    #                       num_point=modelnet_dataset.num_points)
    # segfc_net = PointNetSegFC(num_class=modelnet_dataset.num_classes,
    #                           num_point=modelnet_dataset.num_points)
    # modelnet_dataloader = DataLoader(modelnet_dataset, batch_size=BATCH_SIZE)
    # for (modelnet_pc, modelnet_label) in modelnet_dataloader:
    #     break

    # feature = point_feature(modelnet_pc)

    # seg_fc = segfc_net(modelnet_pc)

    # output = cls_net(modelnet_pc)
    # print(output.shape)
    # pred_num = output.argmax(dim=1)
    # for i in range(4):
    #     print(f'Item {i+1}: {modelnet_dataset.class_names[pred_num[i]]}')

    # print(pred_num.shape, modelnet_label.shape)

    # correct = (pred_num == modelnet_label.squeeze()).sum().item()
    # print(correct)

    # seg = seg_net(modelnet_pc)
    # print(seg.shape)

    points = torch.randn(4, 4096, 9)
    idx = farthest_point_sample(points[..., :3], 1024)
    # idx_points = points.gather(dim=-1, index=idx)
    print(idx.shape)
    # net = GAC_Net(num_classes=13)
    # output = net(points)
    # print(output.shape)
