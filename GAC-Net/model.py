import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # 2*(xn * xm + yn * ym + zn * zm)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # xn*xn + yn*yn + zn*zn
    dist += torch.sum(src**2, -1).view(B, N, 1)
    # xm*xm + ym*ym + zm*zm
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    # [B, nsample]
    view_shape = list(idx.shape)
    # [B, 1]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    # [1, nsample]
    repeat_shape[0] = 1
    # [B, nsample]
    batch_indices = torch.arange(
        B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # [B, npoint]
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # [B, N]
    distance = torch.ones(B, N).to(device) * 1e10
    # randomly select one point ranging from 0 to (N-1) in each batch
    # as the intial centroid
    # [B]
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    # [B]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # update the i-th farthest point
        # [B, npoint]
        centroids[:, i] = farthest
        # take the xyz of the farthest point
        # [B, 1, C]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # compute the Euclidean distance from all N points
        # to the farthest point
        # [B, N]
        dist = torch.sum((xyz - centroid)**2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        # update distances matrix
        # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点的indices，作为最远点用于下一轮迭代
        # [B]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, npoint, C]
    Return:
        group_idx: grouped points index, [B, npoint, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # [B, npoint, N]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(
        1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, npoint, N] 记录中心点与所有点之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx[sqrdists > radius**2] = N
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    # [B, npoint, nsample]
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），
    # 这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, npoint, nsample]， 实际就是把group_idx中的第一个点的值复制为了
    # [B, npoint, nsample]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点
    # [B, npoint, nsample]
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
        grouped_xyz: sampled and neighbouring points position data,
                        [B, npoint, nsample, C]
        fps_points: sampled points data, [B, npoint, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原点云中挑出最远点采样的采样点为new_xyz
    # [B, npoint] centroids indices
    centered_idx = farthest_point_sample(xyz, npoint)
    # [B, npoint, C]
    centered_xyz = index_points(xyz, centered_idx)
    # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    # [B, npoint, nsample] group points indices
    idx = query_ball_point(radius, nsample, xyz, centered_xyz)
    # grouped_xyz:[B, npoint, nsample, C]
    grouped_xyz = index_points(xyz, idx)
    # grouped_xyz减去采样点即中心值
    # [B, npoint, nsmaple, C]
    grouped_xyz_norm = grouped_xyz - centered_xyz.view(B, S, 1, C)
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        # [B, npoint, nsample, D]
        grouped_points = index_points(points, idx)
        # [B, npoint, D]
        centered_points = index_points(points, centered_idx)
        # [B, npoint, C+D]
        centered_points = torch.cat([centered_xyz, centered_points], dim=-1)
        # [B, npoint, nsample, C+D]
        grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        # [B, npoint, nsmaple, C]
        grouped_points = grouped_xyz_norm
        # [B, npoint, C]
        centered_points = centered_xyz
    if returnfps:
        return centered_xyz, grouped_points, grouped_xyz, centered_points
    else:
        return centered_xyz, grouped_points


# take all points as a group
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        centered_xyz: sampled points position data, [B, 1, C]
        grouped_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centered_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        grouped_points = torch.cat(
            [grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        grouped_points = grouped_xyz
    return centered_xyz, grouped_points


# def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     Input:
#         npoint: Number of point for FPS
#         radius: Radius of ball query
#         nsample: Number of point for each ball query
#         xyz: input points position data, [B, N, C]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, C]
#         new_points: sampled points data, [B, npoint, nsample, C+D]
#         grouped_xyz: sampled and neighbouring points position data,
#                       [B, npoint, nsample, C]
#         fps_points: sampled points data, [B, npoint, C+D]
#     """
#     B, N, C = xyz.shape
#     S = npoint
#     # 从原点云中挑出最远点采样的采样点为new_xyz
#     centered_idx = farthest_point_sample(xyz,
#                                     npoint)  # [B, npoint] centroids indices
#     centered_xyz = index_points(xyz, centered_idx)  # [B, npoint, C]
#     # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
#     idx = query_ball_point(
#         radius, nsample, xyz,
#         centered_xyz)  # [B, npoint, nsample] group points indices
#     # grouped_xyz:[B, npoint, nsample, C]
#     grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
#     # grouped_xyz减去采样点即中心值
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1,
#                                                   C)
#     # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
#     if points is not None:
#         grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
#         fps_points = index_points(points, fps_idx)  # [B, npoint, D]
#         fps_points = torch.cat([new_xyz, fps_points],
#                                dim=-1)  # [B, npoint, C+D]
#         new_points = torch.cat([grouped_xyz_norm, grouped_points],
#                                dim=-1)  # [B, npoint, nsample, C+D]
#     else:
#         new_points = grouped_xyz_norm  # [B, npoint, nsmaple, C]
#         fps_points = new_xyz  # [B, npoint, C]
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_points
#     else:
#         return new_xyz, new_points

# # take all points as a group
# def sample_and_group_all(xyz, points):
#     """
#     Input:
#         xyz: input points position data, [B, N, C]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, 1, C]
#         new_points: sampled points data, [B, 1, N, C+D]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     new_xyz = torch.zeros(B, 1, C).to(device)
#     grouped_xyz = xyz.view(B, 1, N, C)
#     if points is not None:
#         new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)],
#                                   dim=-1)
#     else:
#         new_points = grouped_xyz
#     return new_xyz, new_points


class GraphAttention(nn.Module):
    def __init__(self, all_channel, feature_dim, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        # [C+D, D]
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz,
                grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        # [B, npoint, nsample, C]
        delta_p = center_xyz.view(B, npoint, 1, C).expand(
            B, npoint, nsample, C) - grouped_xyz
        # [B, npoint, nsample, D]
        delta_h = center_feature.view(B, npoint, 1, D).expand(
            B, npoint, nsample, D) - grouped_feature
        # [B, npoint, nsample, C+D]
        delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)
        # [B, npoint, nsample, D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a))
        # [B, npoint, nsample, D]
        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # [B, npoint, D]
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)
        return graph_pooling


class GraphAttentionConvLayer(nn.Module):
    def __init__(self,
                 npoint,
                 radius,
                 nsample,
                 in_channel,
                 mlp,
                 group_all,
                 droupout=0.6,
                 alpha=0.2):
        '''
        Input:
                npoint: Number of point for FPS sampling
                radius: Radius for ball query
                nsample: Number of point for each ball query
                in_channel: the dimention of channel
                mlp: A list for mlp input-output channel, such as [64, 64, 128]
                group_all: bool type for group_all or not
        '''
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.GAT = GraphAttention(3 + last_channel, last_channel,
                                  self.droupout, self.alpha)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, npoint]
            new_points_concat: sample points feature data, [B, D', npoint]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            centered_xyz, grouped_points = sample_and_group_all(xyz, points)
        else:
            centered_xyz, grouped_points, grouped_xyz, centered_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, True)
        # centered_xyz: sampled points position data, [B, npoint, C]
        # grouped_points: sampled points data, [B, npoint, nsample, C+D]
        # grouped_xyz:[B, npoint, nsample, C]
        # centered_points: [B, npoint, C+D]
        # [B, C+D, nsample,npoint]
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        # [B, C+D, 1, npoint]
        centered_points = centered_points.unsqueeze(3).permute(0, 2, 3, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            centered_points = F.relu(bn(conv(centered_points)))
            grouped_points = F.relu(bn(conv(grouped_points)))
        # grouped_points: [B, F, nsample, npoint]
        # centered_points: [B, F, 1,npoint]
        new_points = self.GAT(
            center_xyz=centered_xyz,
            center_feature=centered_points.squeeze().permute(0, 2, 1),
            grouped_xyz=grouped_xyz,
            grouped_feature=grouped_points.permute(0, 3, 2, 1))
        new_xyz = centered_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            # [B, N, 3]
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dists[dists < 1e-10] = 1e-10
            # [B, N, 3]
            weight = 1.0 / dists
            # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)
            interpolated_points = torch.sum(index_points(points2, idx) *
                                            weight.view(B, N, 3, 1),
                                            dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class GACNet(nn.Module):
    def __init__(self, num_classes, droupout=0, alpha=0.2):
        super(GACNet, self).__init__()
        # GraphAttentionConvLayer: npoint, radius, nsample, in_channel, mlp,
        # group_all, droupout, alpha
        self.sa1 = GraphAttentionConvLayer(1024, 0.1, 32, 6 + 3, [32, 32, 64],
                                           False, droupout, alpha)
        self.sa2 = GraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 64, 128],
                                           False, droupout, alpha)
        self.sa3 = GraphAttentionConvLayer(64, 0.4, 32, 128 + 3,
                                           [128, 128, 256], False, droupout,
                                           alpha)
        self.sa4 = GraphAttentionConvLayer(16, 0.8, 32, 256 + 3,
                                           [256, 256, 512], False, droupout,
                                           alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(droupout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, point):
        l1_xyz, l1_points = self.sa1(xyz, point)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8, 3, 2048))
    model = GACNet(50)
    output = model(input, point)
