import torch
import torch.nn as nn
from torch_points_kernels import knn


class MLP(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 kernel_size=1,
                 stride=1,
                 bn=False,
                 activation_fn=None):
        super(MLP, self).__init__()

        self.conv = nn.Conv2d(in_channels=d_in,
                              out_channels=d_out,
                              kernel_size=kernel_size,
                              stride=stride)
        self.bn = nn.BatchNorm2d(d_out) if bn else None
        self.activation_fn = activation_fn

    def forward(self, x):
        """
        Map input from d_in to d_out dimensions by 1x1 kernels with
        optional Batch Normalization and Activation Function.

        Input:
            x: [B, d_in, N, K]
        Output:
            x: [B, d_out, N, K]

        """

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d_out):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = MLP(d_in=3 + 3 + 3 + 1,
                       d_out=d_out,
                       bn=True,
                       activation_fn=nn.ReLU())

    def forward(self, feat, xyz, knn_output):
        '''
        Augment point feature by encoding relative point position (xyz and knn_output) 
        and then concantenating with the corresponding point feature (feat).

        Input:
            feat: [B, d_in, N, 1]  # d_in == d_out
            xyz: [B, N, 3]
            knn_output: tuple("idx:[B, N, K]", "dist2:[B, N, K]")
        Output:
            neighbouring_feat: [B, 2*d_out, N, K]

        '''

        idx, dist = knn_output  # [B, N, K]
        B, N, K = idx.size()

        extended_idx = idx.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, N, K]
        extended_xyz = xyz.transpose(-2, -1).unsqueeze(-1).repeat(
            1, 1, 1, K)  # [B, 3, N, K]
        neighbour = extended_xyz.gather(dim=2,
                                        index=extended_idx)  # [B, 3, N, K]
        concat_xyz = torch.cat(
            (extended_xyz, neighbour, extended_xyz - neighbour,
             dist.unsqueeze(1)),
            dim=1)  # [B, 10, N, K]
        relative_pnt_pos_enc = self.mlp(concat_xyz)  # [B, d_out, N, K]
        output = torch.cat((relative_pnt_pos_enc, feat.repeat(1, 1, 1, K)),
                           dim=1)  # [B, 2*d_out, N, K]

        return output


class AttentivePooling(nn.Module):
    def __init__(self, d_in, d_out):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(nn.Linear(d_in, d_in),
                                      nn.Softmax(dim=-2))

        self.mlp = MLP(d_in=d_in,
                       d_out=d_out,
                       bn=True,
                       activation_fn=nn.ReLU())

    def forward(self, feat):
        '''
        Aggregate feature by weighted summation.

        Input:
            feat: [B, d_in, N, K]
        Output:
            agg_feat: [B, d_out, N, 1]
        '''

        scores = self.score_fn(feat.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()  # [B, N, K, d_in] -> [B, d_in, N, K]
        feat = torch.sum(scores * feat, dim=-1,
                         keepdim=True)  # [B, d_in, N, 1]
        agg_feat = self.mlp(feat)  # [B, d_out, N, 1]

        return agg_feat


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbours):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbours = num_neighbours

        self.mlp_in = MLP(d_in=d_in,
                          d_out=d_out // 2,
                          bn=True,
                          activation_fn=nn.ReLU())
        self.mlp_out = MLP(d_in=d_out,
                           d_out=2 * d_out,
                           bn=True,
                           activation_fn=nn.ReLU())
        self.mlp_shortcut = MLP(d_in=d_in,
                                d_out=2 * d_out,
                                bn=True,
                                activation_fn=nn.ReLU())

        self.lose1 = LocalSpatialEncoding(d_out=d_out // 2)
        self.lose2 = LocalSpatialEncoding(d_out=d_out // 2)

        self.pool1 = AttentivePooling(d_in=d_out, d_out=d_out // 2)
        self.pool2 = AttentivePooling(d_in=d_out, d_out=d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, feat, xyz):
        '''
        Stack LoSE and AttPooling units with a skip connection.

        Input:
            feat: [B, d_in, N, 1]
            xyz: [B, N, 3]
        Output:
            aggregated_feat: [B, 2*d_out, N, 1]
        '''

        knn_output = knn(xyz.contiguous(), xyz.contiguous(),
                         self.num_neighbours)  # [B, N, K]

        residual = self.mlp_shortcut(feat)  # [B, 2*d_out, N, 1]

        feat1 = self.mlp_in(feat)  # [B, d_out//2, N, 1]
        lose_feat1 = self.lose1(feat1, xyz, knn_output)  # [B, d_out, N, K]
        att_feat1 = self.pool1(lose_feat1)  # [B, d_out//2, N, 1]

        lose_feat2 = self.lose2(att_feat1, xyz, knn_output)  # [B, d_out, N, K]
        att_feat2 = self.pool2(lose_feat2)  # [B, d_out, N, 1]
        feat_out = self.mlp_out(att_feat2)  # [B, 2*d_out, N, 1]

        return self.lrelu(feat_out + residual)


class RandLANet(nn.Module):
    def __init__(self,
                 d_in,
                 num_classes,
                 num_neighbours=16,
                 decimation=4,
                 dropout=0.5):
        super(RandLANet, self).__init__()

        self.num_neighbours = num_neighbours
        self.decimation = decimation
        self.fc_start = nn.Linear(in_features=d_in, out_features=8)

        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(d_in=8,
                                    d_out=16,
                                    num_neighbours=num_neighbours),
            LocalFeatureAggregation(2 * 16, 64, num_neighbours),
            LocalFeatureAggregation(2 * 64, 128, num_neighbours),
            LocalFeatureAggregation(2 * 128, 256, num_neighbours)
        ])

        self.mlp = MLP(d_in=512, d_out=512, bn=True, activation_fn=nn.ReLU())

        self.decoder = nn.ModuleList([
            MLP(d_in=512, d_out=256, bn=True, activation_fn=nn.ReLU()),
            MLP(d_in=2 * 256, d_out=128, bn=True, activation_fn=nn.ReLU()),
            MLP(d_in=2 * 128, d_out=32, bn=True, activation_fn=nn.ReLU()),
            MLP(d_in=2 * 32, d_out=8, bn=True, activation_fn=nn.ReLU())
        ])

        self.fc_final = nn.Sequential(
            MLP(d_in=2 * 8, d_out=64, bn=True, activation_fn=nn.ReLU()),
            MLP(d_in=64, d_out=32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(p=dropout), MLP(d_in=32, d_out=num_classes))

    def forward(self, points):
        '''
        Input:
            points: [B, N, 3+d_in]
        Output:
            class scores: [B, num_classes, N]
        '''

        B, N, _ = points.size()
        d = self.decimation  # 4
        decimation_ratio = 1

        xyz = points[..., :3]  # [B, N, 3]

        feat_stack = []

        feat = self.fc_start(points).transpose(-2, -1).unsqueeze(
            -1)  # [B, 8, N, 1]

        feat_rs = feat
        xyz_rs = xyz

        for lfa in self.encoder:

            feat_stack.append(feat_rs)
            feat_rs = lfa(feat_rs, xyz_rs)  # [B, 2*d_out, N//decimation_ratio, 1]
            decimation_ratio *= d
            feat_rs = feat_rs[..., :N//decimation_ratio, :]
            xyz_rs = xyz_rs[..., :N//decimation_ratio, :]
            # [B, 512, N//256, 1]

        feat = self.mlp(feat_rs)  # [B, 512, N//256, 1]

        for mlp in self.decoder:

            feat = mlp(feat)  # [B, 512, N//256, 1]

            # find one nearset neighbour for each upsampled point in the
            # downsampled set
            idx, _ = knn(xyz[..., :N // decimation_ratio, :].contiguous(),
                         xyz[..., :d * N // decimation_ratio, :].contiguous(),
                         1)  # [B, d*N//decimation, 1]
            extended_idx = idx.unsqueeze(1).repeat(
                1, feat.size(1), 1, 1)  # [B, d_out, d*N//decimation_ratio, 1]
            # print(f"extended_idx: {extended_idx.shape}")

            feat_neighbour = torch.gather(
                feat, -2, extended_idx)  # [B, d_out, d*N//decimation_ratio, 1]
            # print(f"feat_neighbour: {feat_neighbour.shape}")

            feat_pop = feat_stack.pop()  # [B, d_out, d*N//decimation_ratio, 1]
            # print(f"feat_pop: {feat_pop.shape}")

            feat = torch.cat((feat_neighbour, feat_pop), dim=1)
            # feat = mlp(feat)

            decimation_ratio //= d

        scores = self.fc_final(feat)  # [B, num_classes, N, 1]

        return scores.squeeze(-1)


if __name__ == '__main__':
    # fake data
    torch.manual_seed(0)

    BATCH = 8
    NUM_POINT = 2**10
    D_XYZ = 3
    D_IN = 4
    NUM_NEIGHBOUR = 16

    pc = torch.rand(BATCH, NUM_POINT, D_XYZ + D_IN)
    print(f"pc: {pc.shape}")
    pc_xyz = pc[:, :, :3]
    print(f"pc_xyz: {pc_xyz.shape}")
    pc_feat = pc[:, :, 3:]
    print(f"pc_feat: {pc_feat.shape}")

    model = RandLANet(d_in=D_XYZ + D_IN, num_classes=40, num_neighbours=4)
    class_scores = model(pc)
    print(f"class scores: {class_scores.shape}")
    class_label = class_scores.transpose(-2, -1).max(-1)
    print(f"class label: {class_label[0].shape}")

    import time

    NUM_SAMPLE = NUM_POINT // 4
    t1 = time.time()
    sampled_pc = pc[:, :NUM_SAMPLE, :]
    t2 = time.time()
    print(f'sampling time: {t2 - t1}\nsampled pc: {sampled_pc.shape}')
