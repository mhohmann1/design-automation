import torch
import torch.nn as nn
import NF

class PointCloudEncoder(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(PointCloudEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024+c_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.BatchNorm1d(z_dim)
        )

        self.z_log_var = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.BatchNorm1d(z_dim)
        )

    def forward(self, x, c):
        x = self.encode(x)
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = torch.cat((x, c), dim=1)
        x = self.fc(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        return mean, log_var


class VoxelEncoder(nn.Module):
    def __init__(self, z_dim, voxel_dim, c_dim):
        super(VoxelEncoder, self).__init__()
        self.voxel_dim = voxel_dim
        self.enconv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        # self.flatten_size = self.calculate_flatten_size(self.voxel_dim)
        self.flatten_size = 21952

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size+c_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.z_mean = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.BatchNorm1d(z_dim)
        )

        self.z_log_var = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.BatchNorm1d(z_dim)
        )

    def calculate_flatten_size(self, voxel_dim):
        with torch.no_grad():
            return self.enconv(torch.zeros(1, 1, *voxel_dim)).view(-1).size(0)

    def forward(self, x, c):
        x = x.view(-1, 1, *self.voxel_dim)
        x = self.enconv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, c), dim=1)
        x = self.fc(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        return mean, log_var

class VoxelDecoder(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(VoxelDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim+c_dim, 64*7**3),
            nn.BatchNorm1d(64*7**3),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        z = torch.cat((z, c), dim=1)
        out = self.fc(z)
        out = out.view(-1, 64, 7, 7, 7)
        out = self.deconv(out)
        return out.squeeze(1)

class CC_CVAE_FLOW(nn.Module):
    def __init__(self, z_dim=128, voxel_dim=(32, 32, 32), c_dim=20):
        super(CC_CVAE_FLOW, self).__init__()
        self.z_dim = z_dim
        self.voxel_dim = voxel_dim
        self.c_dim = c_dim
        self.point_encoder = PointCloudEncoder(self.z_dim, self.c_dim)
        self.voxel_encoder = VoxelEncoder(self.z_dim, voxel_dim, self.c_dim)
        self.decoder = VoxelDecoder(self.z_dim, self.c_dim)

        # see 'https://github.com/AWehenkel/Normalizing-Flows'

        conditioner_type = NF.AutoregressiveConditioner
        conditioner_args = {"in_size": z_dim, "hidden": [z_dim] * 3, "out_size": z_dim}

        normalizer_type = NF.AffineNormalizer
        normalizer_args = {}

        nb_flow = 8

        self.flow = NF.buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type,
                                              normalizer_args)

    def transform(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        log_det, _ = self.flow.compute_ll(z0)
        return z0, log_det

    def sample(self, size, c):
        z = torch.randn(size, self.z_dim).cuda()
        z0 = self.flow.invert(z)
        return self.decoder(z0, c)

    def forward(self, x, c):
        if x.size(2) == 3:
            x = x.permute(0, 2, 1)
            mean, log_var = self.point_encoder(x, c)

        elif x.size(2) == self.voxel_dim[0]:
            mean, log_var = self.voxel_encoder(x, c)

        z, log_det = self.transform(mean, log_var)
        x_pred = self.decoder(z, c)
        return x_pred, mean, log_var, log_det, z
