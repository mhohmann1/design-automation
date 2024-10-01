from torch import nn
import torch
import torch.nn.functional as F
from math import log2

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, 1, "same")
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, 1, "same")
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, upscale):
        recall = F.relu(self.bn1(self.conv1(upscale)))
        recall = self.bn2(self.conv2(recall))
        recall = upscale + recall
        return recall

class SubPixelBlock(nn.Module):
    def __init__(self, factor, scaling_factor):
        super(SubPixelBlock, self).__init__()
        self.conv = nn.Conv2d(int(128 / pow(2,factor)), int(256 / pow(2, factor)), 3, 1, "same")
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)

    def forward(self, upscale):
        output = self.conv(upscale)
        output = F.relu(self.pixel_shuffle(output))
        return output

class Upscale(nn.Module):
    def __init__(self, ratio):
        super(Upscale, self).__init__()
        self.initial_conv = nn.Conv2d(2, 128, 3, 1, "same")
        self.residual_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(16)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, "same"),
            nn.BatchNorm2d(128)
        )
        self.upsample_layers = nn.Sequential(*[SubPixelBlock(i, 2) for i in range(int(log2(ratio)))])
        self.final_conv = nn.Conv2d(16, 1, 1, 1, "same")

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        temp = x
        x = self.residual_blocks(x)
        x = self.mid_conv(x)
        x = x + temp
        x = self.upsample_layers(x)
        x = torch.sigmoid(self.final_conv(x))
        return x

def total_variation(img):
    tv_h = torch.sum(torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]), dim=(1, 2, 3))
    tv_w = torch.sum(torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]), dim=(1, 2, 3))
    return torch.mean(tv_h + tv_w)

def train(model, dataloader, criterion, optimizer, device, depth_loss=False, distance=70):
    model.train()
    train_loss = 0.0
    for high_res, low_res, low_up, side in dataloader:
        low_res = low_res.permute((0,3,1,2)).to(device)
        high_res = high_res.permute((0,3,1,2)).to(device)
        low_up = low_up.permute((0,3,1,2)).to(device)
        side = side.permute((0,3,1,2)).to(device)
        combined = torch.cat((low_res, side), dim=1)
        preds = model(combined)
        if depth_loss:
            preds = low_up + preds*distance
        loss = criterion(preds, high_res)
        if depth_loss:
            loss += 0.001 * total_variation(preds/256.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * low_res.size(0)
    average_loss = train_loss / len(dataloader.dataset)
    return average_loss

def test(model, dataloader, criterion, device, depth_loss=False, distance=70):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for high_res, low_res, low_up, side in dataloader:
            low_res = low_res.permute((0, 3, 1, 2)).to(device)
            high_res = high_res.permute((0, 3, 1, 2)).to(device)
            low_up = low_up.permute((0, 3, 1, 2)).to(device)
            side = side.permute((0, 3, 1, 2)).to(device)
            combined = torch.cat((low_res, side), dim=1)
            preds = model(combined)
            if depth_loss:
                preds = low_up + preds*distance
            loss = criterion(preds, high_res)
            if depth_loss:
                loss += 0.001 * total_variation(preds/256.)
            test_loss += loss.item() * low_res.size(0)
        average_loss = test_loss / len(dataloader.dataset)
        return average_loss, preds




