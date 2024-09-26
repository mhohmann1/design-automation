from torch.utils.data import DataLoader, random_split
from utils.dataloader import *
from utils.loss_function import loss_fun, loss_cc3d
from model.cvae_flow import CC_CVAE_FLOW
import argparse
import torch
from utils import plot_loss
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np

def train(model, dataloader, optimizer, epoch, num_classes, input_type="voxel", add_cc3d=False):
    model.train()
    len_dataset = len(dataloader.dataset)
    tot_train_loss, tot_rec_loss, tot_lat_loss, tot_cc3d_loss = 0, 0, 0, 0
    for points, condition, voxel in dataloader:
        points = points.to(device)
        voxel = voxel.to(device)
        condition = condition.to(device)
        condition_num = condition[:, 0].reshape(-1, 1)
        condition_cat = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), num_classes)
        condition = torch.cat((condition_num, condition_cat), 1)
        if input_type == "pointcloud":
            x_pred, mean, log_var, log_det, z = model(points, condition)
        elif input_type == "voxel":
            x_pred, mean, log_var, log_det, z = model(voxel, condition)
        loss, rec_loss, lat_loss = loss_fun(x_pred, voxel, mean, log_var, log_det)
        tot_train_loss += loss.item()
        tot_rec_loss += rec_loss.item()
        tot_lat_loss += lat_loss.item()
        if add_cc3d:
            loss += loss_cc3d(x_pred, voxel) * points.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = (tot_train_loss + tot_cc3d_loss) / len_dataset
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train Loss: {average_loss:.6f}")
    return average_loss, tot_rec_loss / len_dataset, tot_lat_loss / len_dataset

def test(model, dataloader, epoch, num_classes, input_type="voxel", add_cc3d=False):
    model.eval()
    len_dataset = len(dataloader.dataset)
    with torch.no_grad():
        tot_test_loss, tot_rec_loss, tot_lat_loss, tot_cc3d_loss = 0, 0, 0, 0
        for points, condition, voxel in dataloader:
            points = points.to(device)
            voxel = voxel.to(device)
            condition = condition.to(device)
            condition_num = condition[:, 0].reshape(-1, 1)
            condition_cat = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), num_classes)
            condition = torch.cat((condition_num, condition_cat), 1)
            if input_type == "pointcloud":
                x_pred, mean, log_var, log_det, z = model(points, condition)
            elif input_type == "voxel":
                x_pred, mean, log_var, log_det, z = model(voxel, condition)
            loss, rec_loss, lat_loss = loss_fun(x_pred, voxel, mean, log_var, log_det)
            tot_test_loss += loss.item()
            tot_rec_loss += rec_loss.item()
            tot_lat_loss += lat_loss.item()
            if add_cc3d:
                loss += loss_cc3d(x_pred, voxel) * points.size(0)
    average_loss = (tot_test_loss + tot_cc3d_loss) / len_dataset
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch: {epoch}, Test Loss: {average_loss:.6f}")
    return average_loss, tot_rec_loss / len_dataset, tot_lat_loss / len_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Variational Autoencoder")
    parser.add_argument('-p', '--path', default="../ShapeNet/02958343", help="Path of dataset.", type=str)
    parser.add_argument('-e', '--epochs', default=100, help="Number of epochs.", type=int)
    parser.add_argument('-b', '--batch_size', default=32, help ="Number of batch size.", type=int)
    parser.add_argument('-z', "--latent_size", default=128, help="Latent Z dimension.", type=int)
    parser.add_argument('-w', "--workers", default=16, help="Number of workers.", type=int)
    parser.add_argument('-lr', "--learning_rate", default=1e-3, help="Size of learning rate.", type=int)
    parser.add_argument('-i', "--input_type", default="voxel", help="Input voxel or pointcloud.", type=str)
    parser.add_argument('-v', "--voxel_dim", default=(32, 32, 32), help="Dimensions of voxel.", type=tuple)
    parser.add_argument('-c', "--cond_dim", default=13, help="Conditional dimensions 19 cars and 12 planes.", type=int)
    parser.add_argument('-cn', "--cond_num_dim", default=1, help="Conditional dimensions for numericals.", type=int)
    parser.add_argument('-cp', "--cond_path", default="./car/cars.csv", help="CSV of classes.", type=str)
    parser.add_argument('-cnp', "--cond_num_path", default="./car/drag_coefficients.xlsx", help="Numerical conditions.", type=str)
    parser.add_argument('-cl', "--class_mode", default="car", help="Class mode car or plane.", type=str)
    parser.add_argument('-cc', '--cc3d', default=False, help="Add cc3d regularization.", type=bool)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = args.path
    dataset = Data(data_path, args.cond_path, args.cond_num_path, args.class_mode)

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size],
                                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
                             drop_last=True)

    print("Samples in Trainingset:", len(train_loader.dataset))
    print("Samples in Validationset:", len(valid_loader.dataset))
    print("Samples in Testingset:", len(test_loader.dataset))

    print("Latent size: ", args.latent_size)
    print("Voxel dimensions: ", args.voxel_dim)

    print("CC3D [TRUE/FALSE]: ", args.cc3d)

    vox_cvae = CC_CVAE_Flow(args.latent_size, args.voxel_dim, args.cond_dim + args.cond_num_dim).to(device)
    # print(vox_cvae)

    print("Input type:", args.input_type)
    print("Learning rate: ", args.learning_rate)
    optimizer = torch.optim.Adam(vox_cvae.parameters(), lr=args.learning_rate)

    rnd_num = np.random.randint(1_000_000)
    print(rnd_num)

    save_dir = f"./{rnd_num}/"
    os.mkdir(save_dir)
    os.mkdir(f"{save_dir}tnse/")

    scheduler = StepLR(optimizer, step_size=args.epochs // 4, gamma=0.1)

    train_hist, test_hist = [], []

    for epoch in range(1, args.epochs + 1):

        mean_loss, mean_bce, mean_latent = train(vox_cvae, train_loader, optimizer, epoch, args.cond_dim, args.input_type, args.cc3d)
        train_hist.append([mean_loss, mean_bce, mean_latent])
        mean_loss, mean_bce, mean_latent = test(vox_cvae, valid_loader, epoch, args.cond_dim, args.input_type, args.cc3d)
        test_hist.append([mean_loss, mean_bce, mean_latent])

        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            PATH = f"{rnd_num}_pcvox_cvae_e{epoch}.tar"
            LOSS = train_hist[-1:][0]

            torch.save({
                        "epoch": epoch,
                        "model_state_dict": vox_cvae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": LOSS,
                        }, save_dir + PATH)

            train_hist_ = np.array(train_hist)
            test_hist_ = np.array(test_hist)

            np.save(save_dir + f"{rnd_num}_train_hist.npy", train_hist_)
            np.save(save_dir + f"{rnd_num}_test_hist.npy", test_hist_)

            plot_loss(train_hist_[:, 0], test_hist_[:, 0], "ELBO", save_img=True, show_img=False, path=save_dir+f"{rnd_num}_elbo.png")
            plot_loss(train_hist_[:, 1], test_hist_[:, 1], "BCE", save_img=True, show_img=False, path=save_dir+f"{rnd_num}_bce.png")
            plot_loss(train_hist_[:, 2], test_hist_[:, 2], "KLD", save_img=True, show_img=False, path=save_dir+f"{rnd_num}_kld.png")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            save_latent_cvae(vox_cvae, train_loader, epoch, save_dir, args.input_type, args.cond_dim, device)

    print(f"Saved files in {rnd_num}")
