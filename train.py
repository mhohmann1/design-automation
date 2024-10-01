from torch.utils.data import DataLoader, random_split
from utils.dataloader import Data
from utils.loss_function import loss_fun, loss_cc3d
import argparse
import torch
from utils.helpers import plot_loss
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from model.cvae_flow import CC_CVAE_FLOW
from tqdm import tqdm

def train(model, dataloader, optimizer, c_type, num_classes, input_type, add_cc3d):
    model.train()
    len_dataset = len(dataloader.dataset)
    tot_train_loss, tot_rec_loss, tot_lat_loss, tot_cc3d_loss = 0, 0, 0, 0
    for points, condition, voxel, _ in dataloader:
        points = points.to(device)
        voxel = voxel.to(device)
        condition = condition.to(device)
        if c_type == "categorical":
            condition = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), num_classes)
        elif c_type == "numerical":
            condition = condition[:, 0].reshape(-1, 1)
        else:
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
    return average_loss, tot_rec_loss / len_dataset, tot_lat_loss / len_dataset

def test(model, dataloader, c_type, num_classes, input_type, add_cc3d):
    model.eval()
    len_dataset = len(dataloader.dataset)
    with torch.no_grad():
        tot_test_loss, tot_rec_loss, tot_lat_loss, tot_cc3d_loss = 0, 0, 0, 0
        for points, condition, voxel, _ in dataloader:
            points = points.to(device)
            voxel = voxel.to(device)
            condition = condition.to(device)
            if c_type == "categorical":
                condition = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), num_classes)
            elif c_type == "numerical":
                condition = condition[:, 0].reshape(-1, 1)
            else:
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
    return average_loss, tot_rec_loss / len_dataset, tot_lat_loss / len_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conditional Variational Autoencoder")
    parser.add_argument('-p', '--path', default="data/ShapeNet/02958343", help="Path of dataset.", type=str)
    parser.add_argument('-e', '--epochs', default=100, help="Number of epochs.", type=int)
    parser.add_argument('-b', '--batch_size', default=32, help ="Number of batch size.", type=int)
    parser.add_argument('-z', "--latent_size", default=128, help="Number of latent Z size.", type=int)
    parser.add_argument('-w', "--workers", default=0, help="Number of workers.", type=int)
    parser.add_argument('-lr', "--learning_rate", default=1e-3, help="Size of learning rate.", type=int)
    parser.add_argument('-i', "--input_type", default="voxel", help="Input voxel or pointcloud.", type=str)
    parser.add_argument('-v', "--voxel_dim", default=(32, 32, 32), help="Dimensions of voxel.", type=tuple)
    parser.add_argument('-cc', "--cond_cat_dim", default=19, help="Number of categories/classes (single categories -> cars=19 and planes=12).", type=int)
    parser.add_argument('-cn', "--cond_num_dim", default=1, help="Number of numerical conditions.", type=int)
    parser.add_argument('-ccp', "--cond_cat_path", default="data/car/cars.csv", help="File-Path of categorical conditions.", type=str)
    parser.add_argument('-cnp', "--cond_num_path", default="data/car/drag_coefficients.xlsx", help="File-Path of numerical conditions.", type=str)
    parser.add_argument('-cm', "--class_mode", default="car", help="Class mode car or plane.", type=str)
    parser.add_argument('-ct', "--cond_type", default="both", help="Condition type(s), enter: categorical, numerical or both (default).", type=str)
    parser.add_argument('-cc3d', '--cc3d', default=True, help="Add cc3d regularization.", type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = Data(args.path, args.cond_cat_path, args.cond_num_path, args.class_mode)

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    print("Samples in Trainingset:", len(train_loader.dataset))
    print("Samples in Validationset:", len(valid_loader.dataset))
    print("Samples in Testingset:", len(test_loader.dataset))

    print("Latent size: ", args.latent_size)
    print("Voxel dimensions: ", args.voxel_dim)

    print("CC3D [TRUE/FALSE]: ", args.cc3d)

    if args.cond_type == "both":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_cat_dim + args.cond_num_dim).to(device)
    elif args.cond_type == "categorical":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_cat_dim).to(device)
    elif args.cond_type == "numerical":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_num_dim).to(device)
    else:
        print("Enter 'both' (default), 'categorical' or 'numerical'!")
        exit()

    print("Input type:", args.input_type)
    print("Learning rate: ", args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.epochs // 2, gamma=0.5)

    RND_NUM = np.random.randint(1_000_000)
    print(RND_NUM)

    SAVED_DIR = f"./saved_model/{RND_NUM}/"
    try:
        os.mkdir(SAVED_DIR)
    except FileExistsError:
        pass

    BEST_LOSS = np.inf

    train_hist, test_hist = [], []

    for epoch in tqdm(range(1, args.epochs + 1)):

        mean_train_loss, mean_train_bce, mean_train_latent = train(model, train_loader, optimizer, args.cond_type, args.cond_cat_dim, args.input_type, args.cc3d)
        train_hist.append([mean_train_loss, mean_train_bce, mean_train_latent])
        mean_test_loss, mean_test_bce, mean_test_latent = test(model, valid_loader, args.cond_type, args.cond_cat_dim, args.input_type, args.cc3d)
        test_hist.append([mean_test_loss, mean_test_bce, mean_test_latent])

        if mean_test_loss < BEST_LOSS:
            F_PATH = f"{SAVED_DIR}{RND_NUM}_cc_cvae_flow.tar"

            torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": mean_test_loss,
                        }, F_PATH)

            BEST_LOSS = mean_test_loss

        print(f"Epoch: {epoch}/{args.epochs}, Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_test_loss:.6f}, Best Loss: {BEST_LOSS:.6f}")

        train_hist_ = np.array(train_hist)
        test_hist_ = np.array(test_hist)

        np.save(SAVED_DIR + f"{RND_NUM}_train_hist.npy", train_hist_)
        np.save(SAVED_DIR + f"{RND_NUM}_test_hist.npy", test_hist_)

        plot_loss(train_hist_[:, 0], test_hist_[:, 0], "ELBO", save_img=True, show_img=False, path=SAVED_DIR+f"{RND_NUM}_elbo.png")
        plot_loss(train_hist_[:, 1], test_hist_[:, 1], "BCE", save_img=True, show_img=False, path=SAVED_DIR+f"{RND_NUM}_bce.png")
        plot_loss(train_hist_[:, 2], test_hist_[:, 2], "KLD", save_img=True, show_img=False, path=SAVED_DIR+f"{RND_NUM}_kld.png")

        scheduler.step()

    print(f"Saved Files in {RND_NUM}")

