from torch.utils.data import DataLoader, random_split
from utils.dataloader import Data
from utils.loss_function import loss_fun, loss_cc3d
import argparse
from model.cvae_flow import CC_CVAE_FLOW
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.helpers import load_model
from super_resolution_pytorch.model.upscale_model import Upscale
from utils.helpers import eval_prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="(C) Variational Autoencoder")
    parser.add_argument('-p', '--path', default="data/ShapeNet/02958343", help="Path of dataset.", type=str)
    parser.add_argument('-b', '--batch_size', default=32, help ="Number of batch size.", type=int)
    parser.add_argument('-z', "--latent_size", default=128, help="Number of latent Z size.", type=int)
    parser.add_argument('-w', "--workers", default=0, help="Number of workers.", type=int)
    parser.add_argument('-lr', "--learning_rate", default=1e-3, help="Size of learning rate.", type=int)
    parser.add_argument('-m', "--model_id", default=121958, help="ID of saved model (see ./saved_model).", type=int)
    parser.add_argument('-r', "--ratio", default=256//32, help="Ratio (high // low).", type=int)
    parser.add_argument('-i', "--input_type", default="voxel", help="Input voxel or pointcloud.", type=str)
    parser.add_argument('-v', "--voxel_dim", default=(32, 32, 32), help="Dimensions of voxel.", type=tuple)
    parser.add_argument('-cc', "--cond_cat_dim", default=19, help="Number of categories/classes (single categories -> cars=19 and planes=12).", type=int)
    parser.add_argument('-cn', "--cond_num_dim", default=1, help="Number of numerical conditions.", type=int)
    parser.add_argument('-ccp', "--cond_cat_path", default="data/car/cars.csv", help="File-Path of categorical conditions.", type=str)
    parser.add_argument('-cnp', "--cond_num_path", default="data/car/drag_coefficients.xlsx", help="File-Path of numerical conditions.", type=str)
    parser.add_argument('-ct', "--cond_type", default="both", help="Condition type(s), enter: categorical, numerical or both (default).", type=str)
    parser.add_argument('-cm', "--class_mode", default="car", help="Class mode car or plane.", type=str)
    parser.add_argument('-sr', '--super_res', default=True, help="Enable super-resolution.", type=bool)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = Data(args.path, args.cond_cat_path, args.cond_num_path, args.class_mode, augmentation=False)

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)


    occ_model = Upscale(args.ratio).to(device)
    optimizer = torch.optim.Adam(occ_model.parameters())
    path = f"super_resolution_pytorch/saved_model/occupancy_model_{args.class_mode}.tar"

    occ_model = load_model(path, occ_model, optimizer)
    occ_model.eval()
    print("Occupancy model loaded")

    depth_model = Upscale(args.ratio).to(device)
    optimizer = torch.optim.Adam(depth_model.parameters())
    path = f"super_resolution_pytorch/saved_model/depth_model_{args.class_mode}.tar"

    depth_model = load_model(path, depth_model, optimizer)
    depth_model.eval()
    print("Depth model loaded")


    if args.cond_type == "both":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_cat_dim + args.cond_num_dim).to(device)
    elif args.cond_type == "categorical":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_cat_dim).to(device)
    elif args.cond_type == "numerical":
        model = CC_CVAE_FLOW(args.latent_size, args.voxel_dim, args.cond_num_dim).to(device)
    else:
        print("Enter 'both' (default), 'categorical' or 'numerical'!")
        exit()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model = load_model(f"saved_model/{args.model_id}/{args.model_id}__cc_cvae_flow.tar", model, optimizer)

    eval_prediction(model, test_loader, device, input_type=args.input_type, condition_type=args.cond_type, cat_nums=args.cond_cat_dim, super_resolution=args.super_res, occupancy_model=occ_model,
                    depth_model=depth_model)
