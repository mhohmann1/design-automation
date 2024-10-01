import numpy as np
import torch
import argparse
from super_resolution_pytorch.utils.dataloader import  Data
from super_resolution_pytorch.utils.helpers import plot_loss
from torch.utils.data import DataLoader, random_split
from super_resolution_pytorch.model.upscale_model import Upscale, train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Super-Resolution-Network")
    parser.add_argument('-p', '--path', default="data/ShapeNetSR/02958343", help="Path of dataset.", type=str)
    parser.add_argument('-e', '--epochs', default=100, help="Number of epochs.", type=int)
    parser.add_argument('-b', '--batch_size', default=32, help ="Number of batch size.", type=int)
    parser.add_argument('-w', "--workers", default=0, help="Number of workers.", type=int)
    parser.add_argument('-lr', "--learning_rate", default=1e-4, help="Size of learning rate.", type=int)
    parser.add_argument('-m', '--model_type', default="depth", help="Enter 'depth' (default) or 'occupancy'.", type=str)
    parser.add_argument('--high', default=256, help="High-Resolution of voxel grid.", type=int)
    parser.add_argument('--low', default=32, help="Low-Resolution of voxel grid.", type=int)
    parser.add_argument('-cm', "--class_mode", default="car", help="Class mode car or plane.", type=str)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.model_type == "depth":
        dataset = Data(args.parse,  False)
    elif args.model_type == "occupancy":
        dataset = Data(args.parse, True)
    else:
        print("Error")
        exit()

    print("Samples in Dataset:", len(dataset))

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset,[train_size, valid_size, test_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    print("Samples in Trainingset:", len(train_loader.dataset))
    print("Samples in Validationset:", len(valid_loader.dataset))
    print("Samples in Testingset:", len(test_loader.dataset))

    ratio = args.high//args.low

    model = Upscale(ratio).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    BEST_LOSS = np.inf

    PR_PATH = "super_resolution_pytorch/saved_model/"

    train_hist, test_hist = [], []

    for epoch in range(1, args.epochs + 1):
        if args.model_type=="depth":
            train_loss = train(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device, depth_loss=True)
            test_loss, test_output = test(model=model, dataloader=valid_loader, criterion=criterion, device=device, depth_loss=True)
        elif args.model_type=="occupancy":
            train_loss = train(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
            test_loss, test_output = test(model=model, dataloader=valid_loader, criterion=criterion, device=device)
        else:
            print("Error")
            exit()

        train_hist.append(train_loss)
        test_hist.append(test_loss)

        if test_loss < BEST_LOSS:
            F_PATH = f"{PR_PATH}{args.model_type}_model_{args.class_mode}.tar"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }, F_PATH)

            BEST_LOSS = test_loss

        print(f"Epoch: {epoch}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {test_loss:.6f}, Best Loss: {BEST_LOSS:.6f}")

        train_hist_ = np.array(train_hist)
        test_hist_ = np.array(test_hist)

        np.save(f"{PR_PATH}{args.model_type}_train_hist.npy", train_hist_)
        np.save(f"{PR_PATH}{args.model_type}_test_hist.npy", test_hist_)

        plot_loss(train_hist_[:], test_hist_[:], "MSE", save_img=True, show_img=False, path=f"{PR_PATH}{args.model_type}_loss.png")