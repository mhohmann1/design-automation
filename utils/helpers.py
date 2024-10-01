import matplotlib.pyplot as plt
from utils.calc_mesh import compare_meshes
from pytorch3d.ops import cubify
from utils.loss_function import loss_fun
import torch
import binvox_rw
import numpy as np
from super_resolution_pytorch.utils.helpers import return_odms, make_super_resolution

def plot_loss(train_loss, test_loss, label, save_img=False, show_img=False, path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label=f"Training {label}")
    plt.plot(test_loss, label=f"Testing {label}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{label}")
    plt.legend(loc="upper right")
    if save_img:
        plt.savefig(path)
    if show_img:
        plt.show()
    plt.close()

def load_model(path, model, optimizer, eval=True):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Epoch: ", epoch)
    print("Loss: ", loss)
    if eval:
        model.eval()
        return model
    else:
        return model, optimizer

def calc_iou(prediction, gt, mode="batch"):
    IoU = 0
    if mode == "batch":
        for recon, voxel in zip(prediction, gt):
            intersection = np.sum(np.logical_and(recon, voxel))
            union = np.sum(np.logical_or(recon, voxel))
            IoU += float(intersection) / float(union)
        return IoU
    elif mode == "single":
        intersection = np.sum(np.logical_and(prediction, gt))
        union = np.sum(np.logical_or(prediction, gt))
        IoU += float(intersection) / float(union)
        return IoU

def compute_confusion_matrix(actual, predicted):
    actual = (actual.detach().cpu().numpy() >= 0.5)
    predicted = (predicted.detach().cpu().numpy() >= 0.5)
    TP = (predicted & actual).sum().item()
    TN = ((~predicted) & (~actual)).sum().item()
    FP = (predicted & (~actual)).sum().item()
    FN = ((~predicted) & actual).sum().item()
    return TP, TN, FP, FN


def eval_prediction(model, data_loader, device="cuda", input_type="voxel", condition_type="both", cat_nums=19, super_resolution=False, occupancy_model=None, depth_model=None, high=256, low=32, num_points=10_000):
    iou_low = 0.0
    iou_high = 0.0
    elbo_loss = 0.0
    bce_loss = 0.0
    kld_loss = 0.0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    chamfer_distance = 0.0
    f1_score = 0.0
    norm_consist = 0.0

    len_dataset = len(data_loader.dataset)

    for points, condition, voxel, path in data_loader:
        model.eval()
        with torch.no_grad():
            points, voxel, condition = points.to(device), voxel.to(device), condition.to(device)

            if input_type=="voxel":
                input = voxel

            elif input_type=="pointcloud":
                input = points

            else:
                print("Error")
                exit()

            if condition_type=="both":
                condition_num = condition[:, 0].reshape(-1, 1)
                condition_cat = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), cat_nums)
                condition = torch.cat((condition_num, condition_cat), dim=1)
                pred_voxel, mean, log_var, log_det, z = model(input, condition)

            elif condition_type=="numerical":
                condition = condition[:, 0].reshape(-1, 1)
                pred_voxel, mean, log_var, log_det, z = model(input, condition)

            elif condition_type=="categorical":
                condition = torch.nn.functional.one_hot(condition[:, -1].view(-1).long(), cat_nums)
                pred_voxel, mean, log_var, log_det, z = model(input, condition)

            elif condition_type is None:
                pred_voxel, mean, log_var, log_det, z = model(input)

            else:
                print("Error")
                exit()

            pred_voxel = (pred_voxel >= 0.5).type(torch.float32)
            elbo, bce, kld = loss_fun(pred_voxel, voxel, mean, log_var, log_det)
            elbo_loss += elbo
            bce_loss += bce
            kld_loss += kld
            iou_low += calc_iou(pred_voxel.detach().cpu().numpy(), voxel.detach().cpu().numpy())
            TP, TN, FP, FN = compute_confusion_matrix(voxel, pred_voxel)
            true_positive += TP
            false_positive += FP
            true_negative += TN
            false_negative += FN

            if super_resolution:
                for e, p in enumerate(path):
                    with open(p.replace("ShapeNet", "ShapeNetCoreSR") + "/model.binvox", "rb") as f:
                        gt_voxel = binvox_rw.read_as_3d_array(f).data.astype(np.uint8)

                    gt_mesh = cubify(torch.tensor(gt_voxel).unsqueeze(0), 0.5)
                    odms = return_odms(pred_voxel[e].cpu().detach().numpy(), occupancy_model, depth_model, device,
                                       h=high, l=low, dis=70, threshold=1.5*high//low)
                    sr_pred = make_super_resolution(pred_voxel[e].cpu().detach().numpy(), odms, h=high, l=low).astype(
                        np.uint8)
                    iou_high += calc_iou(sr_pred, gt_voxel, mode="single")
                    sr_mesh = cubify(torch.tensor(sr_pred).unsqueeze(0), 0.5)
                    mesh_metrics = compare_meshes(sr_mesh, gt_mesh, num_samples=num_points, reduce=False)
                    chamfer_distance += mesh_metrics["Chamfer-L2"]
                    f1_score += mesh_metrics["F1@0.100000"]
                    norm_consist += mesh_metrics["AbsNormalConsistency"]
                    iou_high += calc_iou(sr_pred, gt_voxel, mode="single")

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)

    print(30 * "-")
    print(f"{len_dataset} samples to evaluate")
    print(30 * "-")
    print("Evaluation for low resolution")
    print(30 * "-")
    print(f"ELBO: {elbo_loss / len_dataset:.6f}")
    print(f"BCE: {bce_loss / len_dataset:.6f}")
    print(f"KLD: {kld_loss / len_dataset:.6f}")
    print(f"IoU (mean): {iou_low / len_dataset:.6f}")
    print(f"TPR: {tpr * 100:.2f}%")
    print(f"FNR: {(1 - tpr) * 100:.2f}%")
    print(f"FPR: {fpr * 100:.2f}%")
    print(f"TNR: {(1 - fpr) * 100:.2f}%")
    print(30 * "-")
    if super_resolution:
        print("Evaluation for super resolution")
        print(30 * "-")
        print(f"IoU (mean): {iou_high / len_dataset:.6f}")
        print(f"CD: {chamfer_distance.item() / len_dataset:.4f}")
        print(f"F1: {f1_score.item() / len_dataset:.4f}")
        print(f"NC: {norm_consist.item() / len_dataset:.4f}")
        print(30 * "-")