import os
import numpy as np
from torch.utils.data import Dataset
import scipy
import torch
import binvox_rw
import pandas as pd
from utils_dataloader import *


class Data(Dataset):
    def __init__(self, obj_dir, cond_cat, cond_num, class_mode="car", cond_scale="normalized", augmentation=True,
                 num_points=2048):
        self.paths = []
        self.conditions = []
        self.augmentation = augmentation
        self.num_points = num_points
        self.cond_scale = cond_scale
        self.class_mode = class_mode

        if self.class_mode == "plane":
            cond_num = pd.read_csv(cond_num)
            class_dict = {
                "seaplane": 0,
                "biplane": 1,
                "delta wing": 2,
                "propeller plane": 3,
                "straight wing": 4,
                "bomber": 5,
                "swept wing": 6,
                "transport airplane": 7,
                "fighter": 8,
                "jet": 9,
                "airliner": 10,
                "airplane": 11
            }

        elif self.class_mode == "car":
            cond_num = pd.read_excel(cond_num)
            class_dict = {
                "pace car": 0,
                "cab": 1,
                "hatchback": 2,
                "minivan": 3,
                "hot rod": 4,
                "touring car": 5,
                "limousine": 6,
                "ambulance": 7,
                "roadster": 8,
                "jeep": 9,
                "beach wagon": 10,
                "car": 11,
                "sport utility": 12,
                "racer": 13,
                "cruiser": 14,
                "sports car": 15,
                "sedan": 16,
                "coupe": 17,
                "convertible": 18
            }

        else:
            raise Exception("Not available")

        cond_cat = pd.read_csv(cond_cat)

        objs = os.listdir(obj_dir)
        for obj in objs:
            if "." in obj:
                continue
            multi_cat = cond_cat.loc[cond_cat["fullId"] == "3dw." + obj]["wnlemmas"]
            if len(multi_cat) < 1:
                continue
            if self.class_mode == "car":
                condition = cond_num[cond_num["file"] == obj]["Cd"].to_numpy()
            elif self.class_mode == "plane":
                condition = cond_num[cond_num["Airplane ID"] == obj]["Drag Coefficient"].to_numpy()
            if len(condition) == 0:
                continue
            multi_cat = multi_cat.to_list()[0].split(",")
            arr_cat = []
            for single_cat in multi_cat:
                try:
                    arr_cat.append(class_dict[single_cat])
                except:
                    continue
            try:
                condition = np.append(condition, arr_cat[0])
            except:
                continue
            self.paths.append(os.path.join(obj_dir, obj))
            self.conditions.append(condition)

        remove_conditions = True

        if remove_conditions:
            c_pos = np.mean(self.conditions, axis=0) + 2 * np.std(self.conditions, axis=0)
            c_neg = np.mean(self.conditions, axis=0) - 2 * np.std(self.conditions, axis=0)
            indices_to_remove = []
            for index, condi in enumerate(self.conditions):
                if np.greater(condi, c_pos).any() or np.less(condi, c_neg).any():
                    indices_to_remove.append(index)

            for index in sorted(indices_to_remove, reverse=True):
                del self.conditions[index]
                del self.paths[index]

        if self.cond_scale == "normalized":
            self.cond_mean = np.array([0.40091659060638885, 0.48605318793918634])
            self.cond_std = np.array([0.06054546843690462, 0.27105443641593674])

            self.conditions = np.array(self.conditions)

            if self.class_mode == "car":
                self.conditions[:, 0] = (self.conditions[:, 0] - self.cond_mean[0]) / self.cond_std[0]

            elif self.class_mode == "plane":
                self.conditions[:, 0] = (self.conditions[:, 0] - self.cond_mean[1]) / self.cond_std[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        condition = self.conditions[idx]
        path = self.paths[idx]
        pc = np.load(path + "/pointcloud.npz")["points"]
        pc = random_pc(pc, self.num_points)
        pc = normalize_point_clouds(pc, "shape_bbox")
        with open(path + "/model.binvox", "rb") as f:
            voxel = binvox_rw.read_as_3d_array(f).data.astype(np.uint8)
        if self.augmentation:
            angle = np.random.choice([0, 90, 180, 270], replace=True)
            voxel = rotate_model(voxel, "voxel", angle)
            pc = rotate_model(pc, "pointcloud", angle)
        return torch.tensor(pc, dtype=torch.float32), torch.tensor(condition, dtype=torch.float32), torch.tensor(voxel,
                                                                                                                 dtype=torch.float32)

