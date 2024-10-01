import os
import numpy as np
import torch
from torch.utils.data import Dataset
from super_resolution_pytorch.utils.helpers import prepare_batch

class Data(Dataset):
    def __init__(self, obj_dir, occupancy=False):
        self.paths = []
        self.occupancy = occupancy

        def extract_number(filename):
            number = "".join(filter(str.isdigit, filename))
            return int(number) if number else 0

        objs = os.listdir(obj_dir)

        for obj in objs:
            if "." in obj:
                continue
            current_path = os.path.join(obj_dir, obj)
            files_in_path = sorted(os.listdir(current_path), key=extract_number)
            for file in files_in_path:
                if "low_odm" not in file:
                    continue
                path_append = os.path.join(current_path, file)
                self.paths.append(path_append)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        high_path = path.replace("low_odm", "odm")
        low_res = np.load(path).astype(np.int16)
        high_res = np.load(high_path).astype(np.int16)
        high, low, low_up, side = prepare_batch(high_res, low_res, h=256, l=32, occupancy=self.occupancy)
        return torch.tensor(high, dtype=torch.float32), torch.tensor(low, dtype=torch.float32), torch.tensor(low_up, dtype=torch.float32), torch.tensor(side, dtype=torch.float32)