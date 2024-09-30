import numpy as np
import scipy

def normalize_point_clouds(pc, mode):
    '''
    copied from https://github.com/luost26/diffusion-point-cloud/blob/0bfd688379e78ac75fa75e6a2c5029e362496169/test_gen.py#L16
    Args:
        pcs: list of [N,3] or tensor in shape: B,N,3
    '''
    if mode == "shape_unit":
        shift = pc.mean(axis=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
    elif mode == "shape_bbox":
        pc_max = pc.max(axis=0, keepdims=True)  # (1, 3)
        pc_min = pc.min(axis=0, keepdims=True)  # (1, 3)
        shift = ((pc_min + pc_max) / 2).reshape(1, 3)
        scale = ((pc_max - pc_min).max() / 2).reshape(1, 1)
    pc = (pc - shift) / scale
    return pc

def rotate_model(data, model_type="pointcloud", angle=None):
    if angle == None:
        angle = np.random.choice([0, 90, 180, 270], replace=True)
    if model_type == "pointcloud":
        theta = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
        data = data.dot(rotation_matrix)
    elif model_type == "voxel":
        data = scipy.ndimage.rotate(data, angle, axes=(2, 0), reshape=False)
    return data

def random_pc(data, num_points=2048):
    idx = np.random.choice(data.shape[0], num_points)
    data = data[idx, :]
    return data
