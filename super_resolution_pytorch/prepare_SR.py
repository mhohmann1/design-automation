import binvox_rw
import numpy as np
import scipy
import os
import argparse
from super_resolution_pytorch.utils.helpers import odm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Super-Resolution-Network")
    parser.add_argument('-p', '--path', default="data/ShapeNetCorev1/", help="Path of dataset.", type=str)
    parser.add_argument('--high', default=256, help="High-Resolution of voxel grid.", type=int)
    parser.add_argument('--low', default=32, help="Low-Resolution of voxel grid.", type=int)
    args = parser.parse_args()

    np.random.seed(42)

    down = args.high // args.low

    for folder in os.listdir(args.path):
        class_folder = os.path.join(args.path, folder)
        for obj in os.listdir(class_folder):
            obj_path = os.path.join(class_folder, obj) + "/model.obj"

            command = "./binvox " + obj_path + " -d " + str(args.high) + " -pb -cb -c -dc -aw -e"
            os.system(command)

            model_path = obj_path.replace(".obj", ".binvox")

            with open(model_path, 'rb') as f:
                try:
                    model = binvox_rw.read_as_3d_array(f)
                except ValueError:
                    print("Error in reading")
            model = model.data.astype(np.uint8)

            try:
                os.makedirs(model_path.replace("ShapeNetCorev1", "ShapeNetCoreSR").replace("model.binvox", ""))
            except FileExistsError:
                print("Already there")
            os.replace(model_path, model_path.replace("ShapeNetCorev1", "ShapeNetCoreSR"))

            a, b, c = np.where(model == 1)
            low_model = np.zeros((args.low, args.low, args.low)).astype(np.uint8)
            for x, y, z in zip(a, b, c):
                low_model[x // down, y // down, z // down] = 1
            low_model[scipy.ndimage.binary_fill_holes(low_model)] = 1

            np.save(obj_path.replace("ShapeNetCorev1", "ShapeNetCoreSR").replace("model.obj", "low_model.npy"), low_model)

            faces = odm(model, args.high, args.low)
            low_faces = odm(low_model, args.high, args.low)

            for i in range(6):
                np.save(obj_path.replace("ShapeNetCorev1", "ShapeNetCoreSR").replace("model.obj",f"odm_{i}.npy"), faces[i].astype(np.uint16))
                np.save(obj_path.replace("ShapeNetCorev1", "ShapeNetCoreSR").replace("model.obj",f"low_odm_{i}.npy"), low_faces[i].astype(np.uint8))
