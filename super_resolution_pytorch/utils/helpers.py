import torch
import numpy as np
import matplotlib.pyplot as plt

def prepare_batch(high, low, h=256, l=32, valid=False, side=-1, occupancy=False):
    ratio = h//l
    alter = (side==-1)
    if alter:
        side = np.random.randint(0, 6)
    side = np.ones((l,l,1)) * side
    low = low.reshape((l,l,1))
    high = high.reshape((h,h,1))
    if not occupancy:
        a,b,c = np.where(low>0)
        up = np.zeros((h,h,1))
        for x,y,z in zip(a,b,c):
            up[ratio*x:ratio*(x+1), ratio*y:ratio*(y+1), 0] = (low[x,y,0]) *ratio
        change = np.where(high==h)
        if valid:
            up_change = np.where(up==h)
        else:
            up_change = change
        up = up + 1
        up[up_change] = 0
        high = high + 1
        high[change] = 0
    else:
        on = np.where(high!=h)
        off = np.where(high==h)
        high[on] = 1.0
        high[off] = 0.0
        up = np.zeros((1,1,1))
    return high, low, up, side

def odm(data, high, low):
    dim = data.shape[0]
    down = high // low
    a,b,c = np.where(data == 1)
    large = int(dim * 1.5)
    big_list = [[[[-1,large]for j in range(dim)] for i in range(dim)] for k in range(3)]
    for i,j,k in zip(a,b,c):
        big_list[0][i][j][0] = (max(k,big_list[0][i][j][0]))
        big_list[0][i][j][1] = (min(k,big_list[0][i][j][1]))
        big_list[1][i][k][0] = (max(j,big_list[1][i][k][0]))
        big_list[1][i][k][1] = (min(j,big_list[1][i][k][1]))
        big_list[2][j][k][0] = (max(i,big_list[2][j][k][0]))
        big_list[2][j][k][1] = (min(i,big_list[2][j][k][1]))
    faces = np.zeros((6,dim,dim))
    for i in range(dim):
        for j in range(dim):
            faces[0,i,j] =   dim -1 - big_list[0][i][j][0]         if    big_list[0][i][j][0]   > -1 else dim
            faces[1,i,j] =   big_list[0][i][j][1]        		   if    big_list[0][i][j][1]   < large else dim
            faces[2,i,j] =   dim -1 - big_list[1][i][j][0]         if    big_list[1][i][j][0]   > -1 else dim
            faces[3,i,j] =   big_list[1][i][j][1]        		   if    big_list[1][i][j][1]   < large else dim
            faces[4,i,j] =   dim -1 - big_list[2][i][j][0]         if    big_list[2][i][j][0]   > -1 else dim
            faces[5,i,j] =   big_list[2][i][j][1]         		   if    big_list[2][i][j][1]   < large else dim
    return faces

def extract_prepare_odms(object, h, l):
    ratio = h//l
    low_odms = odm(object, h, l).reshape(-1,l,l,1) # low
    sides = []
    low_ups = []
    for i, low in enumerate(low_odms):
        sides.append(np.ones((l,l,1)) * i%6)
        a,b,c = np.where(low > 0)
        up = np.zeros((h,h,1))
        for x,y,z in zip(a,b,c):
            up[ratio*x:ratio*(x+1), ratio*y:ratio*(y+1), 0] = (low[x,y,0]) *ratio
        up_change = np.where(up==h)
        up = up + 1
        up[up_change] = 0
        low_ups.append(up)
    return low_odms, np.array(low_ups), np.array(sides)

def recover_depths(preds, ups, high, dis):
    preds = np.round_(preds*dis).reshape((-1,high,high))
    ups = np.array(ups).reshape((-1,high,high))
    for pred,up,i in zip(preds, ups, range(preds.shape[0])):
        pred = np.array(pred)
        pred  = up + pred
        off = np.where(pred > high)
        pred[off] = high-1
        preds[i] = pred
    return preds

def recover_occupancy(preds, high, threshold = 0.5):
    preds[np.where( preds >  threshold)] = 1.0
    preds[np.where( preds <= threshold)] = 0.0
    return preds.reshape((-1,high,high))

def fast_smoothing(odms, high, low, threshold):
    for i,odm in (enumerate(odms)):
        copy = np.array(odm)
        on = np.where(odm != high)
        for x,y in zip(*on):
            window = odm[x-3:x+4,y-3:y+4] #window
            considered = np.where(abs(window - odm[x,y])< threshold)
            copy[x,y] = np.average(window[considered])
        odms[i] = np.round_(copy)
    return odms

def recover_odms(depths, occs, ups, high, low, dis, threshold=20):
    odms = recover_depths(depths, ups, high, dis)
    occs = recover_occupancy(occs, high)
    off = np.where(occs == 0)
    odms = odms -1
    odms[off] = high
    odms = fast_smoothing(np.array(odms), high, low, threshold)
    return odms

def return_odms(prediction, occ_model, depth_model, device, h=256, l=32, dis=70, threshold=20):
    lows, low_ups, sides = extract_prepare_odms(np.array(prediction), h, l)
    low_res = torch.tensor(lows, dtype=torch.float32).permute((0,3,1,2)).to(device)
    low_ups = torch.tensor(low_ups, dtype=torch.float32).permute((0,3,1,2)).numpy()
    side = torch.tensor(sides, dtype=torch.float32).permute((0,3,1,2)).to(device)
    combined = torch.cat((low_res, side), dim=1)
    occ_model.eval()
    occ_preds = occ_model(combined).detach().cpu().numpy()
    depth_model.eval()
    depth_preds = depth_model(combined).detach().cpu().numpy()
    odms = recover_odms(depth_preds, occ_preds, low_ups, h, l, dis, threshold)
    return odms

def upsample(obj, high, low):
    ratio = high // low
    big_obj = np.zeros((high, high, high))
    for i in range(low):
        for j in range(low):
            for k in range(low):
                big_obj[i * ratio: (i + 1) * ratio, j * ratio:(j + 1) * ratio, k * ratio:(k + 1) * ratio] = obj[i, j, k]
    return big_obj

def apply_occupancy(obj, odms, high):
    unoccupied = np.where(odms == high)
    for x, y, z in zip(*unoccupied):
        if x == 0 or x == 1:
            obj[y, z, :] -= 0.25
        elif x == 2 or x == 3:
            obj[y, :, z] -= 0.25
        else:
            obj[:, y, z] -= 0.25
    ones = np.where(obj >= .6)
    zeros = np.where(obj < .6)
    obj[ones] = 1
    obj[zeros] = 0
    return obj

def apply_depth(obj, odms, high):
    for i in range(6):
        if i % 2 == 0:
            face = np.array(odms[i])
            on = np.where(face != high)
            face[on] = high - face[on]
            odms[i] = face
    prediction = np.array(obj)
    depths = np.where(odms <= high)
    for x, y, z in zip(*depths):
        pos = int(odms[x, y, z])
        if x == 0:
            prediction[y, z, pos:high] -= .25
        if x == 1:
            prediction[y, z, 0:pos] -= .25
        if x == 2:
            prediction[y, pos:high, z] -= .25
        elif x == 3:
            prediction[y, 0:pos, z] -= .25
        elif x == 4:
            prediction[pos:high, y, z] -= .25
        elif x == 5:
            prediction[0:pos, y, z] -= .25
    ones = np.where(prediction == 1)
    zeros = np.where(prediction < 1)
    prediction[ones] = 1
    prediction[zeros] = 0
    return prediction

def make_super_resolution(obj, odms, h=256, l=32):
    upsampled_obj = upsample(obj, h, l)
    occupancy_obj = apply_occupancy(upsampled_obj, odms, h)
    depth_obj = apply_depth(occupancy_obj, odms, h)
    return depth_obj

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