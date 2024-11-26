import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data import get_datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os
os.environ['CUDA_VISIBLE_DEVICE']='0,1,2'


def eval_diffusion(window_size, EVAL_DATASETS, nT=10, batch_size=512,
                   PATH="./checkpoints/", device="cuda"):
    _, dataset_test = get_datasets(datasets=EVAL_DATASETS, window_size=window_size)

    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=64)



    with torch.no_grad():

        fd_list = []
        fake_ecgs = np.zeros((1, 128 * window_size))
        real_ecgs = np.zeros((1, 128 * window_size))
        real_ppgs = np.zeros((1, 128 * window_size))
        true_rois = np.zeros((1, 128 * window_size))

        for y_ecg, x_ppg, ecg_roi in tqdm(testloader):

            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            generated_windows = []

            for ppg_window in torch.split(x_ppg, 128 * 4, dim=-1):

                if ppg_window.shape[-1] != 128 * 4:
                    ppg_window = F.pad(ppg_window, (0, 128 * 4 - ppg_window.shape[-1]), "constant", 0)

                ppg_conditions1 = Conditioning_network1(ppg_window)
                ppg_conditions2 = Conditioning_network2(ppg_window)

                xh = dpm(
                    cond1=ppg_conditions1,
                    cond2=ppg_conditions2,
                    mode="sample",
                    window_size=128 * 4
                )

                generated_windows.append(xh.cpu().numpy())

            xh = np.concatenate(generated_windows, axis=-1)[:, :, :128 * window_size]

            fd = calculate_FD(y_ecg, torch.from_numpy(xh).to(device))

            fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128 * window_size)))
            real_ecgs = np.concatenate((real_ecgs, y_ecg.reshape(-1, 128 * window_size).cpu().numpy()))
            real_ppgs = np.concatenate((real_ppgs, x_ppg.reshape(-1, 128 * window_size).cpu().numpy()))
            true_rois = np.concatenate((true_rois, ecg_roi.reshape(-1, 128 * window_size).cpu().numpy()))
            fd_list.append(fd)

        mae_hr_ecg, rmse_score = evaluation_pipeline(real_ecgs[1:], fake_ecgs[1:])

        tracked_metrics = {
            "RMSE_score": rmse_score,
            "MAE_HR_ECG": mae_hr_ecg,
            "FD": sum(fd_list) / len(fd_list),
        }

        return tracked_metrics

def GENECG_RDDM(data,nT=10,PATH="./checkpoints/", device="cuda:0",Sample_Length=512):
    x_ppgs = data["ppg_seg"]
    x_ppgs = np.array([[num for num in sublist] for sublist in x_ppgs])
    num,Signal_Length=x_ppgs.shape

    dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
        PATH=PATH,
        nT=nT,
        type="RDDM",
        device="cuda"
    )

    dpm = nn.DataParallel(dpm)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    dpm.eval()
    Conditioning_network1.eval()
    Conditioning_network2.eval()
    with torch.no_grad():
        Zero_Length =Sample_Length- Signal_Length %Sample_Length
        x_ppgs = np.concatenate([x_ppgs[:, :Signal_Length], x_ppgs[:, -Zero_Length:]], axis=1)
        Num_sample=x_ppgs.shape[1]//Sample_Length
        x_ppgs = torch.tensor(x_ppgs).float().to(device)
        count=0
        generated_ecg=[]

        for ppg_window in torch.split(x_ppgs, Sample_Length, dim=-1):

            if ppg_window.shape[-1] != Sample_Length:
                ppg_window = F.pad(ppg_window, (0, Sample_Length - ppg_window.shape[-1]), "constant", 0)

            ppg_conditions1 = Conditioning_network1(ppg_window[:,None])
            ppg_conditions2 = Conditioning_network2(ppg_window[:,None])

            xh = dpm(
                cond1=ppg_conditions1,
                cond2=ppg_conditions2,
                mode="sample",
                window_size=128 * 4
            )
            count += 1

            if generated_ecg.__len__() ==0:
                generated_ecg=(xh.cpu().numpy()[:,0])
            else:
                gen_ecg=xh.cpu().numpy()[:,0]
                if count==Num_sample:
                    generated_ecg=np.concatenate([generated_ecg,gen_ecg[:,:-Zero_Length]],axis=1)
                else:
                    generated_ecg=np.concatenate([generated_ecg,gen_ecg],axis=1)
    return generated_ecg

if __name__=='__main__':
    dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
        PATH="./checkpoints/",
        nT=10,
        type="RDDM",
        device="cuda"
    )
    keep_same=0

    ppg_dir = "/data/chenjiarong/vitaldb/"
    gecg_dir = "/data/chenjiarong/vitaldb_genecg_RDDM/"
    if not os.path.exists(gecg_dir):
        os.mkdir(gecg_dir)
    if keep_same:
        pkl_files = glob.glob(ppg_dir + "*.pkl")
        for path in pkl_files:
            print(path)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            generated_ecg=GENECG_RDDM(data)
            datanew = data.assign(gen_ecg=generated_ecg.tolist())

            newpath=gecg_dir+path.split('.')[0].split('/')[-1]+"&gen_ecg.pkl"
            with open(newpath, 'wb') as file:
                pickle.dump(datanew,file)
    else:
    # 做个校正的过程，原始代码的β感觉不对
        pkl_files = glob.glob(gecg_dir + "*.pkl")
        for path in pkl_files:
            print(path)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            generated_ecg=GENECG_RDDM(data)
            datanew = data.assign(gen_ecg_newbeta=generated_ecg.tolist())
            with open(path, 'wb') as file:
                pickle.dump(datanew,file)