import pandas as pd
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
import torch.nn as nn
import glob
import os
os.environ['CUDA_VISIBLE_DEVICE']='2,3'




def GENECG_RDDM(x_ppgs,nT=10,PATH="./checkpoints/", device="cuda:0",Sample_Length=512):

    num, Signal_Length = x_ppgs.shape

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
        if Signal_Length %Sample_Length !=0:
            # 需要补零的长度计算
            Zero_Length = Sample_Length - Signal_Length %Sample_Length
            x_ppgs = np.concatenate([x_ppgs[:, :Signal_Length], x_ppgs[:, -Zero_Length:]], axis=1)
        else:
            Zero_Length = 0

        x_ppgs = torch.tensor(x_ppgs).float().to(device)

        count = 0
        generated_ecg = []

        for ppg_window in torch.split(x_ppgs, Sample_Length, dim=-1):

            if ppg_window.shape[-1] != Sample_Length:
                ppg_window = F.pad(ppg_window, (0, Sample_Length - ppg_window.shape[-1]), "constant", 0)

            ppg_conditions1 = Conditioning_network1(ppg_window[:,None])
            ppg_conditions2 = Conditioning_network2(ppg_window[:,None])

            xh = dpm(
                cond1=ppg_conditions1,
                cond2=ppg_conditions2,
                mode="sample",
                window_size= 128 * 4
            )
            count += 1

            if generated_ecg.__len__() ==0:
                generated_ecg = (xh.cpu().numpy()[:,0])
            else:
                gen_ecg = xh.cpu().numpy()[:,0]
                generated_ecg = np.concatenate([generated_ecg,gen_ecg[:,:-Zero_Length]],axis=1)

    return generated_ecg

if __name__=='__main__':

    ppg_dir = "./data/"
    ppg_files = glob.glob(ppg_dir+"*.npy")
    gecg_dir = "./gen_ecg/"
    if not os.path.exists(gecg_dir):
        os.mkdir(gecg_dir)
    for ppg_file in ppg_files:
        ppg = np.load(ppg_file)
        #这里需要增加一个维度
        if len(ppg.shape) == 1:
            ppg = ppg[None,:]
        gen_ecg = GENECG_RDDM(ppg)
        data = pd.DataFrame({"ppg":ppg.tolist(),
                             "gen_ecg":gen_ecg.tolist()})
        gecg_path = gecg_dir + ppg_file.split('.')[1].split('/')[-1] + ".pkl"
        data.to_pickle(gecg_path)

    plt.subplot(211)
    plt.plot(ppg[0],label="PPG",color="blue")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.subplot(212)
    plt.plot(gen_ecg[0],label="RDDM_ECG",color="red")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()