import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from PIL import Image, ImageOps  # type: ignore
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import LABEL_ORDER, get_data
from losses.wce import multitask_WCE
from model.lightning import LitResNet
from model.resnet import ResNet18MultiHead

import numpy as np
import pandas as pd
from metrics import uceis_argmax_accuracy

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = get_data(cfg)
    weights = [datasets[0].dataset.compute_class_weights(label=l) for l in LABEL_ORDER]

    num_classes = [len(w) for w in weights]

    model = ResNet18MultiHead(num_classes=num_classes).to(device)
    criterion = multitask_WCE(weights, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    lit_model = LitResNet(cfg, model, optimizer, criterion).to(device)
    lit_model = lit_model.load_from_checkpoint(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\src\outputs\2022-07-19\22-11-41\checkpoints\overlap-epoch=38.ckpt', model=model, loss_fn=criterion)  # FILL IN THE CHECKPOINT PATH
    #lit_model = lit_model.load_from_checkpoint(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-07-26\19-54-50\checkpoints\overlap-epoch=00.ckpt', model=model, loss_fn=criterion)  # FILL IN THE CHECKPOINT PATH
    lit_model = lit_model.load_from_checkpoint(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-08-04\12-53-14\checkpoints\overlap-epoch=08.ckpt', model=model, loss_fn=criterion)  # FILL IN THE CHECKPOINT PATH
    #wandb_logger: WandbLogger = hydra.utils.instantiate(cfg.pl_logging)

    
    device_name = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    trainer = pl.Trainer(
        #accelerator="auto",
        accelerator=device_name,
        max_epochs=cfg.model.epochs,
        log_every_n_steps=cfg.model.log_every_n_steps,
    #    logger=wandb_logger,
        devices='1'
    )
     
    train_loader = next(
        (ds.loader for ds in datasets if "train" in ds.split.name.lower())
    )
    val_loader = next(
        (ds.loader for ds in datasets if "validation" in ds.split.name.lower())
    )
    test_loader = next(
        (ds.loader for ds in datasets if "test" in ds.split.name.lower())
    )
    #out = trainer.test(lit_model, test_loader)

    y_true_UCEIS = []
    y_true_vascular = []
    y_true_bleeding = []
    y_true_erosion = []

    y_pred_UCEIS = []
    y_pred_vascular = []
    y_pred_bleeding = []
    y_pred_erosion = []

    with torch.no_grad():
        for id, (data, target) in enumerate(test_loader):
            target.transpose_(0, 1)

            y_true_vascular.append(target[0].item())
            y_true_bleeding.append(target[1].item())
            y_true_erosion.append(target[2].item())
            y_true_UCEIS.append(target.sum().item())

            output = lit_model(data)
            prediction_0 = output[0].argmax(dim=1, keepdim=True)[0][0].item()
            y_pred_vascular.append(prediction_0)

            prediction_1 = output[1].argmax(dim=1, keepdim=True)[0][0].item()
            y_pred_bleeding.append(prediction_1)

            prediction_2 = output[2].argmax(dim=1, keepdim=True)[0][0].item()
            y_pred_erosion.append(prediction_2)

            y_pred_UCEIS.append(prediction_0 + prediction_1 + prediction_2)
        
    pd.DataFrame(y_pred_UCEIS).to_csv("predicted_2022-08-04-12-53-14-checkpoints-overlap-epoch=08.csv")
    pd.DataFrame(y_true_UCEIS).to_csv("groundtruth_2022-08-04-12-53-14-checkpoints-overlap-epoch=08.csv")

def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')


def average_convolve(y_pred, slide=50):
    '''
    Each array is in shape (num_samples, 3) [[2,3,3], [0,1,2], ...]
    '''
    uceis_pred = running_mean_convolve(y_pred, slide)
    print(uceis_pred)
    return uceis_pred

def convolve_video_level(y_pred_UCEIS, y_true_UCEIS):
    video_level_pred = []
    video_level_conv = []
    video_level_true = []
    
    video_level_pred.append(y_pred_UCEIS[:371].max())
    video_level_pred.append(y_pred_UCEIS[371:530].max())
    video_level_pred.append(y_pred_UCEIS[530:1693].max())
    video_level_pred.append(y_pred_UCEIS[1693:1902].max())
    video_level_pred.append(y_pred_UCEIS[1902:4149].max())
    video_level_pred.append(y_pred_UCEIS[4149:4343].max())
    video_level_pred.append(y_pred_UCEIS[4343:4567].max())
    video_level_pred.append(y_pred_UCEIS[4567:4975].max())
    video_level_pred.append(y_pred_UCEIS[4975:5443].max())
    video_level_pred.append(y_pred_UCEIS[5443:5885].max())
    video_level_pred.append(y_pred_UCEIS[5885:6136].max())
    video_level_pred.append(y_pred_UCEIS[6136:6986].max())
    video_level_pred.append(y_pred_UCEIS[6986:7207].max())
    video_level_pred.append(y_pred_UCEIS[7207:7510].max())
    video_level_pred.append(y_pred_UCEIS[7510:7811].max())
    video_level_pred.append(y_pred_UCEIS[7811:7960].max())
    video_level_pred.append(y_pred_UCEIS[7960:8128].max())
    video_level_pred.append(y_pred_UCEIS[8128:].max())
    
    
    video_level_conv.append(average_convolve(y_pred_UCEIS[:371]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[371:530]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[530:1693]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[1693:1902]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[1902:4149]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[4149:4343]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[4343:4567]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[4567:4975]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[4975:5443]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[5443:5885]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[5885:6136]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[6136:6986]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[6986:7207]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[7207:7510]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[7510:7811]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[7811:7960]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[7960:8128]).max())
    video_level_conv.append(average_convolve(y_pred_UCEIS[8128:]).max())

    
    video_level_true.append(y_true_UCEIS[:371].max())
    video_level_true.append(y_true_UCEIS[371:530].max())
    video_level_true.append(y_true_UCEIS[530:1693].max())
    video_level_true.append(y_true_UCEIS[1693:1902].max())
    video_level_true.append(y_true_UCEIS[1902:4149].max())
    video_level_true.append(y_true_UCEIS[4149:4343].max())
    video_level_true.append(y_true_UCEIS[4343:4567].max())
    video_level_true.append(y_true_UCEIS[4567:4975].max())
    video_level_true.append(y_true_UCEIS[4975:5443].max())
    video_level_true.append(y_true_UCEIS[5443:5885].max())
    video_level_true.append(y_true_UCEIS[5885:6136].max())
    video_level_true.append(y_true_UCEIS[6136:6986].max())
    video_level_true.append(y_true_UCEIS[6986:7207].max())
    video_level_true.append(y_true_UCEIS[7207:7510].max())
    video_level_true.append(y_true_UCEIS[7510:7811].max())
    video_level_true.append(y_true_UCEIS[7811:7960].max())
    video_level_true.append(y_true_UCEIS[7960:8128].max())
    video_level_true.append(y_true_UCEIS[8128:].max())

    return video_level_pred, video_level_conv, video_level_true

def convolve_frame_level(y_pred_UCEIS, slide):
    ret_array = np.empty((len(y_pred_UCEIS),1), dtype='float16')
    #TODO:Change for different windows sizes. Suitale for window_size = 3
    # for i in range slide:
    ret_array[0] = (((y_pred_UCEIS[0]+y_pred_UCEIS[1])/2.) + ((y_pred_UCEIS[0]+y_pred_UCEIS[1]+y_pred_UCEIS[2])/3.))/2.
    ret_array[1] = (((y_pred_UCEIS[0]+y_pred_UCEIS[1])/2.) + ((y_pred_UCEIS[0]+y_pred_UCEIS[1]+y_pred_UCEIS[2])/3.) + ((y_pred_UCEIS[1]+y_pred_UCEIS[2]+y_pred_UCEIS[3])/3.))/3.

    for i in range (slide-1, len(y_pred_UCEIS)-slide+1):
        ret_array[i] = (((y_pred_UCEIS[i-2]+y_pred_UCEIS[i-1]+y_pred_UCEIS[i])/3.) + ((y_pred_UCEIS[i-1]+y_pred_UCEIS[i]+y_pred_UCEIS[i+1])/3.) + ((y_pred_UCEIS[i]+y_pred_UCEIS[i+1]+y_pred_UCEIS[i+2])/3.))/3.

    ret_array[len(y_pred_UCEIS)-2] = (((y_pred_UCEIS[len(y_pred_UCEIS)-1]+y_pred_UCEIS[len(y_pred_UCEIS)-2])/2.) + ((y_pred_UCEIS[len(y_pred_UCEIS)-1]+y_pred_UCEIS[len(y_pred_UCEIS)-2]+y_pred_UCEIS[len(y_pred_UCEIS)-3])/3.) + ((y_pred_UCEIS[len(y_pred_UCEIS)-2]+y_pred_UCEIS[len(y_pred_UCEIS)-3]+y_pred_UCEIS[len(y_pred_UCEIS)-4])/3.))/3.
    ret_array[len(y_pred_UCEIS)-1] = (((y_pred_UCEIS[len(y_pred_UCEIS)-1]+y_pred_UCEIS[len(y_pred_UCEIS)-2])/2.) + ((y_pred_UCEIS[len(y_pred_UCEIS)-1]+y_pred_UCEIS[len(y_pred_UCEIS)-2]+y_pred_UCEIS[len(y_pred_UCEIS)-3])/3.))/2.

    return ret_array
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    #main()
    #y_pred_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-07-21\13-30-23\predicted.csv',delimiter=',',dtype=None)
    #y_pred_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-07-26\12-29-44\predicted.csv',delimiter=',',dtype=None)
    y_pred_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-08-04\14-31-49\predicted_2022-08-04-12-53-14-checkpoints-overlap-epoch=08.csv',delimiter=',',dtype=None)
    y_pred_UCEIS = y_pred_UCEIS[1:,1]

    #y_true_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-07-21\13-30-23\groundtruth.csv',delimiter=',',dtype=None)
    #y_true_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-07-26\12-29-44\groundtruth.csv',delimiter=',',dtype=None)
    y_true_UCEIS = np.genfromtxt(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\surgease-experimantal-setup\outputs\2022-08-04\14-31-49\groundtruth_2022-08-04-12-53-14-checkpoints-overlap-epoch=08.csv',delimiter=',',dtype=None)
    y_true_UCEIS = y_true_UCEIS[1:,1]

    #my_arr = [1,2,3,2,3,1]
    #print(my_arr)
    #ret = convolve_frame_level(my_arr, 3)
    #print(ret)
    #video_level_pred, video_level_conv, video_level_true = convolve_video_level(y_pred_UCEIS, y_true_UCEIS)
    y_conv_UCEIS = convolve_frame_level(y_pred_UCEIS, 3)
    
    print('MSE before: ', np.square(np.subtract(y_true_UCEIS, y_pred_UCEIS)).mean())
    print('MSE after smoothing: ', np.square(np.subtract(y_true_UCEIS, y_conv_UCEIS)).mean())

    correct = (y_true_UCEIS == y_pred_UCEIS)
    accuracy = correct.sum() / correct.size
    print('Accuracy before: ', accuracy)
    correct = (y_true_UCEIS == np.round(y_conv_UCEIS))
    accuracy = correct.sum() / correct.size
    print('Accuracy after smoothing: ', accuracy)
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    #x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]  #for video. There is 18 video total in test set
    x = range(0, len(y_pred_UCEIS))
    plt.scatter(x, y_true_UCEIS, cmap="copper", c='green', alpha=0.3)
    plt.scatter(x, y_pred_UCEIS, cmap="copper", c='red', alpha=0.3)
    plt.scatter(x, y_conv_UCEIS, cmap="copper", c='blue', alpha=0.3)
    plt.show()
    a=0
