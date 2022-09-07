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
from losses.wce import multitask_WCE, multitask_MSE_ordinal
from model.lightning import LitResNet
from model.resnet import ResNet18MultiHead

import pandas as pd

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = get_data(cfg)
    weights = [
        datasets[0].dataset.compute_class_weights(label=l) for l in LABEL_ORDER
    ]

    num_classes = [len(w) for w in weights]

    model = ResNet18MultiHead(num_classes=num_classes).to(device)
    #criterion = multitask_WCE(weights, device=device)
    criterion = multitask_MSE_ordinal(weights, device=device)
    #optimizer = torch.optim.Adam(
    #    model.parameters(),
    #    lr=cfg.model.learning_rate,
    #    weight_decay=cfg.model.weight_decay,
    #)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    lit_model = LitResNet(cfg, model, optimizer, criterion).to(device)
    wandb_logger: WandbLogger = hydra.utils.instantiate(cfg.pl_logging)

    # saves checkpoints to 'outputs/[day]/[time]/checkpoints' at every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        monitor="validation.loss",
        mode="min",
        filename="%s-{epoch:02d}" % cfg.dataset.name,
        save_top_k=3,
    )

    device_name = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                

    trainer = pl.Trainer(
        accelerator=device_name,
        max_epochs=cfg.model.epochs,
        log_every_n_steps=cfg.model.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
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
    
    trainer.fit(lit_model, train_loader, val_loader)
    
    for checkpoint in Path("checkpoints").iterdir():
        artifact = wandb.Artifact(
            checkpoint.name.replace("=", "_"), type="model"
        )
        artifact.add_file(checkpoint)
        wandb.run.log_artifact(artifact)
    
    torch.save(lit_model.state_dict(), 'saved_model')
    out = trainer.test(lit_model, test_loader)
    
    '''
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

            for i in range(4): #batch size
                new_target = target[i]

                #new_target.transpose_(0, 1)

                y_true_vascular.append(new_target[0].item())
                y_true_bleeding.append(new_target[1].item())
                y_true_erosion.append(new_target[2].item())
                y_true_UCEIS.append(new_target.sum().item())

                output = lit_model(data)
                prediction_0 = output[0].argmax(dim=1, keepdim=True)[0][0].item()
                y_pred_vascular.append(prediction_0)

                prediction_1 = output[1].argmax(dim=1, keepdim=True)[0][0].item()
                y_pred_bleeding.append(prediction_1)

                prediction_2 = output[2].argmax(dim=1, keepdim=True)[0][0].item()
                y_pred_erosion.append(prediction_2)

                y_pred_UCEIS.append(prediction_0 + prediction_1 + prediction_2)
        
    pd.DataFrame(y_pred_UCEIS).to_csv("predicted.csv")
    pd.DataFrame(y_true_UCEIS).to_csv("groundtruth.csv")
    '''


if __name__ == "__main__":
    main()
