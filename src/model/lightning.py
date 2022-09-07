import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from data import LABEL_ORDER
from metrics import (
    single_head_accuracy,
    uceis_0_vs_rest,
    uceis_03_vs_48,
    uceis_argmax_accuracy,
)


class LitResNet(pl.LightningModule):
    def __init__(self, cfg, model, optimizer, loss_fn, device="cpu") -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn", "model"])

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _compute_predictions(model_outputs):
        predictions = []
        for model_output in model_outputs:
            predictions.append(torch.max(model_output, dim=-1)[1])
        return predictions

    def _predict_with_loss(self, batch, store_predictions=False):
        # training_step defines the train loop.
        model_inputs, targets = batch

        model_inputs = model_inputs.to(self.device)
        targets = targets.to(self.device)

        model_outputs = self.forward(model_inputs.to(self.device))
        loss, losses = self.loss_fn(model_outputs, targets)

        metrics = {
            "loss": loss,
            **{
                f"loss_{k.value}": losses[LABEL_ORDER.index(k)]
                for k in LABEL_ORDER
            },
        }
        metrics.update(
            {
                f"acc_{k.value}": single_head_accuracy(
                    model_outputs[LABEL_ORDER.index(k)],
                    targets[:, LABEL_ORDER.index(k)],
                )
                for k in LABEL_ORDER
            }
        )
        metrics.update(
            {
                "uceis_argmax_accuracy": uceis_argmax_accuracy(
                    model_outputs, targets
                ),
            }
        )
        if store_predictions:
            predictions = self._compute_predictions(model_outputs)
            metrics.update(
                {
                    "predictions": {
                        k.value: predictions[LABEL_ORDER.index(k)]
                        for k in LABEL_ORDER
                    },
                    "targets": {
                        k.value: targets[:, LABEL_ORDER.index(k)]
                        for k in LABEL_ORDER
                    },
                }
            )

        return model_outputs, targets, loss, losses, metrics

    def training_step(self, batch, batch_idx):
        model_outputs, targets, loss, losses, metrics = self._predict_with_loss(
            batch,
        )
        for k, m in metrics.items():
            self.log(f"train.{k}", m)
        return loss

    def _evaluate_step(self, batch, batch_idx, prefix="validation"):
        model_outputs, targets, loss, losses, metrics = self._predict_with_loss(
            batch,
            store_predictions=True,
        )
        for k, m in metrics.items():
            self.log(f"{prefix}.{k}", m)
        return metrics

    def _evaluate_epoch_end(self, outputs, prefix="validation"):
        uceis_preds = 0
        uceis_target = 0
        for k in LABEL_ORDER:
            tmp = [tmp[f"predictions"][k.value] for tmp in outputs]
            #preds = torch.cat(tmp).detach().numpy()
            preds = torch.cat(tmp).cpu().numpy()
            tmp = [tmp[f"targets"][k.value] for tmp in outputs]
            #targets = torch.cat(tmp).detach().numpy()
            targets = torch.cat(tmp).cpu().numpy()
            uceis_preds += preds
            uceis_target += targets

            self.logger.experiment.log(
                {
                    f"{prefix}.{k.value}_conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets,
                        preds=preds,
                        class_names=list(
                            range(4)
                        ),  # Hack vascular only has 3 classes
                    )
                },
                step=self.global_step,
            )

        self.logger.experiment.log(
            {
                f"{prefix}.uceis_conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=uceis_target,
                    preds=uceis_preds,
                    class_names=np.arange(9),
                )
            },
            step=self.global_step,
        )

        self.log(f"{prefix}.uceis_0_vs_rest", uceis_0_vs_rest(preds, targets))
        self.log(f"{prefix}.uceis_03_vs_48", uceis_03_vs_48(preds, targets))

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._evaluate_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, prefix="test")

    def test_epoch_end(self, outputs):
        self._evaluate_epoch_end(outputs, prefix="test")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}
