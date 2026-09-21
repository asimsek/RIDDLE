import os
import numpy as np
import torch
from . import flows as fnn
from .integrity import flow_loss, train_flow_epoch


def train_ANODE(
    model,
    optimizer,
    dataloader_train,
    dataloader_test,
    model_file_name,
    epochs,
    savedir="ANODE_models/",
    device=torch.device("cpu"),
    verbose=True,
    no_logit=False,
    data_std=None,
):
    os.makedirs(savedir, exist_ok=True)
    if no_logit and data_std is None:
        raise ValueError("Need data_std to correct losses when trained without logit")
    train_loss_return = compute_loss_over_batches(
        model, dataloader_train, device, correct_logit=data_std if no_logit else None
    )
    val_loss_return = compute_loss_over_batches(
        model, dataloader_test, device, correct_logit=data_std if no_logit else None
    )
    train_losses = np.full(epochs + 1, 1e20, dtype=np.float32)
    val_losses = np.full(epochs + 1, 1e20, dtype=np.float32)
    train_losses[0], val_losses[0] = train_loss_return[0], val_loss_return[0]
    for epoch in range(epochs):
        train_loss_return = train_epoch(
            model,
            optimizer,
            dataloader_train,
            device,
            verbose=verbose,
            data_std=data_std if no_logit else None,
        )
        val_loss_return = compute_loss_over_batches(
            model, dataloader_test, device, correct_logit=data_std if no_logit else None
        )
        train_losses[epoch + 1], val_losses[epoch + 1] = train_loss_return[0], val_loss_return[0]
        print(f"Epoch {epoch + 1}: train_loss={train_loss_return[0]}; val_loss={val_loss_return[0]}")


def train_epoch(model, optimizer, data_loader, device, verbose=True, data_std=None):
    return train_flow_epoch(
        model,
        optimizer,
        data_loader,
        device,
        batch_norm_class=fnn.BatchNormFlow,
        verbose=verbose,
        data_std=data_std,
    )


def compute_loss_over_batches(model, data_loader, device, correct_logit=None):
    return flow_loss(model, data_loader, device, correct_logit)
