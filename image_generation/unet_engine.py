import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import os

def batch_train(
    model:torch.nn.Module,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    test_dataloader:DataLoader,
    device:torch.device,
    loss_fn:torch.nn,
    optimiser:torch.optim,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    epochs:int,
    divisor:int
):
    start = timer()
    patience = 20
    epochs_no_imp = 0
    overfit_counter = 0

    for epoch in range(1,epochs+1,1):
        epoch_start = timer()
        print(f"Epoch: {epoch} \n--------------------------------------------------------------------------------------- ")
        