import os
import json
import random
import warnings
import numpy as np
import setproctitle
from tqdm import tqdm

import torch
import torch.nn as nn
import torchnet as tnt

from src import utils, model_utils 
from src.dataset import UtaeDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.learning.weight_init import weight_init

warnings.filterwarnings("ignore")
setproctitle.setproctitle("miras_utae_training_process")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def set_seed(seed=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]
    
def checkpoint(log, fold_dir, epoch=None, best=False, test=None):
    if best:
        log = {epoch: log}
        best_path = os.path.join(fold_dir, "train_log_best.json")
        if os.path.exists(best_path):
            os.remove(best_path)
        with open(best_path, "w") as outfile:
            json.dump(log, outfile, indent=4)
    else: 
        if test:
            json_name = "test_log.json"
        else:
            json_name = "train_log.json"
        with open(os.path.join(fold_dir, json_name), "w") as outfile:
            json.dump(log, outfile, indent=4)

def iterate(model,
            data_loader,
            criterion,
            config,
            optimizer=None,
            mode="train",
            device=None):
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.AverageValueMeter()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        if mode != "train":
            with torch.inference_mode():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)
        pred = out.squeeze(dim=1)
        loss = criterion(pred, y.float())
        if mode == "train":
            loss.backward()
            optimizer.step()

        tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        loss_meter.add(loss.item())
        iou_meter.add(iou.item())
        acc_meter.add(acc.item())
        if (i + 1) % config.display_step == 0:
            print(
                "Step [{}/{}], Loss: {:.4f}, mIoU : {:.2f}, Acc {:.2f}".format(
                    i + 1, len(data_loader), 
                    loss_meter.value()[0], 
                    iou_meter.value()[0], 
                    acc_meter.value()[0]
                )
            )
    return loss_meter.value()[0], iou_meter.value()[0], acc_meter.value()[0]

def train_loop(config):
    set_seed()
    fold_sequence = [
        [[1, 2, 3, 4], [5]],
        [[2, 3, 4, 5], [1]],
        [[3, 4, 5, 1], [2]],
        [[4, 5, 1, 2], [3]],
        [[5, 1, 2, 3], [4]],
    ]
    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold) in enumerate(fold_sequence):
        fold_string = f"[FOLD {train_folds[0]}]"
        print(f"{fold_string} Training folds {train_folds}, Validation set {val_fold}")
        device = torch.device(config.device)
        fold_dir = os.path.join(config.res_dir, config.exp_name, f"fold_{train_folds[0]}")
        os.makedirs(fold_dir, exist_ok=True)

        dt_train = UtaeDataset(folder=config.dataset_folder,
                               norm=True,
                               reference_date=config.ref_date,
                               folds=train_folds)

        dt_val = UtaeDataset(folder=config.dataset_folder,
                             norm=True,
                             reference_date=config.ref_date,
                             folds=val_fold)

        collate_fn = lambda x: utils.pad_collate(x, config.pad_value)
        train_loader = DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        print(f"{fold_string} Dataset sizes: Train {len(dt_train)}, Validation {len(dt_val)}")

        model = model_utils.get_model(config)
        # model = nn.DataParallel(model, device_ids=[0, 1])  # Specify GPUs if needed
        model = model.to(device)
        model.apply(weight_init)
        # model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        # weights = torch.ones(config.num_classes, device=device).float()
        # weights[config.ignore_index] = 0
        # criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=3 * config.epochs // 4,
                                                               eta_min=1e-4)
        train_log = {}
        best_mIoU = 0
        print(f"{fold_string} Training. . .")
        pbar = tqdm(range(1, config.epochs + 1), total=config.epochs)
        for epoch in pbar:
            model.train()
            train_loss, train_iou, train_acc = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            model.eval()
            val_loss, val_iou, val_acc = iterate(
                model,
                data_loader=val_loader,
                criterion=criterion,
                config=config,
                optimizer=None,
                mode="val",
                device=device,
            )
            if epoch < 3 * config.epochs // 4:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix(lr=f"{lr:0.6f}", gpu_mem=f"{gpu_mem:0.2f} GB")

            print(
                f"{fold_string} "
                f"Epoch: {epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_iou: {train_iou:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_iou: {val_iou:.4f} | "
                f"val_acc: {val_acc:.4f} | "
                )

            train_log[epoch] = {
                "train_loss": train_loss,
                "train_iou": train_iou,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_acc": val_acc,
                "lr": lr,
                }

            if val_iou > best_mIoU:
                print(f"{fold_string} Validation mIoU Score Improved ({best_mIoU:0.4f} ---> {val_iou:0.4f})")
                best_mIoU = val_iou
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(fold_dir, "model.pth.tar"))
                checkpoint(log=train_log[epoch], fold_dir=fold_dir, epoch=epoch, best=True)
            checkpoint(log=train_log, fold_dir=fold_dir)
            torch.cuda.empty_cache()
        print(f"{fold_string} training process is completed")

if __name__ == "__main__":
    class config:
        model="utae"
        encoder_widths=[64, 64, 64, 128]
        decoder_widths=[32, 32, 64, 128]
        out_conv=[32, 1]
        str_conv_k=4
        str_conv_s=2
        str_conv_p=1
        agg_mode="att_group"
        encoder_norm="group"
        n_head=16
        d_model=256
        d_k=4
        pad_value=0
        padding_mode="reflect"
        ignore_index=None   
        epochs=100
        batch_size=8
        num_workers=20
        display_step=860
        lr=0.001
        fold=None
        dataset_folder="./JAXA"
        ref_date="2020-01-01"
        res_dir="./artifacts"
        exp_name="binary_JAXA"
        device="cuda"
    
    train_loop(config=config)
