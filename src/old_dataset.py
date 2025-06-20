import os
import json
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Dataset


class UtaeDataset(Dataset):
    def __init__(
        self, folder, norm=True, folds=None, reference_date="2020-01-01", sats=["S2"]
    ):
        super(UtaeDataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.folds = folds
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.sats = sats

        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.json"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(0, 366))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        data = {
            satellite: np.load(
                os.path.join(
                    self.folder,
                    "DATA_{}".format(satellite),
                    "{}_{}.npy".format(satellite, id_patch),
                )
            ).astype(np.float32)
            for satellite in self.sats
        }
        data = {s: torch.from_numpy(a) for s, a in data.items()}

        if self.norm is not None:
            data = {
                s: (d - self.norm[s][0][None, :, None, None])
                / self.norm[s][1][None, :, None, None]
                for s, d in data.items()
            }

        for s in data:
            d = data[s]  # d shape: (T, C, H, W)
            T, C, H, W = d.shape
            d = d.view(T * C, 1, H, W)
            d = F.interpolate(d, size=(128, 128), mode="bilinear", align_corners=False)
            d = d.view(T, C, 128, 128)
            data[s] = d

        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target.astype(int))
        # target = torch.where((target == 4) | (target == 5) | (target == 6) | (target == 7) | (target == 8) | (target == 9) | (target == 10) | (target == 11) | (target == 12),  0, target)
        # target = torch.where((target == 5) | (target == 8) | (target == 9) | (target == 10) | (target == 11),  5, target)
        # target = torch.where(target == 12, 8, target)
        target = torch.where(target == 3, 1, 0)
        target = (
            F.interpolate(
                target.unsqueeze(0).unsqueeze(0).float(),
                size=(128, 128),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        dates = {s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats}
        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]
        return (data, dates), target


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = UtaeDataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))
