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
        self,
        folder,
        folds=None,
        reference_date="2020-01-01",
        sat="S2",  # Changed to a single sat
    ):
        super(UtaeDataset, self).__init__()
        self.folder = folder
        self.folds = folds
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.sat = sat  # Store the single satellite

        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.json"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_table = None  # Simplified for a single satellite
        self.date_range = np.array(range(0, 366))

        dates = self.meta_patch["dates-{}".format(sat)]
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
        self.date_table = {
            index: np.array(list(d.values()))
            for index, d in date_table.to_dict(orient="index").items()
        }

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

    def __len__(self):
        return self.len

    def get_dates(self, id_patch):  # Simplified for a single satellite
        return self.date_range[np.where(self.date_table[id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        data = np.load(
            os.path.join(
                self.folder,
                "DATA_{}".format(self.sat),
                "{}_{}.npy".format(self.sat, id_patch),
            )
        ).astype(np.float32)

        data = torch.from_numpy(data)
        data = data / 10000.0

        T, C, H, W = data.shape
        data = data.view(T * C, 1, H, W)
        data = F.interpolate(data, size=(128, 128), mode="bilinear")
        data = data.view(T, C, 128, 128)

        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target.astype(int))
        target = torch.where(target == 3, 1, 0)
        # target = torch.where(target > 3, torch.tensor(0), target)
        target = (
            F.interpolate(
                target.unsqueeze(0).unsqueeze(0).float(),
                size=(128, 128),
                mode="bilinear",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        dates = torch.from_numpy(self.get_dates(id_patch))

        return (data, dates), target
