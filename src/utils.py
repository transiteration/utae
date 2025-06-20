import collections.abc
import re
import numpy as np
import torch
from torch.nn import functional as F

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


cmap = np.array(
    [
        [0, 0, 0],  # Class 0 - No data (Black)
        [0, 0, 255],  # Class 1 - Water (Blue)
        [192, 192, 192],  # Class 2 - Urban/Built-up (Gray)
        [255, 255, 0],  # Class 3 - Rice (Yellow)
        [220, 20, 60],  # Class 4 - Other Crops (Crimson)
        [59, 237, 85],  # Class 5 - Grass/Shrub (Light Green)
        [139, 69, 19],  # Class 6 - Woody Crops/Orchards (Brown)
        [210, 180, 140],  # Class 7 - Barren (Light Brown)
        [0, 128, 0],  # Class 8 - Evergreen Forest (Dark Green)
        [154, 205, 50],  # Class 9 - Deciduous Forest (Warm Green)
        [255, 105, 180],  # Class 10 - Plantation Forest (Pink)
        [102, 205, 170],  # Class 11 - Mangrove Forest (Aquamarine)
        [166, 113, 227],  # Class 12 - Aquaculture (Lavander)
    ]
)

cmap_2 = np.array(
    [
        [0, 0, 0],  # Class 0 - No data (Black)
        [0, 0, 255],  # Class 1 - Water (Blue)
        [192, 192, 192],  # Class 2 - Urban/Built-up (Gray)
        [255, 255, 0],  # Class 3 - Rice (Yellow)
        [220, 20, 60],  # Class 4 - Other Crops (Crimson)
        [0, 128, 0],  # Class 5 - Evergreen Forest (Dark Green)
        [139, 69, 19],  # Class 6 - Woody Crops/Orchards (Brown)
        [210, 180, 140],  # Class 7 - Barren (Light Brown)
        [166, 113, 227],  # Class 8 - Aquaculture (Lavander)
    ]
)

class_names = [
    "No data",
    "Water",
    "Urban/Built-up",
    "Rice",
    "Other Crops",
    "Grass/Shrub",
    "Woody Crops/Orchards",
    "Barren",
    "Evergreen Forest",
    "Deciduous Forest",
    "Plantation Forest ",
    "Mangrove Forest",
    "Aquaculture",
]

class_names_2 = [
    "No data",
    "Water",
    "Urban/Built-up",
    "Rice",
    "Other Crops",
    "Forest",
    "Woody Crops/Orchards",
    "Barren",
    "Aquaculture",
]
