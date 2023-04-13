import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
import torch as th

import lmdb
import pickle
from torchvision import transforms


def load_data(
    *,
    data_dir,
    batch_size,
    num_workers=16,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = ImageDataset(
        path=data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    while True:
        yield from loader


def load_data_local(
    *,
    data_dir,
    batch_size,
):
    env = lmdb.open(
        data_dir,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    if not env:
        raise IOError("Cannot open lmdb dataset", data_dir)

    with env.begin(write=False) as txn:
        length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    print("data: ", length)

    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose(transform)

    zfill = 6

    data_image = []
    data_rendered = []
    data_normal = []
    data_albedo = []

    with env.begin(write=False) as txn:
        for index in range(length):

            key = f"image_{str(index).zfill(zfill)}".encode("utf-8")
            image_bytes = txn.get(key)

            key = f"normal_{str(index).zfill(zfill)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(zfill)}".encode("utf-8")
            albedo_bytes = txn.get(key)

            key = f"rendered_{str(index).zfill(zfill)}".encode("utf-8")
            rendered_bytes = txn.get(key)

            buffer = BytesIO(image_bytes)
            image = Image.open(buffer)

            buffer = BytesIO(normal_bytes)
            normal = pickle.load(buffer)

            buffer = BytesIO(albedo_bytes)
            albedo = pickle.load(buffer)

            buffer = BytesIO(rendered_bytes)
            rendered = pickle.load(buffer)

            image = transform(image)

            data_image.append(image)
            data_normal.append(normal)
            data_albedo.append(albedo)
            data_rendered.append(rendered)

    data_image = th.stack(data_image, 0)
    data_rendered = th.stack(data_rendered, 0)
    data_normal = th.stack(data_normal, 0)
    data_albedo = th.stack(data_albedo, 0)

    while True:
        idxs = np.random.choice(length, batch_size, replace=False)
        yield {
            "image": data_image[idxs],
            "rendered": data_rendered[idxs],
            "normal": data_normal[idxs],
            "albedo": data_albedo[idxs],
        }


class ImageDataset(Dataset):
    def __init__(
        self,
        path,
        shard=0,
        num_shards=1,
    ):
        super().__init__()

        self.zfill = 6

        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)][shard:][::num_shards]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):

        index = self.idxs[index]

        with self.env.begin(write=False) as txn:
            key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
            image_bytes = txn.get(key)

            key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
            albedo_bytes = txn.get(key)

            key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
            rendered_bytes = txn.get(key)

        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)

        buffer = BytesIO(normal_bytes)
        normal = pickle.load(buffer)

        buffer = BytesIO(albedo_bytes)
        albedo = pickle.load(buffer)

        buffer = BytesIO(rendered_bytes)
        rendered = pickle.load(buffer)
        image = self.transform(image)

        return {
            "image": image,
            "normal": normal,
            "albedo": albedo,
            "rendered": rendered,
        }
