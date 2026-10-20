# Training data loaders. Adapted from InsightFace arcface_torch (MIT License) and CR-FIQA.
# Every sample is returned together with a unique integer index, which is used to
# store and accumulate its relative point margin across epochs.
import numbers
import os
import queue as Queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    """InsightFace record format (train.rec / train.idx), e.g. MS1MV2 (faces_emore)."""

    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        np.bool = np.bool_  # mxnet 1.x still uses the removed np.bool alias
        import mxnet as mx

        self.mx = mx
        self.transform = TRAIN_TRANSFORM
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = self.mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = self.mx.image.imdecode(img).asnumpy()
        sample = self.transform(sample)
        return sample, label, idx

    def __len__(self):
        return len(self.imgidx)


class ImageFolderDataset(Dataset):
    """Aligned 112x112 images stored as <root>/<identity>/<image>, e.g. CASIA-WebFace."""

    def __init__(self, root_dir, num_classes=None):
        super(ImageFolderDataset, self).__init__()
        self.transform = TRAIN_TRANSFORM
        self.root_dir = root_dir
        self.imgidx, self.labels = [], []
        identities = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        for lb, identity in enumerate(identities[:num_classes]):
            for img in os.listdir(os.path.join(root_dir, identity)):
                self.imgidx.append(os.path.join(identity, img))
                self.labels.append(lb)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root_dir, self.imgidx[index]))
        sample = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sample, label, index

    def __len__(self):
        return len(self.imgidx)
