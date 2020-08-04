from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import numpy as np
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
            # '''get attribute ratio'''   np.array([int(i[1][0]) for i in self.train_dataset]).sum() / 206000
        print('Finished preprocessing the CelebA dataset...')

        # guoxuyang balanced sample

        def get_balanced_dataset(dataset):
            dataset_attrs = np.array([l[1] for l in dataset])
            pair_dataset = []
            for k in range(len(self.selected_attrs)):
                pos_dataset = []
                neg_dataset = []
                for data_item in range(len(dataset_attrs)):
                    if dataset_attrs[data_item, k]:
                        pos_dataset.append(dataset[data_item])
                    else:
                        neg_dataset.append(dataset[data_item])
                pair_dataset.append([pos_dataset, neg_dataset])
            return pair_dataset
        self.train_pair_dataset = get_balanced_dataset(self.train_dataset)
        self.test_pair_dataset = get_balanced_dataset(self.test_dataset)


    # def __getitem__(self, index, mode=None):
    #     """Return one image and its corresponding attribute label."""
    #     if mode is None:
    #         mode = self.mode
    #     dataset = self.train_dataset if mode == 'train' else self.test_dataset
    #     filename, label = dataset[index]
    #     image = Image.open(os.path.join(self.image_dir, filename))
    #     return self.transform(image), torch.FloatTensor(label)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        pos_dataset, neg_dataset = self.train_pair_dataset[0]

        filename1, label1 = random.choice(pos_dataset)
        image1 = Image.open(os.path.join(self.image_dir, filename1))
        label1 = torch.FloatTensor(label1)

        filename0, label0 = random.choice(neg_dataset)
        image0 = Image.open(os.path.join(self.image_dir, filename0))
        label0 = torch.FloatTensor(label0)
        return self.transform(image1), self.transform(image0), label1, label0


    def get_balanced_batch(self, batch_size, mode='train', att_idx=0):
        dataset = self.train_pair_dataset if mode == 'train' else self.test_pair_dataset
        pos_dataset, neg_dataset = dataset[att_idx]

        def get_random_data(sample_size, half_dataset):
            imgs, labels = [], []
            for kk in range(sample_size):
                if mode == 'train':
                    filename, label = random.choice(half_dataset)
                else:
                    filename, label = half_dataset[kk]
                image = Image.open(os.path.join(self.image_dir, filename))
                imgs.append(self.transform(image).unsqueeze(0))
                labels.append(torch.FloatTensor(label).unsqueeze(0))
            return imgs, labels
        imgs_1, labels_1 = get_random_data(batch_size // 2, pos_dataset)
        imgs_0, labels_0 = get_random_data(batch_size // 2, neg_dataset)
        return torch.cat(imgs_1 + imgs_0, dim=0), torch.cat(labels_1 + labels_0, dim=0)


    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader