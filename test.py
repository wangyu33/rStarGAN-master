#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : test.py
# Author: WangYu
# Date  : 2020-05-08

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from model import Generator, Discriminator
import numpy as np
from torchvision.utils import save_image
import pandas as pd
from config import get_config
import os


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
        self.img_list = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        # print(self.selected_attrs in all_attr_names)
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        # print(self.attr2idx['Smiling'])
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
                self.img_list.append([filename])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

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
                pair_dataset.extend(pos_dataset)
                pair_dataset.extend(neg_dataset)
            return pair_dataset

        self.test_pair_dataset = get_balanced_dataset(self.test_dataset)

    def __getitem__(self, index, mode=None):
        """Return one image and its corresponding attribute label."""
        if mode is None:
            mode = self.mode
        dataset = self.test_pair_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, mode='test', num_workers=1):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=10,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def select_pic(img_list=None, image_dir=None):
    transform = T.Compose([
        T.CenterCrop(178),
        T.Resize(128),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_list = pd.read_csv('img_list.csv')
    # 在img_list 输入挑选的数据
    # smile
    if img_list == None:
        temp = [3, 11, 29, 36, 51, 52, 122, 185, 200, 305, 1240, 1387, 1420, 1428, 1597, 1694, 1894, 1956]
        img_list = temp
    name = test_list.name
    label = test_list.label
    imgs = []
    labels = []
    for i in img_list:
        img = Image.open(os.path.join(image_dir, name[i]))
        label1 = label[i]
        if label1[1] == 'T':
            labels.append(torch.FloatTensor([1.0]).unsqueeze(0))
        else:
            labels.append(torch.FloatTensor([0.0]).unsqueeze(0))
        imgs.append(transform(img).unsqueeze(0))
        # labels.append(torch.FloatTensor(label1).unsqueeze(0))
    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    # print(imgs.shape)
    # print(labels.shape)
    return imgs, labels


def acc_test(D_model, img_path, pos_iter = 20, neg_iter = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dsc = Discriminator(128, 64, 1, 6).to(device)
    dsc.load_state_dict(torch.load(D_model))
    pos_acc = 0
    for i in range(pos_iter):
        list = [i for i in range(i * 10, i * 10 + 10)]
        pos_img, pos_label = select_pic(list, img_path)
        pos_img = pos_img.to(device)
        pos_label = pos_label.to(device)
        D_label = dsc(pos_img)[1]
        D_label = torch.where(D_label > 0, torch.ones_like(D_label), torch.zeros_like(D_label))
        pos_acc += torch.sum(D_label == pos_label)
        print('the {} batch\'s pos_acc is {}'.format(i + 1, pos_acc * 1.0 / (i * 10 + 10)))
    neg_acc = 0
    for i in range(neg_iter):
        list = [i for i in range(i * 10 + 1999 - neg_iter * 10, i * 10 + 10 + 1999 - neg_iter * 10)]
        neg_img, neg_label = select_pic(list, img_path)
        neg_img = neg_img.to(device)
        neg_label = neg_label.to(device)
        D_label = dsc(neg_img)[1]
        D_label = torch.where(D_label > 0, torch.ones_like(D_label), torch.zeros_like(D_label))
        neg_acc += torch.sum(D_label == neg_label)
        print('the {} batch\'s neg_acc is {}'.format(i + 1, neg_acc * 1.0 / (i * 10 + 10)))
    print('pos_acc is {}'.format(pos_acc * 1.0 / pos_iter / 10))
    print('neg_acc is {}'.format(neg_acc * 1.0 / neg_iter / 10))


def refresh_csv(selected_attrs):
    celeba_loader = get_loader(image_dir, attr_path, selected_attrs, mode='test')
    img_list = celeba_loader.dataset.test_pair_dataset
    df = pd.DataFrame(data=img_list)
    df.to_csv('img_list.csv', encoding='utf-8', header=['name', 'label'], index=False)
    return celeba_loader


def main(G_model_path, selected_attrs, select=0,
         output_dir=r'C:\Users\wangyu\Desktop\stargan\rStarGAN-master\rStarGAN-master\test_pic', select_list=None, g_type='res'):
    """
    :param G_model_path: 给一个或者给个List都行
    :param selected_attrs:
    :param D_model:
    :param select: 0: 随机选， 1：选一批人 （跑1前必须先跑0）
    :param select_list: select=1时，指定哪些人要选出来
    :param g_type: 'res' / 'default'/ 一个List
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not isinstance(G_model_path, list):
        G_model_path = [G_model_path]

    if not isinstance(g_type, list):
        g_type = [g_type] * len(G_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    celeba_loader = refresh_csv(selected_attrs)

    generators = []
    for kk, g_model_path in enumerate(G_model_path):
        generator = Generator(64, len(selected_attrs), 6, g_type[kk]).to(device)
        generator.load_state_dict(torch.load(g_model_path))
        generators.append(generator)
    # dsc = Discriminator(128, 64, len(selected_attrs), 6).to(device)
    # dsc.load_state_dict(torch.load(D_model))
    #
    # D_acc = 0
    if select == 0:
        for i, data in enumerate(celeba_loader):
            print(i)
            img = data[0].to(device)
            # print(img.shape)
            label = data[1].to(device)
            # print(label.shape)

            img_fakes = []
            for g in generators:
                img_fake = g(img, 1 - label)[0]
                img_fake = torch.clamp(img_fake, -1, 1)
                img_fakes.append(img_fake)
            # D_label = dsc(img)[1]
            # D_label = torch.where(D_label > 0, torch.ones_like(D_label), torch.zeros_like(D_label))
            # D_acc += torch.sum(D_label == label)
            # print('the {} batch\'s acc is {}'.format(i+1, D_acc*1.0/(i*10 + 10)))
            # print(img_fake.shape)
            img_ = torch.cat([img] + img_fakes, dim=3).to(device)
            sample_path = os.path.join(output_dir, '%06d-images.jpg' % i)
            save_image(denorm(img_), sample_path, nrow=1, padding=0)
    else:
        img, label = select_pic(select_list, image_dir=image_dir)
        img = img.to(device)
        label = label.to(device)

        img_fakes = []
        for g in generators:
            img_fake = g(img, 1 - label)[0]
            img_fake = torch.clamp(img_fake, -1, 1)
            img_fakes.append(img_fake)
        # print(img_fake.shape)
        img_ = torch.cat([img] + img_fakes, dim=3).to(device)
        print(img_.shape)
        sample_path = os.path.join(output_dir, 'select.jpg')
        save_image(denorm(img_), sample_path, nrow=1, padding=0)


if __name__ == '__main__':
    image_dir = r"E:\软件与工具\temp\CelebA\Img\img_align_celeba\img_align_celeba"
    attr_path = r"E:\软件与工具\temp\CelebA\Anno\list_attr_celeba.txt"

    D_path = r'E:\model\model\46_Eyeglasses_0.4反转_rStarGAN3\180000-D.ckpt'
    refresh_csv(['Eyeglasses'])
    acc_test(D_path, image_dir, pos_iter=14)

    # select_pic()
    # 输入模型地址就可以
    # path = r'C:\Users\wangyu\Desktop\stargan\rStarGAN-master\rStarGAN-master\model\Smiling'
    # D_model = path + r'\180000-D.ckpt'
    # G_model_path_12w = path + r'\120000-G.ckpt'
    # G_model_path_18w = path + r'\180000-G.ckpt'
    # main(G_model_path_12w,G_model_path_18w, selected_attrs = ["Smiling"], D_model = D_model,select=0)

    # refresh_csv(['Smiling'])
    # acc_test(D_model = D_model)

    # cfg = get_config('58')
    # cfg_path = 'output/' + cfg['experiment_name']
    # model_path = cfg_path  + '/models'
    # image_dir = cfg.celeba_image_dir
    # attr_path = cfg.attr_path

    #glass
    img_list = [4, 9, 25, 49, 162, 191, 204]
    main([#r'E:\model\model\55_Eyeglasses_0.2反转_rStarGAN3\140000-G.ckpt',
          r'E:\model\model\53_Eyeglasses_0.3反转_rStarGAN3\200000-G.ckpt',
          r'E:\model\model\46_Eyeglasses_0.4反转_rStarGAN3\180000-G.ckpt',
          r'E:\model\model\57_Eyeglasses_0.45反转_rStarGAN3\200000-G.ckpt'],
         selected_attrs = ['Eyeglasses'],
         output_dir='output_ablation_study/' + 'Eyeglasses', g_type='res', select=1,
        #select_list=img_list
    )
    # g_type = ['default','res']
