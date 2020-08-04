# -*- coding: utf-8 -*-

"""
# File name:    config.py.py
# Time :        2020/4/22 23:11
# Author:       guoxuyang@bytedance.com
# Description:  
"""

import addict  # nesting dict

configs = [
    addict.Dict({
        # experiment name
        "experiment_name": "01_origin_stargan_blond_hair",
        "gpu": '0',
        "selected_attrs": ["Blond_Hair"],
        "noise_T_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "celeba_image_dir": r"E:\软件与工具\temp\CelebA\Img\img_align_celeba\img_align_celeba",
        "attr_path": r"E:\软件与工具\temp\CelebA\Anno\list_attr_celeba.txt",
        #"resume_iters": 160000,
    }),

    addict.Dict({
        # experiment name
        "experiment_name": "02_origin_stargan__blond_hair__symmetric0.3",
        "gpu": '3',
        "selected_attrs": ["Blond_Hair"],
        "noise_T_matrix": [[0.7, 0.3], [0.3, 0.7]],
        "celeba_image_dir": "../../data/celeba_data/img_align_celeba",
        "attr_path": "../../data/celeba_data/Anno/list_attr_celeba.txt",
        "resume_iters": 160000,
    }),

    addict.Dict({
        # experiment name
        "experiment_name": "03_origin_stargan__bangs",
        "gpu": '4',
        "selected_attrs": ["Bangs"],
        "noise_T_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "celeba_image_dir": "../../data/celeba_data/img_align_celeba",
        "attr_path": "../../data/celeba_data/Anno/list_attr_celeba.txt",
        # "resume_iters": 160000,
    }),

    addict.Dict({
        # experiment name
        "experiment_name": "04_origin_stargan__bangs__symmetric0.3",
        "gpu": '5',
        "selected_attrs": ["Bangs"],
        "noise_T_matrix": [[0.7, 0.3], [0.3, 0.7]],
        "celeba_image_dir": "../../data/celeba_data/img_align_celeba",
        "attr_path": "../../data/celeba_data/Anno/list_attr_celeba.txt",
        # "resume_iters": 160000,
    }),
]


def get_config(id):
    for c in configs:
        if c.experiment_name.startswith(id):
            config = c
            return config
