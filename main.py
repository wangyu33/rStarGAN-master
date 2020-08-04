
import torch
import tensorflow as tf
print(torch.cuda.is_available())
print(tf.test.is_built_with_cuda())
import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import configparser


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # change gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size // 2,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size // 2,
                                 'RaFD', config.mode, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train(config.noise_T_matrix)
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    def add_default_value_to_cfg(cfg, key, value):
        if key not in cfg:
            cfg[key] = value

    import config
    import sys
    import json

    cfg = config.get_config(sys.argv[1])
    cfg['c_dim'] = len(cfg['selected_attrs'])

    output_root_dir = 'output/%s' % cfg['experiment_name']
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    with open(output_root_dir + '/setting.json', 'w') as f:
        f.write(json.dumps(cfg, indent=4, separators=(',', ': ')))

    add_default_value_to_cfg(cfg, 'log_dir', output_root_dir + '/logs')
    add_default_value_to_cfg(cfg, 'model_save_dir', output_root_dir + '/models')
    add_default_value_to_cfg(cfg, 'sample_dir', output_root_dir + '/samples')
    add_default_value_to_cfg(cfg, 'result_dir', output_root_dir + '/results')

    # Model configuration
    add_default_value_to_cfg(cfg, 'celeba_crop_size', 178)
    add_default_value_to_cfg(cfg, 'image_size', 128)
    add_default_value_to_cfg(cfg, 'g_conv_dim', 64)
    add_default_value_to_cfg(cfg, 'd_conv_dim', 64)
    add_default_value_to_cfg(cfg, 'g_repeat_num', 6)
    add_default_value_to_cfg(cfg, 'd_repeat_num', 6)

    add_default_value_to_cfg(cfg, 'lambda_gp', 10.0)
    add_default_value_to_cfg(cfg, 'lambda_adv', 1.0)

    # Training configuration
    add_default_value_to_cfg(cfg, 'dataset', 'CelebA')
    add_default_value_to_cfg(cfg, 'batch_size', 16)
    add_default_value_to_cfg(cfg, 'num_iters', 200000)
    add_default_value_to_cfg(cfg, 'num_iters_decay', 180000)
    add_default_value_to_cfg(cfg, 'g_lr', 0.0001)
    add_default_value_to_cfg(cfg, 'd_lr', 0.0001)
    add_default_value_to_cfg(cfg, 'n_critic', 5)
    add_default_value_to_cfg(cfg, 'beta1', 0.5)
    add_default_value_to_cfg(cfg, 'beta2', 0.999)
    add_default_value_to_cfg(cfg, 'resume_iters', None)

    # Testing configuration
    add_default_value_to_cfg(cfg, 'test_iters', 200000)

    # Miscellaneous
    add_default_value_to_cfg(cfg, 'num_workers', 1)
    add_default_value_to_cfg(cfg, 'mode', 'train')
    add_default_value_to_cfg(cfg, 'use_tensorboard', True)

    add_default_value_to_cfg(cfg, 'log_step', 10)
    add_default_value_to_cfg(cfg, 'sample_step', 2000)
    add_default_value_to_cfg(cfg, 'model_save_step', 20000)
    add_default_value_to_cfg(cfg, 'lr_update_step', 1000)

    # modify
    add_default_value_to_cfg(cfg, 'g_type', 'default')

    print(cfg)
    main(cfg)
