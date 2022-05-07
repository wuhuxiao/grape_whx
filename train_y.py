import argparse
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from training.gnn_y import train_gnn_y
from uci.uci_subparser import add_uci_subparser
from utils.logger import *
from utils.utils import auto_select_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None, )  # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None, )  # default to be all true
    parser.add_argument('--aggr', type=str, default='mean', )
    parser.add_argument('--node_dim', type=int, default=16)
    parser.add_argument('--edge_dim', type=int, default=16)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.8)
    parser.add_argument('--valid', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='y6')
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    args = parser.parse_args()
    print(args)

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_path = './{}/test/{}/{}/'.format(args.domain, args.data, args.log_dir)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)

    logger = get_logger('GRAPE', log_file=log_path + '/train.log')
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print_log(cmd_input, logger=logger)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args, logger)
    train_gnn_y(data, args, log_path, device, logger)


if __name__ == '__main__':
    main()
