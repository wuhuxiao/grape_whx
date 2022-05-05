import argparse
import os
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import *

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.logger import *
from utils.utils import auto_select_gpu


def test_gnn_y(data, args, log_path, device=torch.device('cpu'), logger=None):
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    impute_model = MLPNet(input_dim, 1,
                          hidden_layer_sizes=impute_hiddens,
                          hidden_activation=args.impute_activation,
                          dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
    n_row, n_col = data.df_X.shape
    predict_model = MLPNet(n_col, 1,
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

    # 载入训练好的模型参数
    load_path = log_path
    print("loading from {} ".format(load_path))
    model = torch.load(load_path + 'model_best_test_acc.pt', map_location=device)
    impute_model = torch.load(load_path + 'impute_model_best_test_acc.pt', map_location=device)
    predict_model = torch.load(load_path + 'predict_model_best_test_acc.pt', map_location=device)

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    edge_attr = data.edge_attr.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    train_y_mask = all_train_y_mask.clone().detach()

    model.eval()
    impute_model.eval()
    predict_model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i in range(2):
            if i == 0:
                time_start = time.time()
                x_embd = model(x, edge_attr, edge_index)
                print_log('无指标缺失', logger=logger)
                X = impute_model(
                    [x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X)[:, 0]
                time_end = time.time()
                print_log('time cost %.4f s' % (time_end - time_start), logger=logger)
            else:
                time_start = time.time()
                x_embd = model(x, train_edge_attr, train_edge_index)
                print_log('20%指标缺失', logger=logger)

                X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X)[:, 0]
                time_end = time.time()
                print_log('time cost %.4f s'%( time_end - time_start), logger=logger)
            pred_test = pred
            label_test = y

            pred_test_sig = sigmoid(pred_test)
            # pred_test_int = pred_test_sig.cpu().numpy()
            # label_test_int = label_test.cpu().numpy()

            pred_test_int = pred_test_sig.detach().cpu().numpy()
            label_test_int = label_test.detach().cpu().numpy()

            auc = roc_auc_score(label_test_int, pred_test_int)
            pred_test_int[pred_test_int > 0.5] = 1
            pred_test_int[pred_test_int < 1] = 0

            confusion = confusion_matrix(label_test_int, pred_test_int)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            acc = (TP + TN) / float(TP + TN + FP + FN)
            if TP == 0:
                sensi = 0
            else:
                sensi = TP / float(TP + FN)
            speci = TN / float(TN + FP)

            print_log('auc:  %.4f' % auc, logger=logger)
            print_log('Accuracy:  %.4f' % acc, logger=logger)
            print_log('Sensitivity:  %.4f' % sensi, logger=logger)
            print_log('Specificity:  %.4f' % speci, logger=logger)
            print_log('======-======', logger=logger)


def main():
    parser = argparse.ArgumentParser()
    #

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
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='test')
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser('uci')
    # mc settings
    subparser.add_argument('--domain', type=str, default='uci')
    subparser.add_argument('--data', type=str, default='mimic')
    subparser.add_argument('--train_edge', type=float, default=0.8)
    subparser.add_argument('--split_sample', type=float, default=0.)
    subparser.add_argument('--split_by', type=str, default='y')  # 'y', 'random'
    subparser.add_argument('--split_train', action='store_true', default=False)
    subparser.add_argument('--split_test', action='store_true', default=False)
    subparser.add_argument('--train_y', type=float, default=0.7)
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot
    args = parser.parse_args(args=['uci'])
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
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger('GRAPE', log_file=log_path + '/train.log')
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print_log(cmd_input, logger=logger)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args, logger)
    test_gnn_y(data, args, log_path, device, logger)


if __name__ == '__main__':
    main()
