import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from ignite.metrics import Precision
from sklearn.metrics import *

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge
from utils.logger import *


def train_gnn_y(data, args, log_path, device=torch.device('cpu'), logger=None):
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

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []
    Test_acc = []
    Test_sensi = []
    Test_speci = []
    best_test_score = 0
    best_test_acc = 0
    best_test_sensi = 0
    best_test_speci = 0
    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask * all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
        msg = "all y num is {}, train num is {}, valid num is {}, test num is {}" \
            .format(
            all_train_y_mask.shape[0], torch.sum(train_y_mask),
            torch.sum(valid_y_mask), torch.sum(test_y_mask))
        print_log(msg, logger=logger)
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_y_mask = all_train_y_mask.clone().detach()
        print("all y num is {}, train num is {}, test num is {}" \
            .format(
            all_train_y_mask.shape[0], torch.sum(train_y_mask),
            torch.sum(test_y_mask)))
    # ??????????????????
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])
        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]
        if args.data == 'mimic' or args.data == 'mimic_key':
            loss = criterion(sigmoid(pred_train), label_train)
        else:
            loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        predict_model.eval()

        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                X = impute_model(
                    [x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X)[:, 0]
                pred_valid = pred[valid_y_mask]
                label_valid = y[valid_y_mask]
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_l1.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_rmse.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, train_edge_attr, train_edge_index)
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            X = torch.reshape(X, [n_row, n_col])
            pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            if args.data == 'mimic' or args.data == 'mimic_key':
                pred_test_sig = sigmoid(pred_test)
                mse = criterion(pred_test_sig, label_test)
                pred_test_int = pred_test_sig.cpu().numpy()
                label_test_int = label_test.cpu().numpy()

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
                Test_acc.append(acc)
                Test_sensi.append(sensi)
                Test_speci.append(speci)

            else:
                mse = F.mse_loss(pred_test, label_test)
                # l1 = F.l1_loss(pred_test, label_test)
            # test_l1 = l1.item()
            test_rmse = np.sqrt(mse.item())
            best_score = acc + sensi * 2 + speci
            if acc > 0.8 and sensi > 0.8 and speci > 0.8 and best_score>best_test_score:
                best_test_score = best_score
                best_valid_l1_epoch = epoch
                torch.save(model, log_path + 'model_best_test_acc.pt')
                torch.save(impute_model, log_path + 'impute_model_best_test_acc.pt')
                torch.save(predict_model, log_path + 'predict_model_best_test_acc.pt')
                print_log('epoch: %d' % epoch, logger=logger)

                if args.valid > 0.:
                    print('valid rmse: ', valid_rmse)
                    print('valid l1: ', valid_l1)
                print_log('test rmse:  %.4f' % test_rmse, logger=logger)
                print_log('auc:  %.4f' % auc, logger=logger)
                print_log('Accuracy:  %.4f' % acc, logger=logger)
                print_log('Sensitivity:  %.4f' % sensi, logger=logger)
                print_log('Specificity:  %.4f' % speci, logger=logger)
                print_log('======-======', logger=logger)

            Train_loss.append(train_loss)
            Test_rmse.append(test_rmse)

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1
    # obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_acc'] = Test_acc
    obj['curves']['test_sensi'] = Test_sensi
    obj['curves']['test_speci'] = Test_speci
    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model.pt')
    torch.save(impute_model, log_path + 'impute_model.pt')
    torch.save(predict_model, log_path + 'predict_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path + 'curves.png', keys=None,
               clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path + 'lr.png', keys=['lr'],
               clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path + 'outputs.png',
                groups=[['pred_train', 'label_train'],
                        ['pred_test', 'label_test']
                        ],
                num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse, best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1, best_valid_l1_epoch))
