from datetime import datetime
from functools import reduce, partialmethod
from pprint import pprint
from rdkit import RDLogger
from scipy.stats import rankdata, spearmanr
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from torch_geometric.loader import DataLoader
from time import time
from torch.utils.data import Subset, SubsetRandomSampler, WeightedRandomSampler
import argparse
import copy
import heapq
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from model import *
RDLogger.DisableLog('rdApp.*')


with open('data/Text_similarity_five.pkl', 'rb') as f:
    drug_similarity = torch.from_numpy(pickle.load(f))

with open('data/effect_side_semantic.pkl', 'rb') as f:
    side_similarity = torch.from_numpy(pickle.load(f))

with open('data/drug_side.pkl', 'rb') as f:
    Y = pickle.load(f)

dd_sim = None
ss_sim = None


def train(model, train_loader, sampler, optimizer):
    model.train()
    train_loss = [0., 0., 0.]

    drug_mask = (sampler.sum(axis=1) > 0).astype(float)
    side_mask = (sampler.sum(axis=0) > 0).astype(float)

    for train_data in train_loader:
        i, x, y = train_data
        fx, D, S = model(x)
        j = np.arange(994)

        fx = fx.flatten()
        y = y.flatten()
        lw = torch.ones_like(y)

        lw = lw * sampler[i].flatten()
        lw = lw * ((y > 0).float() + .03 * (y == 0).float())
        lw = lw.to(model.device)

        predictive_loss = torch.sum(lw * (fx - y.to(model.device)) ** 2)
        predictive_loss *= (10000 / predictive_loss.item())
        loss = predictive_loss
        train_loss[0] += predictive_loss.item()

        if dd_sim is not None:
            d_weight = sampler.sum(axis=1)
            d_weight = (d_weight.max() - d_weight)[i]
            d_weight = torch.from_numpy(np.outer(d_weight, d_weight.T)).to(model.device)

            drug_sim = (dd_sim * drug_mask[:, None])[i][:, i].to(model.device)
            drug_embedding_loss = torch.sum(torch.abs(torch.mm(D, D.T) - drug_sim) * d_weight)

            if drug_embedding_loss.item() > 0:
                drug_embedding_loss *= (1 / drug_embedding_loss.item())
                loss += drug_embedding_loss
                train_loss[1] += drug_embedding_loss.item()

        if ss_sim is not None:
            s_weight = sampler.sum(axis=0)
            s_weight = (s_weight.max() - s_weight)
            s_weight = torch.from_numpy(np.outer(s_weight, s_weight.T)).to(model.device)

            side_sim = (ss_sim * side_mask[:, None])[j][:, j].to(model.device)
            side_embedding_loss = torch.sum(torch.abs(torch.mm(S, S.T) - side_sim) * s_weight)

            if side_embedding_loss.item() > 0:
                side_embedding_loss *= (1 / side_embedding_loss.item())
                loss += side_embedding_loss
                train_loss[2] += side_embedding_loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss[0] /= len(train_loader)
    train_loss[1] /= len(train_loader)
    train_loss[2] /= len(train_loader)

    return train_loss


def test(model, test_loader, test_mask):
    model.eval()
    ys = []
    fxs = []

    drug_score = {
        'scc': {},
        'rmse': {},
        'mae': {},
        'auc': {},
        'map': {},
    }

    FX = np.zeros((750, 994))
    FX2 = np.zeros((750, 994))
    DV = None

    for test_data in test_loader:
        i, x, y = test_data
        fx, D, S = model(x)

        if DV is None:
            DV = np.zeros((750, D.shape[1]))

        DV[i] = D.cpu().detach().numpy()
        fx = fx.detach().cpu().numpy()
        FX[i] = fx

        fx = fx.flatten()
        y = y.flatten()
        d = np.repeat(i, 994)

        mask = test_mask[i].flatten()
        fx = fx[mask]
        y = y[mask]
        d = d[mask]

        for _d in np.unique(d):
            _y = y[d == _d]
            _fx = fx[d == _d]
            _y2 = [1 if y > 0 else 0 for y in _y]

            _yk = _y[np.array(_y2).astype(bool)]
            _fxk = _fx[np.array(_y2).astype(bool)]

            if np.unique(_yk).shape[0] > 1:
                _scc = spearmanr(_yk, _fxk).correlation
                drug_score['scc'][_d] = _scc

            if _yk.shape[0] > 0:
                _rmse = np.sqrt(metrics.mean_squared_error(_yk, _fxk))
                _mae = metrics.mean_absolute_error(_yk, _fxk)

                drug_score['rmse'][_d] = _rmse
                drug_score['mae'][_d] = _mae

            if np.count_nonzero(_y == 0) * np.count_nonzero(_y > 0) > 0:
                fpr, tpr, thresholds = metrics.roc_curve((_y > 0).float(), _fx, pos_label=1)
                threshold = thresholds[np.argmax(tpr - fpr)]

                _auc = metrics.auc(fpr, tpr)
                _map = metrics.average_precision_score((_y > 0).float(), _fx)

                FX2[_d] = (FX[_d] > threshold).astype(float)

                drug_score['auc'][_d] = _auc
                drug_score['map'][_d] = _map

        ys.extend(y.tolist())
        fxs.extend(fx.tolist())

    SV = S.cpu().detach().numpy()

    d_auroc = np.mean(list(drug_score['auc'].values()))
    d_auprc = np.mean(list(drug_score['map'].values()))

    ys = np.array(ys)
    fxs = np.array(fxs)
    ys_bin = [1 if y > 0 else 0 for y in ys]

    ys_known = ys[np.array(ys_bin).astype(bool)]
    fxs_known = fxs[np.array(ys_bin).astype(bool)]

    scc = spearmanr(ys_known, fxs_known).correlation
    rmse = np.sqrt(metrics.mean_squared_error(ys_known, fxs_known))
    mae = metrics.mean_absolute_error(ys_known, fxs_known)
    fpr, tpr, thresholds = metrics.roc_curve(ys_bin, fxs, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(ys_bin, fxs)

    test_score = (scc, rmse, mae, auroc, auprc, d_auroc, d_auprc)
    test_result = (FX, FX2, drug_score, DV, SV)

    return test_score, test_result


def boost(estimators, estimator_weights, test_loader, test_mask, mode):
    ys = []
    fxs = []

    drug_score = {
        'scc': {},
        'rmse': {},
        'mae': {},
        'auc': {},
        'map': {},
    }

    FX = np.zeros((750, 994))
    FX2 = np.zeros((750, 994))

    for test_data in test_loader:
        i, x, y = test_data

        if mode == 'classification':
            fx = torch.sum(torch.stack(
                list(map(
                    lambda t: t[0](x)[0] * t[1],
                    zip(estimators, estimator_weights)
                    ))), dim=0) / sum(estimator_weights)

        elif mode == 'regression':
            f = torch.stack(
                list(map(
                    lambda t: t(x)[0],
                    estimators
                    )))

            w = estimator_weights
            argsort = f.argsort(dim=0).cpu()

            cumulative_sum = np.zeros((f.shape[1], f.shape[2]))
            weighted_median_index = np.zeros((f.shape[1], f.shape[2]))
            weighted_median_index[:] = -1

            for j in range(len(estimators)):
                for boost_index in range(argsort.shape[0]):
                    cumulative_sum[argsort[boost_index] == j] += w[boost_index]

                    weighted_median_index[np.logical_and(
                        cumulative_sum >= np.sum(w) / 2,
                        weighted_median_index == -1
                        )] = boost_index

            fx = torch.zeros((f.shape[1], f.shape[2])).to(args.device)

            for j in range(len(estimators)):
                fx += f[j] * torch.from_numpy(weighted_median_index == j).float().to(args.device)

        fx = fx.detach().cpu().numpy()
        FX[i] = fx

        fx = fx.flatten()
        y = y.flatten()
        d = np.repeat(i, 994)

        mask = test_mask[i].flatten()
        fx = fx[mask]
        y = y[mask]
        d = d[mask]

        for _d in np.unique(d):
            _y = y[d == _d]
            _fx = fx[d == _d]
            _y2 = [1 if y > 0 else 0 for y in _y]

            _yk = _y[np.array(_y2).astype(bool)]
            _fxk = _fx[np.array(_y2).astype(bool)]

            if np.unique(_yk).shape[0] > 1:
                _scc = spearmanr(_yk, _fxk).correlation
                drug_score['scc'][_d] = _scc

            if _yk.shape[0] > 0:
                _rmse = np.sqrt(metrics.mean_squared_error(_yk, _fxk))
                _mae = metrics.mean_absolute_error(_yk, _fxk)

                drug_score['rmse'][_d] = _rmse
                drug_score['mae'][_d] = _mae

            if np.count_nonzero(_y == 0) * np.count_nonzero(_y > 0) > 0:
                fpr, tpr, thresholds = metrics.roc_curve((_y > 0).float(), _fx, pos_label=1)
                threshold = thresholds[np.argmax(tpr - fpr)]

                _auc = metrics.auc(fpr, tpr)
                _map = metrics.average_precision_score((_y > 0).float(), _fx)

                FX2[_d] = (FX[_d] > threshold).astype(float)

                drug_score['auc'][_d] = _auc
                drug_score['map'][_d] = _map

        ys.extend(y.tolist())
        fxs.extend(fx.tolist())

    d_auroc = np.mean(list(drug_score['auc'].values()))
    d_auprc = np.mean(list(drug_score['map'].values()))

    ys = np.array(ys)
    fxs = np.array(fxs)
    ys_bin = [1 if y > 0 else 0 for y in ys]

    ys_known = ys[np.array(ys_bin).astype(bool)]
    fxs_known = fxs[np.array(ys_bin).astype(bool)]

    scc = spearmanr(ys_known, fxs_known).correlation
    rmse = np.sqrt(metrics.mean_squared_error(ys_known, fxs_known))
    mae = metrics.mean_absolute_error(ys_known, fxs_known)
    fpr, tpr, thresholds = metrics.roc_curve(ys_bin, fxs, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(ys_bin, fxs)

    test_score = [scc, rmse, mae, auroc, auprc, d_auroc, d_auprc]
    test_result = (FX, FX2, drug_score)

    return test_score, test_result


def main(args):
    start_time = time()

    if len(args.session_name) > 0:
        if not os.path.isdir(args.session_name):
            os.mkdir(args.session_name)

    if not torch.cuda.is_available():
        args.device = 'cpu'

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_best = {
        'SCC': [], 'RMSE': [], 'MAE': [],
        'AUROC': [], 'AUPRC': [], 'd-AUROC': [], 'd-AUPRC': [],
    }
    fold_best['nDCG@10'] = []
    fold_best['P@1'] = []
    fold_best['P@15'] = []
    fold_best['R@1'] = []
    fold_best['R@15'] = []

    global Y

    dataset = Dataset()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    kfold = KFold(args.folds, random_state=args.seed, shuffle=True)
    test_fold = np.zeros((750, 994))
    if args.split == 'drug':
        for k, (train_data, test_data) in enumerate(kfold.split(np.arange(750))):
            test_fold[test_data] = k

    else:
        test_fold = test_fold.reshape(-1)

        known = np.where(Y.reshape(-1) > 0)[0]
        for k, (train_data, test_data) in enumerate(kfold.split(np.arange(known.shape[0]))):
            test_fold[known[test_data]] = k

        unknown = np.where(Y.reshape(-1) == 0)[0]
        for k, (train_data, test_data) in enumerate(kfold.split(np.arange(unknown.shape[0]))):
            test_fold[unknown[test_data]] = k

        test_fold = test_fold.reshape((750, 994))


    for k in range(args.folds):
        test_mask = (test_fold == k)
        train_mask = np.logical_and(test_fold != k, test_fold >= 0)

        if len(args.session_name) > 0:
            perf = {
                'train_mask': train_mask,
                'test_mask': test_mask,
                'alpha': np.zeros((args.n_boost)),
                'sample_weight': np.zeros((args.n_boost, 750, 994)),
                'pred': np.zeros((args.n_boost * 2, 750, 994)),
                'true': np.zeros((args.n_boost * 2, 750, 994)),
                'train_score': {
                    'scc': np.zeros((args.n_boost * 2)),
                    'rmse': np.zeros((args.n_boost * 2)),
                    'mae': np.zeros((args.n_boost * 2)),
                    'auc': np.zeros((args.n_boost * 2)),
                    'map': np.zeros((args.n_boost * 2)),
                },
                'test_score': {
                    'scc': np.zeros((args.n_boost * 2)),
                    'rmse': np.zeros((args.n_boost * 2)),
                    'mae': np.zeros((args.n_boost * 2)),
                    'auc': np.zeros((args.n_boost * 2)),
                    'map': np.zeros((args.n_boost * 2)),
                },
                'drug_score': {
                    'scc': np.zeros((args.n_boost * 2, 750)),
                    'rmse': np.zeros((args.n_boost * 2, 750)),
                    'mae': np.zeros((args.n_boost * 2, 750)),
                    'auc': np.zeros((args.n_boost * 2, 750)),
                    'map': np.zeros((args.n_boost * 2, 750)),
                },
                'drug_vec': np.zeros((750, args.embed_dim)),
                'side_vec': np.zeros((994, args.embed_dim)),
            }


        print('{}> Fold {} <{}'.format('=' * 34, k+1, '=' * 34))

        sampler = np.ones((750, 994))
        sampler *= train_mask.astype(float)
        sampler /= sampler.sum()

        estimators = []
        estimator_weights = []

        model = Model(device=args.device, drug_features=dataset.len_features(), embed_dim=args.embed_dim, dropout=args.dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_criterion = 0
        best = [0, np.inf, np.inf, 0, 0, 0, 0]
        # SCC, RMSE, MAE, AUROC, AUPRC, d-AUROC, d-AUPRC

        epochs = args.n_boost * args.epochs

        for epoch in range(epochs):
            train_loss = train(model, data_loader, sampler, optimizer)
            train_score, train_result = test(model, data_loader, train_mask)

            print('  Epoch {}{}[Fold {}]'.format(epoch+1, ' '*57, k+1))
            print('    Train:   SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
                train_score[0], train_score[1], train_score[2]))
            print('             AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
                train_score[3], train_score[4], train_score[5], train_score[6]))

            test_score, test_result = test(model, data_loader, test_mask)

            print('    Test:    SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
                test_score[0], test_score[1], test_score[2]))
            print('             AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
                test_score[3], test_score[4], test_score[5], test_score[6]))
            print('')


            if (epoch + 1) % args.epochs == 0:
                FX, FX2, _, DV, SV = train_result
                T = np.logical_or(
                    np.logical_and(Y > 0, FX2 > 0),
                    np.logical_and(Y == 0, FX2 == 0)
                    )

                E = np.abs(Y - FX)

                if args.mode == 'classification':
                    e = (~T * sampler).sum()

                    if e > .5:
                        continue

                    alpha = np.log((1. - e) / e) / 2
                    estimator_weight = alpha
                    dw = np.exp(-alpha) * T + np.exp(alpha) * ~T

                    if args.drug_boost:
                        dw = np.tile(dw.mean(axis=1), (sampler.shape[1], 1)).T

                elif args.mode == 'regression':
                    l_array = np.abs(Y - FX)
                    l_array /= np.max(l_array)

                    e = (l_array * sampler).sum()

                    if e > .5:
                        continue

                    lr = 1.
                    beta = e / (1. - e)
                    estimator_weight = lr * np.log(1.0 / beta)

                    dw = beta ** (1. - l_array)

                    if args.drug_boost:
                        dw = np.tile(dw.mean(axis=1), (sampler.shape[1], 1)).T

                model = model.cpu()
                model.device = 'cpu'
                estimators.append(model)
                estimator_weights.append(estimator_weight)

                if args.copy_boost:
                    model = copy.deepcopy(model)
                else:
                    model = Model(device=args.device, drug_features=dataset.len_features(), embed_dim=args.embed_dim, dropout=args.dropout).to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                _train_score, _train_result = boost(estimators, estimator_weights, data_loader, train_mask, mode=args.mode)
                _test_score, _test_result   = boost(estimators, estimator_weights, data_loader, test_mask, mode=args.mode)

                star = ''
                if best_criterion < _train_score[4]:
                    best_criterion = _train_score[4]
                    best = _test_score
                    star = '*'

                    if len(args.session_name) > 0:
                        torch.save(model.state_dict(), '{}/model_{}.pt'.format(args.session_name, k+1))


                print('  (Boost{})'.format(star))
                print('    Train:   SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
                    _train_score[0], _train_score[1], _train_score[2]))
                print('             AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
                    _train_score[3], _train_score[4], _train_score[5], _train_score[6]))

                print('     Test:   SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
                    _test_score[0], _test_score[1], _test_score[2]))
                print('             AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
                    _test_score[3], _test_score[4], _test_score[5], _test_score[6]))


                FX, FX2, drug_score = _test_result
                test_drugs = list(drug_score['scc'].keys())

                Y2 = (Y > 0).astype(np.int_)

                K = 15
                ndcgs = np.zeros((K, len(test_drugs)))
                precisions = np.zeros((K, len(test_drugs)))
                recalls = np.zeros((K, len(test_drugs)))

                for d in test_drugs:
                    sides = np.argsort(FX[d])[::-1][:K]

                    dcg = [0] * K
                    idcg = [0] * K
                    precision = [0] * K
                    recall = [0] * K

                    for i in range(K):
                        s = sides[i]

                        dcg[i] = Y2[d][s] / np.log2(i + 2)
                        if i < np.count_nonzero(Y2[d]):
                            idcg[i] = 1 / np.log2(i + 2)

                        tp = 0
                        if FX2[d][s] == 1 and Y2[d][s] == 1:
                            tp = 1
                        precision[i] = tp
                        recall[i] = tp

                    for i in np.arange(1, K):
                        dcg[i] += dcg[i - 1]
                        idcg[i] += idcg[i - 1]
                        precision[i] += precision[i - 1]
                        recall[i] += recall[i - 1]

                    ndcg = [0] * K
                    for i in range(K):
                        ndcg[i] = dcg[i] / idcg[i]
                        precision[i] /= (i + 1)

                        if np.count_nonzero(Y2[d] * test_mask[d]) > 0:
                            recall[i] /= np.count_nonzero(Y2[d] * test_mask[d])

                        di = test_drugs.index(d)

                        ndcgs[i][di] = ndcg[i]
                        precisions[i][di] = precision[i]
                        recalls[i][di] = recall[i]


                ndcg_10 = ndcgs[9].mean()
                p_1 = precisions[0].mean()
                p_15 = precisions[14].mean()
                r_1 = recalls[0].mean()
                r_15 = recalls[14].mean()

                print('             nDCG@10={:.3f}, P@1={:.3f}, P@15={:.3f}, R@1={:.3f}, R@15={:.3f};'.format(
                    ndcg_10, p_1, p_15, r_1, r_15))

                best.extend([ndcg_10, p_1, p_15, r_1, r_15])
                print('')


                if len(args.session_name) > 0:
                    i = len(estimators) - 1

                    perf['alpha'][i] = estimator_weight
                    perf['sample_weight'][i] = sampler.copy()

                    perf['pred'][i] = test_result[0]
                    perf['pred'][args.n_boost+i] = _test_result[0]

                    FX2 = np.logical_or(train_result[1], test_result[1])
                    perf['true'][i] = np.logical_or(
                        np.logical_and(Y, FX2),
                        np.logical_and(Y == 0, FX2 == 0)
                        ).astype(np.int_)

                    FX2 = np.logical_or(_train_result[1], _test_result[1])
                    perf['true'][args.n_boost+i] = np.logical_or(
                        np.logical_and(Y, FX2),
                        np.logical_and(Y == 0, FX2 == 0)
                        ).astype(np.int_)

                    for j, metric in enumerate(['scc', 'rmse', 'mae', 'auc', 'map']):
                        perf['train_score'][metric][i] = train_score[j]
                        perf['train_score'][metric][args.n_boost+i] = _train_score[j]
                        perf['test_score'][metric][i] = test_score[j]
                        perf['test_score'][metric][args.n_boost+i] = _test_score[j]

                        for d in train_result[2][metric]:
                            perf['drug_score'][metric][i][d] = train_result[2][metric][d]

                        for d in test_result[2][metric]:
                            perf['drug_score'][metric][i][d] = test_result[2][metric][d]

                        for d in _train_result[2][metric]:
                            perf['drug_score'][metric][args.n_boost+i][d] = _train_result[2][metric][d]

                        for d in _test_result[2][metric]:
                            perf['drug_score'][metric][args.n_boost+i][d] = _test_result[2][metric][d]

                    perf['drug_vec'] = DV
                    perf['side_vec'] = SV

                    with open('{}/perf_{}.pkl'.format(args.session_name, k+1), 'wb') as f:
                        pickle.dump(perf, f)


                sampler *= dw
                sampler /= sampler.sum()

                global dd_sim, ss_sim

                dd_sim = torch.from_numpy(np.matmul(DV, DV.T))
                ss_sim = torch.from_numpy(np.matmul(SV, SV.T))


        print('    Best:    SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
            best[0], best[1], best[2]))
        print('             AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
            best[3], best[4], best[5], best[6]))
        print('             nDCG@10={:.3f}, P@1={:.3f}, P@15={:.3f}, R@1={:.3f}, R@15={:.3f};'.format(
            best[7], best[8], best[9], best[10], best[11]))

        print('')

        for i, e in enumerate(fold_best.keys()):
            fold_best[e].append(best[i])


    print('{}> Done. <{}'.format('=' * 34, '=' * 34))
    print('    Best:     SCC={:.4f}, RMSE={:.4f}, MAE={:.4f};'.format(
        np.mean(fold_best['SCC']),
        np.mean(fold_best['RMSE']),
        np.mean(fold_best['MAE']),
        ))
    print('              AUROC={:.3f}, AUPRC={:.3f}, d-AUROC={:.3f}, d-AUPRC={:.3f};'.format(
        np.mean(fold_best['AUROC']),
        np.mean(fold_best['AUPRC']),
        np.mean(fold_best['d-AUROC']),
        np.mean(fold_best['d-AUPRC']),
        ))
    print('              nDCG@10={:.3f}, P@1={:.3f}, P@15={:.3f}, R@1={:.3f}, R@15={:.3f};'.format(
        np.mean(fold_best['nDCG@10']),
        np.mean(fold_best['P@1']),
        np.mean(fold_best['P@15']),
        np.mean(fold_best['R@1']),
        np.mean(fold_best['R@15']),
        ))

    end_time = time()

    hours = int((end_time - start_time) / 3600)
    minutes = int(((end_time - start_time) % 3600) / 60)
    seconds = int(((end_time - start_time) % 3600) % 60)

    elapsed = '{}h{:02d}m{:02d}s'.format(hours, minutes, seconds)
    print('    Elapsed:  {}'.format(elapsed))


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    parser = argparse.ArgumentParser(description='my model')
    parser.add_argument('--epochs', type=int, default=50, metavar='N')
    parser.add_argument('--n_boost', type=int, default=10, metavar='N')
    parser.add_argument('--split', type=str, default='random', metavar='STRING')
    parser.add_argument('--drug_boost', action='store_true', default=False)
    parser.add_argument('--copy_boost', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='FLOAT')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N')
    parser.add_argument('--dropout', type=float, default=.125, metavar='FLOAT')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N')
    parser.add_argument('--mode', type=str, default='classification', metavar='STRING')
    parser.add_argument('--folds', type=int, default=10, metavar='N')
    parser.add_argument('--seed', type=int, default=42, metavar='N')
    parser.add_argument('--session_name', type=str, default='', metavar='STRING')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='STRING')
    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='FLOAT')
    args = parser.parse_args()
    main(args)