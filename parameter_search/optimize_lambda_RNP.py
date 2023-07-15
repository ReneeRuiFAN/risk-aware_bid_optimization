import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import copy
from scipy.special import lambertw
import random
import multiprocessing as mp
import time
import os


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


class RNP_bid_policy:
    def __init__(self, v, B, c_hat, w_hat, sigma_w_hat, lam='0', max_price='0'):
        self.lam = lam if type(lam) is float else 1
        self.max_price = max_price if type(max_price) is float else 300
        self.optlam = 1
        self.v = v
        self.budget = B
        self.c_hat = c_hat
        self.w_hat = w_hat
        self.sigma_w_hat = sigma_w_hat

    def bid(self, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        bid = np.divide(self.v * self.c_hat, lam+1)
        bid = np.round(bid, 1)
        loss_left, _ = self.loss(np.array([0] * bid.shape[0]), lam)
        loss_right, _ = self.loss(
            np.array([self.max_price] * bid.shape[0]), lam)
        loss_der, _ = self.loss(bid, lam)
        bid_list = list(map(self.selector, loss_left,
                        loss_right, loss_der, bid))
        return np.array(bid_list)

    def loss(self, bid, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        term1 = - self.v * self.c_hat * \
            norm.cdf((bid-self.w_hat)/self.sigma_w_hat)
        lam_list = [lam] * len(self.w_hat)
        exp = self.w_hat*norm.cdf((bid-self.w_hat)/self.sigma_w_hat) - \
            self.sigma_w_hat * \
            norm.pdf((bid-self.w_hat)/self.sigma_w_hat)
        shortage = exp - self.budget
        loss = term1 + (lam+1)*exp - lam*self.budget
        # shortage is constraint, when shortage smaller than 0, the constaint is satisfied
        return loss, shortage

    def selector(self, loss_left, loss_right, loss_der, bid):
        if loss_left < loss_right and loss_left < loss_der:
            b = 0
        elif loss_right < loss_left and loss_right < loss_der:
            b = 300
        else:
            b = bid
        return b

    def get_lam(self, sample_size='0'):
        if type(sample_size) is int:
            sample_size = sample_size
            samples = np.random.randint(len(self.w_hat), size=sample_size)
            c_hat = self.c_hat[samples]
            w_hat = self.w_hat[samples]
            sigma_w_hat = self.sigma_w_hat[samples]
        else:
            c_hat = self.c_hat
            w_hat = self.w_hat
            sigma_w_hat = self.sigma_w_hat

        lam_min = 0
        lam_max = 20
        shortage = 1
        while lam_max - lam_min > 1e-2 or shortage > 1e-2:
            lam = 0.5 * (lam_min + lam_max)
            bid = self.bid(lam)
            _, short = self.loss(bid, lam)
            shortage = np.mean(short)
            print('lam:{}, shortage:{:.4f}, mean of bid:{:.2f}, max of bid:{}'.format(
                lam, shortage, np.mean(bid), max(bid)))
            if shortage > 0:
                lam_min = lam
            else:
                lam_max = lam
        self.optlam = lam_max
        print('The optimal lam is {}'.format(self.optlam))

        return self.optlam

    def optlam(self):
        return self.optlam


def load_estimator():
    w_hat_train = np.loadtxt('w_hat_train.csv')
    c_hat_train = np.loadtxt('c_hat_train.csv')
    sigma_train = np.loadtxt('sigma_w_hat.csv')

    # # create valid and test dataset
    # w_hat_set = np.loadtxt('w_hat_test.csv')
    # c_hat_set = np.loadtxt('c_hat_test.csv')
    # sigma_set = np.loadtxt('sigma_w_hat_test.csv')

    # valide = int(w_hat_set.shape[0]/2)

    # w_hat_valid = w_hat_set[:valide]
    # c_hat_valid = c_hat_set[:valide]
    # sigma_valid = sigma_set[:valide]

    # w_hat_test = w_hat_set[valide:]
    # c_hat_test = c_hat_set[valide:]
    # sigma_w_hat_test = sigma_set[valide:]
    return w_hat_train, c_hat_train, sigma_train


def real_data():
    # load real data
    w_hat_set = np.loadtxt('w_hat_test.csv')
    valide = int(w_hat_set.shape[0]/2)

    win_train = np.loadtxt('win_train.csv')
    win_valid = np.loadtxt('win_test.csv')[:valide]
    win_test = np.loadtxt('win_test.csv')[valide:]

    c_train = np.loadtxt('c_train.csv')
    c_valid = np.loadtxt('c_test.csv')[:valide]
    c_test = np.loadtxt('c_test.csv')[valide:]

    batch_size = 10000
    budget0 = np.sum(win_valid)
    sumOpportunity = c_test.shape[0]
    print('Original budget: ', budget0)
    max_price = max(win_train)
    print('max price: ', max_price)
    sumMP = np.sum(win_train)
    sumClick = np.sum(c_train)
    v = sumMP/sumClick
    print("From the train set, we have: sumMP:{}, sumClick:{}, v:{}, count:{}".format(
        sumMP, sumClick, v, win_train.shape[0]))
    return budget0, v, sumOpportunity


def run(budget_scale):
    w_hat_train, c_hat_train, sigma_train = load_estimator()
    budget0, v, sumOpportunity = real_data()
    avgBudget = budget_scale*budget0/sumOpportunity
    model = RNP_bid_policy(v=v, B=avgBudget, c_hat=c_hat_train,
                           w_hat=w_hat_train, sigma_w_hat=sigma_train)
    lam = model.get_lam()
    bid = model.bid(optlam=lam)
    filename = './bid_policy/RNP_bid_policy_'+str(budget_scale)+'.csv'
    np.savetxt(filename, bid)
    return lam


def main():
    set_seed(10)
    budget_scale_list = [1. / 2**n for n in range(7)]
    budget0, v, sumOpportunity = real_data()
    avgBudget_list = np.array(budget_scale_list) * budget0/sumOpportunity

    tic = time.time()
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    pool = mp.Pool(processes=ncpus)
    lam_list = pool.map(run, budget_scale_list)
    pool.close()
    toc = time.time()
    print('Done in {:.4f} seconds with multiprocessing'.format(toc-tic))

    df = pd.DataFrame({'budget_scale': budget_scale_list,
                       'avgBudget': avgBudget_list,
                       'lam': lam_list})
    print(df)
    df.to_csv('./optlams/lam_RNP.csv')


if __name__ == '__main__':
    main()
