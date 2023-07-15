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


class RAP_bid_policy:
    def __init__(self, v, B, c_hat, w_hat, sigma_w_hat, lam='0', alpha='0', max_price='0'):
        self.lam = lam if type(lam) is float else 1
        self.max_price = max_price if type(max_price) is float else 300
        self.optlam = 1
        self.v = v
        self.budget = B
        self.c_hat = c_hat
        self.w_hat = w_hat
        self.sigma_w_hat = sigma_w_hat
        self.alpha = alpha
        self.gamma_2 = - self.alpha * self.budget
        self.gamma_1 = 0.5 * np.power(self.alpha*self.sigma_w_hat, 2) + \
            self.alpha*self.w_hat - self.alpha * self.budget
        # when gamma is large, mk new variable log of c_2; L'hopital rule
        self.c_1 = self.v * self.c_hat + self.lam * \
            np.exp(self.gamma_2) - self.w_hat
        self.c_2 = self.lam * np.exp(np.clip(self.gamma_1, 0, 700))
        self.c_3 = self.w_hat + self.alpha * self.sigma_w_hat**2

    def bid(self, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        bid = - lambertw(self.alpha * lam * np.exp(np.clip(self.alpha*(self.v*self.c_hat+lam *
                         np.exp(self.gamma_2)-self.budget), 0, 700)))/self.alpha + self.v*self.c_hat + lam*np.exp(self.gamma_2)
        bid = np.round(bid.real, 1)
        loss_left, _ = self.loss(np.array([0] * bid.shape[0]), lam)
        loss_right, _ = self.loss(
            np.array([self.max_price] * bid.shape[0]), lam)
        loss_der, _ = self.loss(bid, lam)
        bid_list = list(map(self.selector, loss_left,
                        loss_right, loss_der, bid))
        return np.array(bid_list)

    def loss(self, bid, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        term1 = - self.c_1 * norm.cdf((bid-self.w_hat)/self.sigma_w_hat)
        lam_list = [lam] * len(self.w_hat)
        term2 = np.array(
            list(map(self.get_term2, bid, self.w_hat, self.sigma_w_hat, self.gamma_1, self.c_2, self.c_3, lam_list)))
        term3 = - lam*(1-np.exp(self.gamma_2))
        term4 = - self.sigma_w_hat * \
            norm.pdf((bid-self.w_hat)/self.sigma_w_hat)
        loss = term1 + term2 + term3 + term4
        shortage = (-lam*np.exp(self.gamma_2) *
                    norm.cdf((bid-self.w_hat)/self.sigma_w_hat)+term2+term3)/lam
        return loss, shortage

    def get_term2(self, bid, w_hat, sigma_w_hat, gamma_1, c_2, c_3, lam):
        if (bid-c_3)/sigma_w_hat > -30:
            term2 = c_2 * norm.cdf((bid-c_3)/sigma_w_hat)
        else:
            z = (bid-c_3)/sigma_w_hat
            term2 = lam * np.exp(gamma_1+self.logphiLhopital(z))
        return term2

    def logphiLhopital(self, z):
        y = - np.log(np.sqrt(2*np.pi)) - 0.5 * np.power(z, 2) - np.log(-z)
        return y

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

        if self.alpha < 0.1:
            lam_min = 0
            lam_max = 200
        else:
            lam_min = 0
            lam_max = 20
        shortage = 1
        while lam_max - lam_min > 1e-2 or shortage > 0:
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
    budget_scale = 1/16
    avgBudget = budget0*budget_scale/sumOpportunity
    max_price = max(win_train)
    print('max price: ', max_price)
    sumMP = np.sum(win_train)
    sumClick = np.sum(c_train)
    v = sumMP/sumClick
    print("From the train set, we have: sumMP:{}, sumClick:{}, v:{}, count:{}".format(
        sumMP, sumClick, v, win_train.shape[0]))
    return avgBudget, v


def run(alpha):
    avgBudget, v = real_data()
    w_hat_train, c_hat_train, sigma_train = load_estimator()
    model = RAP_bid_policy(v=v, B=avgBudget, c_hat=c_hat_train,
                           w_hat=w_hat_train, sigma_w_hat=sigma_train, alpha=alpha)
    lam = model.get_lam()
    return lam


def main():
    set_seed(10)
    alpha_list = np.arange(0.02, 1.02, 0.02)

    tic = time.time()
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    pool = mp.Pool(processes=ncpus)
    lam_list = pool.map(run, alpha_list)
    pool.close()
    toc = time.time()
    print('Done in {:.4f} seconds with multiprocessing'.format(toc-tic))

    budget_list = [1/16]*len(alpha_list)

    df = pd.DataFrame({'budget': budget_list,
                       'alpha': alpha_list,
                       'lam': lam_list})
    print(df)
    df.to_csv('./optlams/lam_RAP_'+str(16)+'.csv')


if __name__ == '__main__':
    main()
