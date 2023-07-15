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
    def __init__(self, v, B, c_hat, w_hat, sigma_w_hat, lam='0', alpha='0', max_price='0'):
        self.lam = lam if type(lam) is float else 1
        self.max_price = max_price if type(max_price) is float else 300
        self.optlam = 1
        self.v = v
        self.budget = B
        self.c_hat = c_hat
        self.w_hat = w_hat
        self.sigma_w_hat = sigma_w_hat
        self.alpha = alpha if type(alpha) is float else 1e-6

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
        while lam_max - lam_min > 1e-2 or shortage > 1e-2:
            lam = 0.5 * (lam_min + lam_max)
            bid = self.bid(lam)
            _, short = self.loss(bid, lam)
            shortage = np.mean(short)
            print('lam:{}, shortage:{}, mean of bid:{}, max of bid:{}'.format(
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


class Evaluate:
    def __init__(self, bid, w, c, v, B, batch_size='0'):
        self.batch_size = batch_size if type(batch_size) is int else 10000
        self.bid = bid
        self.v = v
        self.budget = B
        self.c = c
        self.w = w
        self.totalBudget = B * bid.shape[0]
        self.batchBudget = B * self.batch_size
        self.rev_batch = []
        self.profit_batch = []
        self.cost_batch = []
        self.clkbatch = []
        self.srbatch = []
        self.stopfreq = 0

    def __call__(self):
        print('start evaluating...')
        impSum = 0
        clkSum = 0
        e = []
        clkbatch = 0
        costSum = 0
        b = 0
        batchCost = 0
        srbatch = 0
        count = 0
        firstout = 0
        for i in range(self.bid.shape[0]):
            if self.totalBudget-costSum > 0:
                if count % self.batch_size == 0:
                    if count != 0:
                        self.cost_batch.append(batchCost)
                        self.clkbatch.append(clkbatch)
                        self.srbatch.append(srbatch)
                        self.rev_batch.append(self.v*clkbatch)
                        self.profit_batch.append(self.v*clkbatch-batchCost)
                        
                    batchCost = 0
                    clkbatch = 0
                    srbatch = 0
                    firstout = 0
                if self.batchBudget-batchCost > self.w[i]:
                    if self.batchBudget-batchCost < self.bid[i]:
                        self.bid[i] = self.batchBudget-batchCost
                        firstout += 1
                        if firstout == 1:
                            self.stopfreq += 1
                    if self.bid[i] >= self.w[i]:
                        impSum += 1
                        srbatch += 1
                        clkSum += self.c[i]
                        clkbatch += self.c[i]
                        e.append(self.w[i])
                        costSum += self.w[i]
                        batchCost += self.w[i]
                    else:
                        e.append(0)
                    b += self.bid[i]
                else:
                    e.append(0)
                    firstout += 1
                    if firstout == 1:
                        self.stopfreq += 1
                        #print("run out of budget for this batch, stop bidding at: ", count)
                count += 1
            else:
                print('run out of total budget')
                break

        if clkSum != 0:
            ecpc = costSum / clkSum
        else:
            ecpc = 'inf'
    #     if impSum != 0:
    #         ctr = clkSum / float(impSum)
    #     else:
    #         ctr = 'inf'
        #print('total bidding times: {} in the total opportunity:{}'.format(len(e), self.bid.shape[0]))
        rev = self.v * clkSum
        profit = rev - costSum
        roi = clkSum / costSum
        self.sr = impSum / self.bid.shape[0]
        er = costSum / self.bid.shape[0]
        br = b / self.bid.shape[0]
        end = len(e)-len(e) % self.batch_size
        e_array = np.array([e[j:j+self.batch_size]
                           for j in range(0, end, self.batch_size)])
        self.e_avg = np.mean(e_array, axis=1)
        # change to on full batch
        print('ROI:{}, Revenue:{}, Profit:{}, ecpc:{}, clicks:{}, impressions:{}, success rate:{:.4f}, average expense:{:.2f}, average bidding price:{:.2f}'.format(
            roi, rev, profit, ecpc, clkSum, impSum, self.sr, er, br))
        # return self.cvar, self.ce, self.e_avg, self.rev_batch, self.profit_batch, self.cost_batch

    def stop_freq(self):
        return self.stopfreq

    def sharpe_rev(self):
        s = np.mean(self.rev_batch)/np.std(self.rev_batch)
        return s

    def sharpe_profit(self):
        s = np.mean(self.profit_batch)/np.std(self.profit_batch)
        return s

    def cvar_profit(self, percentile='0'):
        self.percentile = percentile if type(percentile) is float or int else 5
        x_1 = sorted(self.profit_batch)[0]
        x_2 = sorted(self.profit_batch)[1]
        cvar = (1/30*x_1+(0.05-1/30)*x_2)/0.05
        return cvar

    def cvar_cost(self, percentile='0'):
        self.percentile = percentile if type(
            percentile) is float or int else 95
        x_1 = sorted(self.cost_batch)[29]
        x_2 = sorted(self.cost_batch)[28]
        cvar = (1/30*x_1+(0.05-1/30)*x_2)/0.05
        return cvar

    def cvar_loss(self, percentile='0'):
        self.percentile = percentile if type(
            percentile) is float or int else 95
        var = np.percentile(self.profit_batch, self.percentile)
        cvar = self.profit_batch[self.profit_batch >= var].mean()
        return cvar

    def ce(self, alpha='0'):
        self.alpha = alpha if type(alpha) is float else 0.1
        ce = (1/self.alpha) * \
            np.log(1-np.mean(1-np.exp(self.alpha*np.array(self.e_avg))))
        return ce

    def CI(self, t, obj):
        if obj == 'rev':
            obj = self.rev_batch
        elif obj == 'profit':
            obj = self.profit_batch
        elif obj == 'cost':
            obj = self.cost_batch
        elif obj == 'avg_cost':
            obj = self.e_avg
        tmp = t * np.std(obj)/np.sqrt(len(obj))
        upper = np.mean(obj) + tmp
        lower = np.mean(obj) - tmp
        return upper, lower
    
    def realize_CR(self, obj):
        if obj == 'profit':
            obj = self.profit_batch
        elif obj == 'cost':
            obj = self.cost_batch
        
        obj = sorted(obj)
        upper = obj[28]
        lower = obj[1]
        return upper, lower
    
    def expense(self):
        return self.e_avg

    def revenue(self):
        return self.rev_batch

    def profit(self):
        return self.profit_batch
    
    def srbatch(self):
        return self.srbatch
        
    def clkbatch(self):
        return self.clkbatch

    def exp_profit(self):
        return np.mean(self.profit_batch)

    def std_profit(self):
        return np.std(self.profit_batch)

    def exp_cost(self):
        return np.mean(self.cost_batch)

    def std_cost(self):
        return np.std(self.cost_batch)

    def cost_batch(self):
        return self.cost_batch


def load_estimator():
    # w_hat_train = np.loadtxt('w_hat_train.csv')
    # c_hat_train = np.loadtxt('c_hat_train.csv')
    # sigma_train = np.loadtxt('sigma_w_hat.csv')

    # create valid and test dataset
    w_hat_set = np.loadtxt('w_hat_test.csv')
    c_hat_set = np.loadtxt('c_hat_test.csv')
    sigma_set = np.loadtxt('sigma_hat_test.csv')

    valide = int(w_hat_set.shape[0]/2)

    w_hat_valid = w_hat_set[:valide]
    c_hat_valid = c_hat_set[:valide]
    sigma_valid = sigma_set[:valide]

    # w_hat_test = w_hat_set[valide:]
    # c_hat_test = c_hat_set[valide:]
    # sigma_w_hat_test = sigma_set[valide:]
    return w_hat_valid, c_hat_valid, sigma_valid


def real_data(budget_scale):
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
    avgBudget = budget0*budget_scale/sumOpportunity
    max_price = max(win_train)
    print('max price: ', max_price)
    sumMP = np.sum(win_train)
    sumClick = np.sum(c_train)
    v = sumMP/sumClick
    # print("From the train set, we have: sumMP:{}, sumClick:{}, v:{}, count:{}".format(
    #     sumMP, sumClick, v, win_train.shape[0]))
    return avgBudget, v, c_valid, win_valid


def load_lam(budget_scale):
    df_RNP = pd.read_csv('./optlams/lam_RNP.csv')
    df_RAP = pd.read_csv('./optlams/lam_RAP_'+str(int(1/budget_scale))+'.csv')
    alpha_list = df_RAP['alpha']
    return df_RNP, df_RAP, alpha_list


def bidClose_alpha(v, c, w, c_hat, w_hat, sigma_w_hat, alpha, B, lamdata_RA, lamdata_RN, RA_policy='close', RN_policy='close'):
    lam_list = []
    stop_freq = []
    exp_profit = []
    cvar_profit = []
    sr = []
    clk = []
    std_profit = []
    sharpe_profit = []
    exp_cost = []
    std_cost = []

    for a in alpha:
        if a != 0:
            print('Running evaluation when alpha is {}, average budget is {}, and value per click is {}'.format(
                a, B, v))
            lam = lamdata_RA.lam[lamdata_RA['alpha'] == a].item()
            model = RA_policy(v=v, B=B, c_hat=c_hat, w_hat=w_hat,
                              sigma_w_hat=sigma_w_hat, alpha=a)

        if a == 0:
            print('Running evaluation when alpha is {}, average budget is {}, and value per click is {}'.format(
                a, B, v))
            lam = lamdata_RN.lam[lamdata_RN['avgBudget'] == B].item()
            model = RN_policy(v=v, B=B, c_hat=c_hat, w_hat=w_hat,
                              sigma_w_hat=sigma_w_hat, alpha=a)

        bid = model.bid(lam)
        model = Evaluate(bid=bid, w=w, c=c, v=v, B=B, batch_size=10000)
        model()

        lam_list.append(lam)
        sharpe_profit.append(model.sharpe_profit())
        cvar_profit.append(model.cvar_profit())
        stop_freq.append(model.stop_freq())
        exp_profit.append(model.exp_profit())
        std_profit.append(model.std_profit())

    df = pd.DataFrame({'alpha': alpha,
                       'lam': lam_list,
                       'stop_freq': stop_freq,
                       'exp_profit': exp_profit,
                       'std_profit': std_profit,
                       'cvar_profit': cvar_profit,
                       'sharpe_profit': sharpe_profit})
    return df

def run(budget_scale):
    avgBudget, v, c_valid, win_valid = real_data(budget_scale)
    df_RNP, df_RAP, alpha_list = load_lam(budget_scale)
    w_hat_valid, c_hat_valid, sigma_valid = load_estimator()

    df = bidClose_alpha(v=v,
                        c=c_valid,
                        w=win_valid,
                        c_hat=c_hat_valid,
                        w_hat=w_hat_valid,
                        sigma_w_hat=sigma_valid,
                        alpha=alpha_list,
                        B=avgBudget,
                        lamdata_RA=df_RAP,
                        lamdata_RN=df_RNP,
                        RA_policy=RAP_bid_policy,
                        RN_policy=RNP_bid_policy)

    df.to_csv('./sharpe/RAP_'+str(int(1/budget_scale))+'.csv')
    print(df)


def main():
    set_seed(10)

    budget_scales = [1. / 2**n for n in range(7)]

    tic = time.time()
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    pool = mp.Pool(processes=ncpus)
    pool.map(run, budget_scales)
    pool.close()
    toc = time.time()
    print('Done in {:.4f} seconds with multiprocessing'.format(toc-tic))

if __name__ == '__main__':
    main()

