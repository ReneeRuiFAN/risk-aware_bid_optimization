import numpy as np
from scipy.stats import norm

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
        lam_max = 40
        
        lam_list = []
        bid_mean = []
        cost_mean = []
        
        while lam_max - lam_min > 1e-2 or shortage > 1e-2:
            lam = 0.5 * (lam_min + lam_max)
            bid = self.bid(lam)
            _, short = self.loss(bid, lam)
            shortage = np.mean(short)
            print('lam:{}, shortage:{}, mean of bid:{}, max of bid:{}'.format(
                lam, shortage, np.mean(bid), max(bid)))
            
            lam_list.append(lam)
            bid_mean.append(np.mean(bid))
            cost_mean.append(shortage+self.budget)
            
            if shortage > 0:
                lam_min = lam
            else:
                lam_max = lam
        self.optlam = lam_max
        print('The optimal lam is {}'.format(self.optlam))

        return self.optlam, lam_list, bid_mean, cost_mean

    def optlam(self):
        return self.optlam