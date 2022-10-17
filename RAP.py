import numpy as np
from scipy.stats import norm
from scipy.special import lambertw

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
        self.c_1 = self.v * self.c_hat + self.lam * np.exp(self.gamma_2) - self.w_hat
        self.c_2 = self.lam * np.exp(np.clip(self.gamma_1, 0, 700))
        self.c_3 = self.w_hat + self.alpha * self.sigma_w_hat**2

    def bid(self, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        bid = - lambertw(self.alpha * lam * np.exp(np.clip(self.alpha*(self.v*self.c_hat+lam *
                         np.exp(self.gamma_2)-self.budget), 0, 700)))/self.alpha + self.v*self.c_hat + lam*np.exp(self.gamma_2)
        bid = np.round(bid.real, 1)
        loss_left, _ = self.loss(np.array([0] * bid.shape[0]), lam)
        loss_right, _ = self.loss(np.array([self.max_price] * bid.shape[0]), lam)
        loss_der, _ = self.loss(bid, lam)
        bid_list = list(map(self.selector, loss_left, loss_right, loss_der, bid))
        return np.array(bid_list)

    def loss(self, bid, optlam='0'):
        lam = optlam if type(optlam) is float else self.lam
        term1 = - self.c_1 * norm.cdf((bid-self.w_hat)/self.sigma_w_hat)
        lam_list = [lam] * len(self.w_hat)
        term2 = np.array(
            list(map(self.get_term2, bid, self.w_hat, self.sigma_w_hat, self.gamma_1, self.c_2, self.c_3, lam_list)))
        term3 = - lam*(1-np.exp(self.gamma_2))
        term4 = - self.sigma_w_hat * norm.pdf((bid-self.w_hat)/self.sigma_w_hat)
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
        while lam_max - lam_min > 1e-2 or shortage > 1e-2:
            lam = 0.5 * (lam_min + lam_max)
            bid = self.bid(lam)
            _, short = self.loss(bid, lam)
            shortage = np.mean(short)
            print('lam:{}, shortage:{:.4f}, mean of bid:{:.2f}, max of bid:{}'.format(lam, shortage, np.mean(bid), max(bid)))
            if shortage > 0:
                lam_min = lam
            else:
                lam_max = lam
        self.optlam = lam_max
        print('The optimal lam is {}'.format(self.optlam))
    
        return self.optlam
    
    def optlam(self):
        return self.optlam
    