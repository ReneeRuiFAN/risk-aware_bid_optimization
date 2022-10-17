import numpy as np

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
                if self.batchBudget-batchCost > 0:
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
        print('ROI:{}, Revenue:{}, Profit:{}, ecpc:{}, clicks:{}, impressions:{}, success rate:{:.4f}, average expense:{:.2f}, average bidding price:{:.2f}, early stop: {}'.format(
            roi, rev, profit, ecpc, clkSum, impSum, self.sr, er, br, self.stopfreq))
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
        elif obj == 'rev':
            obj = self.rev_batch
            
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
    
    def sr_rate(self):
        return np.mean(self.srbatch)*100/self.batch_size
        
    def clkbatch(self):
        return self.clkbatch
    
    def clk_rate(self):
        return np.mean(self.clkbatch)

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