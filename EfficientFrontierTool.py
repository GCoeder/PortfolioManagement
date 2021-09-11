# Code was adapted from https://towardsdatascience.com/python-markowitz-optimization-b5e1623060f5 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize



class EfficientFrontier:
    
    """
    A class to find the efficient portfolio from the Efficient Frontier model. CAPM is not used for expected returns, the annualized return is. 
    """
    
    def __init__(self, stocks, seed=1234, num_portfolios=6000, minWeight = 0.03, maxWeight = 0.65):
        """
        Creates the Efficient Frontier object and initializes the values. 

        Parameters
        ----------
            stocks : pandas.DataFrame
                Each series should be the daily closing price of an assets time series data. The data is converted to log returns. 
            seed : int 
                Seed to be used for repreducible code. DEFAULT:1234
            num_portfolios : int
                The number of portfolios to simulate.
            MinWeight : float
                The minimum weight any asset can be assigned. DEFAULT:0.03
            maxWeight : float
                The maximum weight any asset can be assigned. DEFAULT:0.65
                
        Methods
        --------
        plot_frontier()
        best() - returns list of best portfolios weights
        """
            
        self.stocks = stocks
        self.num_portfolios = num_portfolios
        self.seed = seed
        self.minWeight = minWeight
        self.maxWeight = maxWeight
        
        # Calculate log returns
        self.log_ret = np.log(stocks/stocks.shift(1))
        
        self.do_frontier()
        
    def do_frontier(self):
        
        """
        Uses the objects properties, set in the __init__() function, to calculate each portfolios Sharpe Ratio. 

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        np.random.seed(self.seed)
        stocks = self.stocks
        num_ports = self.num_portfolios
        all_weights = np.zeros((num_ports, len(stocks.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        for x in range(num_ports):
            # Weights            
            weights = np.array(np.random.random(len(stocks.columns)))
            weights = weights/np.sum(weights)
            # save weights
            all_weights[x,:] = weights
            ret_arr[x] = np.sum( (self.log_ret.mean() * weights * 252))
            vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov()*252, weights)))
            sharpe_arr[x] = ret_arr[x]/vol_arr[x]
            max_sr_ret = ret_arr[sharpe_arr.argmax()]
            max_sr_vol = vol_arr[sharpe_arr.argmax()]
            
        self.vol_arr = vol_arr
        self.sharpe_arr = sharpe_arr
        self.ret_arr = ret_arr
        self.max_sr_vol = max_sr_vol
        self.max_sr_ret = max_sr_ret
        
    def plot_frontier(self):
        
        plt.figure(figsize=(12,8))
        plt.scatter(self.vol_arr, self.ret_arr, c=self.sharpe_arr, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.scatter(self.max_sr_vol, self.max_sr_ret,c='red', s=50) # red dot
        plt.show()
        
        
    def best(self):
        
        """
        Calculates and returns a list of the best portfolio weights, according to the Efficient Portfolio Frontier and model parameters. 

        Parameters
        ----------
        None

        Returns
        -------
        List of optimal wieghts
        """
        
        def check_sum(weights):
            #return 0 if sum of the weights is 1            
            return np.sum(weights)-1
        
        num_stocks = len(self.stocks.columns)
        cons = ({'type':'eq', 'fun':check_sum})
        init_guess = [ 1.0/num_stocks for s in range(0,num_stocks) ]
        
        bounds = ()
        for s in range(0,num_stocks):
            bounds = bounds + ( (self.minWeight,self.maxWeight), )
            
        opt_results = minimize(self.__neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints = cons)
        
        weights = opt_results.x
        
        return pd.Series(weights, index=self.stocks.columns)
    
    def __get_ret_vol_sr(self, weights):
        weights = np.array(weights)
        ret = np.sum(self.log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov()*252, weights)))
        sr = ret/vol
        return np.array([ret, vol, sr])
    
    def __neg_sharpe(self, weights):
        # the number 2 is the sharpe ratio index from the get_ret_vol_sr        
        return self.__get_ret_vol_sr(weights)[2] * -1