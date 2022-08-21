

import os
import glob
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader.data as web
import itertools
import datetime as dt
from collections import Counter
from sklearn.linear_model import LinearRegression
from scipy import stats
pd.set_option("display.max_column",999)



class PairTradingBacktest(object):
	def __init__(self,dates, X,Y, coint_coeffs,  mue, sigma_eq, zscores,xy_label=['X','Y'], tol=0.0):

		"""
		Class that backtests the pair trading strategy based on cointegration by generating signals of entry and exit points

		Arguments:
		Date pd.seeries/ array with dates that will be evaluated
		X array = independent variable of the cointegration linear regression
		Y array = dependent variable of the cointegration linear regression
		coint_coeffs list  = betas estimated from the cointegration linear regression expected [1, beta]
		mue = long run mean
		sigma_eq = equation sd deviation from OU process
		zscores = array of z scores to be evaluated
		xy_label = [X variable name, y variable name]
		tol = tolerance in percentage terms to consider when spread is close to long run mean to exit the position

		returns:

		back_test_result: backtest result table for period considered and all z scores levels
		summary_Table: summary statistics compute on backtest results for all z scores


		"""      
		self.dates=dates
		self.X = X
		self.Y = Y
		self.xy_label = xy_label
		self.coint_coeffs = coint_coeffs if type(coint_coeffs) is np.ndarray else coint_coeffs
		self.coint_weights = np.array([1, -coint_coeffs[-1]])
		self.xy_label = xy_label
		self.mue = mue
		self.sigma_eq=sigma_eq
		self.zscores = zscores
		
		self.bounds = [ [z,mue-z*sigma_eq, mue+z*sigma_eq ] for z in zscores ]
	   
		self.tol=tol
		
		#computes the spread
		self.spread = Y - coint_coeffs[-1]*X - coint_coeffs[0]
		
		pass
	
	
	@staticmethod
	def numdays(x):
		return pd.Timedelta(x.values[-1]-x.values[0],).days

	@staticmethod
	def annualized_return(x):
		return np.power( (1+x).prod(), 252/x.notnull().sum())-1

	@staticmethod
	def annualized_vol(x):
		return np.sqrt(x.var()*252)
	
	@staticmethod
	def cumulative_return(x):
		return (1+x).prod() - 1

	
	def signal(self, spread,lower,upper,mu,current_position=0,tol=0.0):
		"""function that computes the signal based on spread
		1 = [1,-beta]
		-1 = [-1,beta]

		arguments:
		spread: the spread of the series 
		lower: lower bound to decide if enter or exits position
		upper: upper bound to decide if enter or exits position
		current_position: [-1,0,1] to signal if there is an open position
		tol: tolerance in percentage to decide what is close to mu_e

		"""

		#if there is an open position = 1
		if current_position==1:
			#if spread is larger than mu_e then exits the position otherwise holds
			if spread >= mu*(1-tol):
				return 1, 'exit'
			else:
				return 1, 'hold'
		#if there is a open position is -1
		elif current_position==-1:

			#if spread is lower than mu, then exits the position, otherwise holds
			if spread <= mu*(1+tol):
				return -1, 'exit'
			else:
				return -1, 'hold'
		#if tbere are no open positions in the book
		else:
			#if the spread is higher than the upper bound then we want to enter in the position [-1,beta]
			if spread>=upper:
				return -1,'enter'
			#if spread is lower than lower bound then we want to enter in the position [1,-beta]
			elif spread <= lower:
				return 1,'enter'
			#otherwise no position
			else:
				return 0,"-"

	def compute_signal(self,lower,upper):
		
		"""function that loops through spread array applyting the compute signal"""
		
		#dictionary to store esults
		signals = {'day':[],
				   'signal':[],
				   'action':[],
				   'trade_id':[]
				  }

		signals['day'] = self.dates

		#assumes that there are no position in the book
		current_position=0
		trade_id=0
		for i,e in enumerate(self.spread,1):
			#get signal and action from signal function
			s,a = self.signal(e,lower,upper,self.mue,current_position=current_position,tol=self.tol)

			#replaces no signal for null
			s = np.nan if s==0 else s

			#if reach the end of the spread and there is an open position, closes the position
			if i==len(self.spread) and a=='hold':
				s = signals['signal'][-1]*-1
				a = 'exit'

			signals['signal'].append(s)
			signals['action'].append(a)

			#updates trade id if there is an enter action
			trade_id = trade_id+1 if a=='enter' else trade_id
			
			#replaces trade-id for null if there is not action
			trade_id_temp = np.nan if a =='-' else trade_id
			signals['trade_id'].append(trade_id_temp)

			#updates current position to zero if the action if exit
			current_position= 0 if a == 'exit' else s

	
		
		signals=pd.DataFrame(signals)
		signals['lower_bound']=lower
		signals['upper_bound']=upper
		
		return signals
		
		
	
	def backtest(self):

		""" Generate the  backtest results for the strategies for the assets """


		output=[]
		xlabel,ylabel = self.xy_label
		
		#gets the assets series
		x_series = self.X if type(self.X) is pd.Series else pd.Series(self.X)
		y_series = self.Y if type(self.Y) is pd.Series else pd.Series(self.Y)
		x_series_sign_shift = np.sign(x_series.shift(1))		
		y_series_sign_shift = np.sign(y_series.shift(1))


		#computes daily returns
		x_series_daily_returns = (x_series.pct_change()*x_series_sign_shift).fillna(0) 
		y_series_daily_return = (y_series.pct_change()*y_series_sign_shift).fillna(0) 
		
		#cointegration weights
		coint_weights= self.coint_weights.reshape(1,-1)
		
		#loops through all the zscore levels
		for i,(z,lower,upper) in enumerate(self.bounds,1):

			#calling compute signal functiopn to generate signals and trade enty,exit points
			signals_df = self.compute_signal(lower,upper)

			#getting the raw asset
			signals_df[f'{xlabel}'] = self.X
			signals_df[f'{ylabel}'] = self.Y

			#asset daily returns
			signals_df[f'{xlabel}_daily_return'] = x_series_daily_returns
			signals_df[f'{ylabel}_daily_return'] = y_series_daily_return

			#trade signals 1,-1           
			trading_signal = signals_df['signal'].values.reshape(-1,1)

			#computes trade daily return by using dot product of signal, cointegration weights multiply by x,y daily returns
			signals_df['trade_daily_return'] = (np.dot(trading_signal, coint_weights)*signals_df[[f'{ylabel}_daily_return',f'{xlabel}_daily_return']]).sum(axis=1)
			

			signals_df['trade_daily_return'] = np.where(signals_df.signal.isnull(),np.nan,signals_df['trade_daily_return'])

			#adding ids to the strategy
			signals_df['bounds_id'] = i
			signals_df['zscore'] = z
			signals_df['spread']=self.spread
			signals_df['mue']=self.mue

			output.append(signals_df)
		

		#concatenate all the results
		self.back_test_results = pd.concat(output).reset_index(drop=True)

		#drops strategies with no trades
		self.back_test_results['flag_all_null'] =self.back_test_results.groupby("bounds_id")['signal'].transform(lambda x: x.isnull().mean())
		f=self.back_test_results['flag_all_null']<1.0
		self.back_test_results=self.back_test_results.loc[f,:].copy().reset_index(drop=True) 


		#compoutes some summary statistics
		self.back_test_results['total_period_day']=self.back_test_results.groupby("bounds_id")['day'].transform(lambda x: self.numdays(x))

		self.back_test_results['strategy_daily_vol']=self.back_test_results.groupby("bounds_id")['trade_daily_return'].transform('std')
		self.back_test_results['strategy_daily_avg_return']=self.back_test_results.groupby("bounds_id")['trade_daily_return'].transform('mean')

		self.back_test_results['strategy_annualized_return']=self.back_test_results.groupby("bounds_id")['trade_daily_return'].transform(self.annualized_return)
		self.back_test_results['strategy_annualized_vol']=self.back_test_results.groupby("bounds_id")['trade_daily_return'].transform(self.annualized_vol)
		self.back_test_results['strategy_cumulative_return']=self.back_test_results.groupby("bounds_id")['trade_daily_return'].transform(self.cumulative_return)
		
		return self.back_test_results
	
	
	def summary_tables(self):
		"""Generates summary tables with P&L metrics based on thre back test table"""

		#dictionary with aggregate functions
		dict_agg = {
		'trade_daily_return':[self.cumulative_return,self.annualized_return,self.annualized_vol],
		'day':[self.numdays,],
		'signal':'max',
		'lower_bound':'min',
		'upper_bound':'max',
		'total_period_day':'max',
		'strategy_daily_vol':'max',
		'strategy_daily_avg_return':'max',
		'strategy_annualized_return':'max',
		'strategy_annualized_vol':'max',
		'strategy_cumulative_return':'max'


		}

		#summarizes backtest results computing statistics
		self.trade_analysis = self.back_test_results.groupby(['bounds_id', 'trade_id','zscore']).agg(dict_agg)
		self.trade_analysis.columns=['_'.join(c) if type(dict_agg[c[0]])==list else c[0]  for c in self.trade_analysis.columns ]
		self.trade_analysis.reset_index(inplace=True)
		self.summary = self.trade_analysis.\
		groupby(['bounds_id']).\
		agg({'zscore':'max',
		'trade_id':'max',
		"strategy_annualized_return":'max',
		"strategy_annualized_vol": 'max',
		"strategy_cumulative_return": 'max',     
		'day_numdays':['mean','sum'],})

		self.summary.reset_index(inplace=True)

		self.summary.columns=['_'.join(c) for c in self.summary.columns]

		#renames column for output
		col_names = ['Strategy ID', 
			 'Z-Score',
			 '# Trades',
			 'Annualized Return (%)', 
			 'Annualized Volatility (%)',
			 'Cumulative Return (%)',
			 'Avg # Days Open Position',
			'Total Trading Days']

		self.summary.columns=col_names

		self.summary[['Annualized Return (%)', 'Annualized Volatility (%)','Cumulative Return (%)',]]=self.summary[['Annualized Return (%)', 'Annualized Volatility (%)','Cumulative Return (%)',]]*100
		self.summary
		return self.summary
	
	
	def plot_strategy(self, strategy_id,figsize=(15,7.5)):

		"""Plots the trading and spread for visual analysis of the strategies implement given the strategy_id"""
		temp = self.back_test_results.query(f"bounds_id == {strategy_id} ").copy()
		strategy_id = temp.bounds_id.max()
		zscore = temp.zscore.max()
		title=f'Pairs Trading {self.xy_label[0]} x {self.xy_label[1]} with Z-Score = {zscore}'
		fig= plt.figure(figsize=(15,7.5))
		plt.plot(temp.day,temp.mue,color='black',linestyle='--',label= r'Long Run Mean $\mu_e $')
		plt.plot(temp.day,temp.upper_bound,color='blue',label='Upper/Lower Bounds')
		plt.plot(temp.day,temp.lower_bound,color='blue')
		plt.scatter(temp.day,np.where(temp.action=='enter',1,np.nan)*temp.spread,color='red',marker="X",s=100,label="Entry Point")
		plt.scatter(temp.day,np.where(temp.action=='exit',1,np.nan)*temp.spread,color='blue',marker="^",s=100,label="Exit Point")
		plt.plot(temp.day,temp.spread,color='gray',linestyle='-',alpha=0.75,label='Spread')
		plt.legend(loc='best',fontsize=12,)
		plt.ylabel("Spread")
		plt.xlabel("Date")
		plt.title(title,fontsize=14)
		plt.show()
		return fig
		

		
	



class TradeAnalyzer(object):
	
	def __init__(self,return_series_input, mkt_ticker, rf_ticker='^TYX',historical_returns=None):


		"""
		Class to compute rolling statistics such as rolling beta, alpha, sharpe ratio, VaR and ES given markte risk factor and risk free weightss

		"""
		self.return_series_input=return_series_input
		self.mkt_ticker=mkt_ticker
		self.rf_ticker=rf_ticker
		self.return_series=None
		self.historical_returns=historical_returns
		pass
	
	
	def compute_beta(self,X,y):
		lr=sm.OLS(y, X, missing='drop').fit()
		params=lr.params
		pvalues=lr.pvalues
		beta,alpha=params[1],params[0]
		beta_pval,alpha_pval=pvalues[1],pvalues[0]

		return beta,alpha,beta_pval,alpha_pval


	@staticmethod
	def compute_sharpe(return_series):
		f=np.where(np.isnan(return_series),False,True )
		return return_series[f].mean()*252/np.sqrt(return_series[f].var()*252)


	def rolling_beta(self,return_series):
		f=return_series.trade_daily_return.notnull()   
		rolling_series=[]
		rolling_obj = return_series.loc[f,:].rolling(120,)

		for g in rolling_obj:
			if len(g)>=120:
				x = sm.add_constant(g['excess_mkt_return'].values.reshape(-1,1))
				y = g['trade_daily_return'].values.reshape(-1,1)
				beta,alpha,beta_pval,alpha_pval = self.compute_beta(x,y)
				rolling_series.append([g.index.max(), beta,alpha,beta_pval,alpha_pval])
			else:
				rolling_series.append([g.index.max(), np.nan,np.nan,np.nan,np.nan])

		rolling_series = pd.DataFrame(rolling_series, columns=['day','rolling_beta_6M','rolling_alpha_6M','rolling_beta_pval_6M','rolling_alpha_pval_6M'],)
		rolling_series.set_index("day",inplace=True)
		return_series = pd.concat([return_series,rolling_series],axis=1,)
		return_series['rolling_beta_6M']=return_series['rolling_beta_6M'].fillna(method='ffill')
		return_series['rolling_alpha_6M']=return_series['rolling_alpha_6M'].fillna(method='ffill')
		return_series['rolling_beta_pval_6M']=return_series['rolling_beta_pval_6M'].fillna(method='ffill')
		return_series['rolling_alpha_pval_6M']=return_series['rolling_alpha_pval_6M'].fillna(method='ffill')

		return return_series
			
	@staticmethod
	def rolling_sharpe(return_series):
		f=return_series.iloc[:,0].notnull()
		_excess_daily_return = return_series.loc[f,'excess_daily_return'].copy()
		_daily_return = return_series.loc[f,'excess_daily_return'].copy()

		rolling_mean=_excess_daily_return.rolling(120,min_periods=30).mean() 
		rolling_std=_excess_daily_return.rolling(120,min_periods=30).std() 

		return_series.loc[f,'rolling_sharpe_6M'] = rolling_mean/rolling_std
		return_series['rolling_sharpe_6M'] = return_series['rolling_sharpe_6M'].fillna(method='ffill')
		return_series['rolling_sharpe_6M'] = return_series['rolling_sharpe_6M']*252/(252**0.5)

		return return_series

	@staticmethod
	def compute_profit_loss(return_series):
		f=return_series.notnull()
		_return_series=return_series[f].copy()
		output=pd.DataFrame(index=_return_series.index)

		output['cumulative_return'] = (_return_series+1).cumprod() - 1
		output['total_return'] = (_return_series+1).prod() - 1
		output['annualized_return'] = np.power(1+output['total_return'], 252/f.sum())-1
		output['daily_vol'] = _return_series.std()  
		output['annualized_vol'] = output['daily_vol']*np.sqrt(252)
		output['daily_avg_return'] = _return_series.mean()

		current_return = 1+output['cumulative_return']
		cummax_return = (1+output['cumulative_return']).cummax()
		output['drawdown'] = (cummax_return-current_return)/cummax_return    
		return output

	@staticmethod
	def combine_data(return_series,mkt,risk_free):
		return_series = pd.merge(return_series, mkt['mkt_return'],left_index=True,right_index=True,how='left' )
		return_series = pd.merge(return_series, risk_free['avg_rf_daily'],left_index=True,right_index=True,how='left' )
		return_series['cum_trade_daily_return']=(return_series['trade_daily_return'].fillna(0)+1).cumprod()-1
		return_series['cum_mkt_return']=(return_series['mkt_return'].fillna(0)+1).cumprod()-1
		return_series['excess_daily_return']=return_series['trade_daily_return'].fillna(method='ffill')-return_series['avg_rf_daily'].fillna(method='ffill')
		return_series['excess_mkt_return']=return_series['mkt_return'].fillna(method='ffill')-return_series['avg_rf_daily'].fillna(method='ffill')

		return return_series
	
	@staticmethod
	def value_at_risk(mean,std,conf_interval,time_horizon):
		return -mean*time_horizon + stats.norm.ppf(conf_interval)*std*np.sqrt(time_horizon)
	
	@staticmethod
	def expected_shortfall(mean,std,conf_interval,time_horizon):
		return -mean*time_horizon + std*np.sqrt(time_horizon)*stats.norm.pdf(stats.norm.ppf(conf_interval))/(1-conf_interval)
		

	def compute_var(self,return_series, min_period=30,rolling_window = 252, conf_intervals=0.99, time_horizon=1 ):
		f=return_series.notnull()
		rolling_mean = return_series[f].rolling(window=rolling_window,min_periods=min_period).mean()
		rolling_std = return_series[f].rolling(window=rolling_window,min_periods=min_period).std()

		zscore = stats.norm.ppf(conf_intervals)
		rolling_var = self.value_at_risk(rolling_mean,rolling_std, conf_intervals,time_horizon)
		rolling_es = self.expected_shortfall(rolling_mean,rolling_std, conf_intervals,time_horizon)
		
		rolling_var = rolling_var.shift(1)
		rolling_es=rolling_es.shift(1)
		return rolling_var, rolling_es

	@staticmethod
	def get_market_risk_free(mkt_ticker = 'SPY', rf_ticker='^TYX'):
		mkt = web.DataReader(mkt_ticker,'yahoo',start='2001-01-01',end='2022-06-30')
		mkt['mkt_return'] = mkt.iloc[:,-1].pct_change()

		risk_free = web.DataReader("^TYX",'yahoo',start='2001-01-01',end='2022-06-30')
		risk_free.head()

		risk_free['daily_return'] = risk_free.iloc[:,-1].pct_change()
		risk_free['avg_rf_daily'] = (risk_free['Adj Close'].rolling(360).mean()/252)/100

		risk_free.dropna(inplace=True)

		return mkt, risk_free
	
	
	def generate_analysis(self):
		mkt,risk_free=self.get_market_risk_free('^GSPTSE')
		self.return_series=self.return_series_input.copy()
		original_index=self.return_series.index.copy()

		self.return_series_full = pd.concat([self.historical_returns,self.return_series],) if self.historical_returns is not None else self.return_series
		self.return_series_full = self.combine_data(self.return_series_full,mkt,risk_free)

		self.return_series_full['rolling_var_99_1day'],self.return_series_full['rolling_es_99_1day']= self.compute_var(self.return_series_full['trade_daily_return'])
		self.return_series_full['rolling_var_99_10day'],self.return_series_full['rolling_es_99_10day']=self.compute_var(self.return_series_full['trade_daily_return'],time_horizon=10)
		self.return_series_full=self.rolling_beta(self.return_series_full)
		self.return_series_full=self.rolling_sharpe(self.return_series_full)


		self.return_series=self.return_series_full.loc[original_index,:].copy()
		self.pl_info = self.compute_profit_loss(self.return_series_input)	
		self.return_series = pd.concat([self.return_series, self.pl_info],axis=1)
		self.return_series[self.pl_info.columns]=self.return_series[self.pl_info.columns].fillna(method='ffill')
		mean_return = self.return_series_input.mean()
		std_return = self.return_series_input.std()
		
		self.return_series['var_99_1day'] = self.value_at_risk(mean_return, std_return, 0.99,1)
		self.return_series['var_99_10day'] = self.value_at_risk(mean_return, std_return, 0.99,10)
		self.return_series['es_99_1day'] = self.expected_shortfall(mean_return, std_return, 0.99,1)
		self.return_series['es_99_10day'] = self.expected_shortfall(mean_return, std_return, 0.99,10)
		self.return_series['sharpe'] =self.compute_sharpe(self.return_series['excess_daily_return'])
		self.return_series['n_exceedances_var_99_10day'] = (self.return_series_full['trade_daily_return'] < -self.return_series_full['rolling_var_99_10day']).sum() 

		x = sm.add_constant(self.return_series['excess_mkt_return'].values.reshape(-1,1))
		y = self.return_series['trade_daily_return'].values.reshape(-1,1)		
		self.return_series['beta'],self.return_series['alpha'],self.return_series['beta_pvalue'],self.return_series['alpha_pvalue'] =self.compute_beta(x,y)


		return self.return_series


	def summary_table(self):
		cols=['annualized_return','annualized_vol','total_return', 'alpha','sharpe','beta','beta_pvalue','alpha_pvalue','drawdown','var_99_1day','var_99_10day','es_99_1day','es_99_10day',]
		self.summary=self.return_series[cols].max()*100
		self.summary['alpha']=self.summary['alpha']*252
		self.summary.index=['Annual Return','Annual Vol','Cumulative Return','Alpha (Annual)','Sharpe','Beta','Beta P-Value','Alpha P-Value','Max Drawdown','1-Day VaR 99%','10-Day VaR 99%','1-Day ES 99%','10-Day ES 99%',]
		self.summary=self.summary.to_frame().reset_index()
		self.summary.columns=['Metric','Value (%)']
		return self.summary


	@staticmethod
	def plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel):
		fig=plt.figure(figsize=figsize)
		series.plot(kind='line',label=label_series)
		if mean is not None: mean.plot(kind='line',linestyle='--',color='red', label=label_mean) 
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.legend(loc='best')
		return fig

	def plot_rolling_sharpe(self,title=None,figsize=(15,7.5)):
		series=self.return_series['rolling_sharpe_6M']*100
		mean = self.return_series['sharpe']*100
		label_series=r'Rolling Sharpe 6M (%)'
		label_mean = f"Avg Sharpe = {mean.mean():.2f}%" 
		title='Rolling Sharpe 6M' if title is None else title
		xlabel='Date'
		ylabel='Rolling Shape 6M (%)'
		return self.plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel)


	
	def plot_rolling_beta(self,title=None,figsize=(15,7.5)):
		series=self.return_series['rolling_beta_6M']
		mean = self.return_series['beta']
		label_series=r'Rolling $ \beta$ 6M'
		label_mean = r'Avg $\beta$ = '+f'{mean.mean():.2f}'
		title=r'Rolling $\beta$ 6M' if title is None else title
		xlabel='Date'
		ylabel=r'Rolling $\beta$ 6M'
		
		return self.plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel)



	def plot_rolling_alpha(self,title=None,figsize=(15,7.5)):
		series=self.return_series['rolling_alpha_6M']*252*100
		mean = self.return_series['alpha']*252*100
		label_series=r'Rolling $ \alpha$ 6M (%)'
		label_mean = r'Avg $\alpha$ = '+f'{mean.mean():.2f}'+"%"
		title=r'Rolling Annualized $\alpha$ 6M'
		xlabel='Date'
		ylabel=r'Rolling Annualized $\alpha$ 6M (%)'

		return self.plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel)

	def plot_drawdown(self,title=None,figsize=(15,7.5)):
		series = self.return_series['drawdown']*100*-1
		# mean= pd.Series(np.zeros(shape = self.return_series['drawdown'].shape).flatten(),index=series.index)
		mean=None
		label_series=r'Drawdown (%)'
		label_mean = None
		title=r'Drawdown (%)'
		xlabel='Date'
		ylabel=r'Drawdown (%)'
		return self.plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel)

	def plot_10day_var(self,title=None,figsize=(15,7.5)):
		series = self.return_series['rolling_var_99_10day']*100
		mean= self.return_series['var_99_10day']*100
		label_series=r'Rolling 10-day VaR 99% (%)'
		label_mean = r'Avg 10-day VaR 99% (%)'
		title=r'10-day VaR 99% (%)'
		xlabel='Date'
		ylabel=r'10-day VaR 99% (%)'
		return self.plot_chart(series,mean, title, label_series,label_mean,figsize,xlabel,ylabel)



	def plots(self):
		funcs = [self.plot_rolling_sharpe,self.plot_rolling_beta,self.plot_rolling_alpha,self.plot_drawdown,self.plot_10day_var]
		return {f.__name__:f() for f in funcs}
	
