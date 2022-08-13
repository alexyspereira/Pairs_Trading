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
from functools import partial
from statsmodels.tsa.adfvalues import mackinnonp,mackinnoncrit
pd.set_option("display.max_column",999)




class OLS(object):
    def __init__(self,fit_intercept=True,):
        self.fit_intercept=fit_intercept
        self.missing = 'drop'
        pass
    
    @staticmethod
    def get_variables_name(X,label='X'):
        if type(X) is pd.Series: 
            return [X.name] if X.name is not None else [f'{label}0']
        elif type(X) is pd.DataFrame:
            return X.columns.values.tolist()
        else:
            if len(X.shape)>1:
                return [f'{label}{i}' for i in range(X.shape[1])]
            else:
                return [f'{label}0']    
    
    def fit(self,X,y):
        self.X = X.copy()
        self.y = y.copy()
        self.xnames = self.get_variables_name(X)
        self.ynames = self.get_variables_name(y,label='y')
        self._prepare_data()
        self.estimate_coeff()
        self.error()
        self.compute_rsquared()
        self.coeff_standard_error()
        self.compute_pvalues()
        self.compute_aic_bic()
        self.build_coeff_table()
        self.build_metrics_table()
        self.summary()
        
        
    
    def _prepare_data(self):
        y_shape = self.y.shape if len(self.y.shape)> 1 else (len(self.y),1)
        x_shape = self.X.shape if len(self.X.shape) > 1 else (len(self.X),1)
      
        self.y_input = self.y if type(self.y) is np.ndarray else self.y.values
        self.X = self.X if type(self.X) is np.ndarray else self.X.values
        
        self.y_input = self.y_input.reshape(y_shape)
        self.X = self.X.reshape(x_shape)

        self.df = self.X.shape[0] - self.X.shape[1] - 1
        
        if self.fit_intercept:
            self.X_input = np.hstack( (np.ones(shape = (self.X.shape[0],1)), self.X) )
        else:
            self.X_input=self.X.copy()

        
        self.X_input = self.X_input.transpose()
        self.y_input = self.y_input.transpose()
        pass
    
    
    def estimate_coeff(self):
        self.X_inv = np.linalg.pinv(self.X_input)

        self.X_X_inv = np.linalg.pinv(np.matmul(self.X_input,self.X_inv))
        self.params = np.matmul(self.y_input, np.matmul(self.X_inv,self.X_X_inv))
        
        if self.fit_intercept:
            self.intercept = self.params[:,0]
            self.coefs = self.params[:,1:]
        else:
            self.intercept = np.array([0])
            self.coefs = self.params
            
    
    def predict(self,X):
        
        return np.matmul(self.coefs,X.transpose())+self.intercept.reshape(-1,1)
        
    def compute_residuals(self,X,y):
        return y-self.predict(X).reshape(y.shape)
    
    def coeff_standard_error(self):
        self.cov_resid_matrix = np.cov(self.residual,rowvar=True).reshape(self.y_input.shape[0], self.y_input.shape[0])
        self.cov_betas=np.linalg.pinv(np.matmul(self.X_input,self.X_input.T))
        self.coeffs_se = np.kron(self.cov_betas,self.cov_resid_matrix)
        self.coeffs_se = np.sqrt(np.diag(self.coeffs_se)).reshape( self.params.shape[::-1]).T
    
    def compute_pvalues(self):
        self.tvalues = self.params/self.coeffs_se
        self.pvalues = 2*(1-stats.t.cdf(np.abs(self.tvalues),self.df))
    
    def compute_aic_bic(self):
        det = np.linalg.det(self.cov_resid_matrix)
        nparams=len(self.params.flatten())
        self.aic = np.log( det )+ 2*nparams/self.X.shape[0]
        self.bic = np.log( det)+ np.log(self.X.shape[0])*nparams/self.X.shape[0]
    
    def error(self):
        self.yhat = self.predict(self.X)
        self.residual = self.y_input-self.yhat
        self.ymean = self.y_input.mean(axis=1).reshape(-1,1)
        
        
        self.sse = np.power(self.residual,2).sum()
        self.tss = np.power(self.y_input - self.ymean,2).sum()
        self.ess = np.power(self.yhat - self.ymean,2).sum()
    
    def compute_rsquared(self):
        self.rsquared = 1 - self.sse/self.tss
        self.adj_rsquared = 1 - self.sse/self.tss * ( len(self.y)-1  )/self.df
    

        
    def build_coeff_table(self):
        
        self.variable_names = self.xnames.copy()
        self.variable_names = ['Constant']+self.variable_names  if self.fit_intercept else self.variable_names
        
        self.params_t = pd.DataFrame(self.params.T.tolist(), columns = self.ynames).assign(Variable=self.variable_names,Statistic="Beta")
        self.params_se_t = pd.DataFrame(self.coeffs_se.T.tolist(),columns = self.ynames).assign(Variable=self.variable_names,Statistic='SE')       
        self.tvalues_t = pd.DataFrame(self.tvalues.T.tolist(),columns=self.ynames).assign(Variable=self.variable_names,Statistic='T-Stat')
        self.pvalues_t = pd.DataFrame(self.pvalues.T.tolist(), columns = self.ynames).assign(Variable=self.variable_names,Statistic='P-Value')
        
        self.table_params = pd.concat([self.params_t, self.params_se_t,self.tvalues_t,self.tvalues_t,self.pvalues_t])
        
        self.table_params = self.table_params.pivot_table(index='Variable',columns='Statistic')
        self.table_params=self.table_params.reindex(index= self.variable_names,)

    
    def build_metrics_table(self):
        depvariables = ",".join(self.ynames)
        out= {
            'Dependent Variable':depvariables,
            'R Squared': self.rsquared,
            'Adj. R Squared':self.adj_rsquared,
            'N Obs': self.X.shape[0],
            'Degrees of Freedom':self.df,
            'AIC':self.aic,
            'BIC':self.bic
        }
        
        self.metric_table = pd.Series(out).to_frame()
        self.metric_table.columns=['Metric Value']

    def summary(self):
        self.table_params_out = self.table_params.applymap('{:.3f}'.format)
        self.metric_table_out = self.metric_table.copy()
        self.metric_table_out.iloc[1:,:] = self.metric_table_out.iloc[1:,:].applymap('{:.3f}'.format)
        
        
class ADF(OLS):
    def __init__(self, add_constant=True, maxlags = None):
        super().__init__()        
#         self.add_constant=add_constant
        self.maxlags=maxlags
        self.fit_intercept=add_constant
        self.reg='c' if self.fit_intercept else 'n'
        self.partial_mackinnon_p = np.vectorize(partial(mackinnonp, regression = self.reg))
    
    def evaluate(self, X):
        self.X  = X
        self.nobs=len(X)
        self.compute_diff_shift()
        self.maxlags =  12*(self.X.shape[0]/100)**(1/4) if self.maxlags is None else self.maxlags
        self.maxlags = int(self.maxlags)
        self.results=[]
        for lag in range(self.maxlags+1):
            lr = OLS(fit_intercept=self.fit_intercept)
            #x,y=self.prepare_data(lag)
            lr.fit(*self.prepare_data(lag))
            aic=lr.aic            
            lr.pvalues = self.partial_mackinnon_p(lr.tvalues)
            lr.summary()
            self.results.append([lag,lr,aic,])
        
        self.collect_best_results()
        self.test_results()
    
    def compute_diff_shift(self):
        self.series_shift = np.roll(self.X,shift=1)[1:]
        self.series_diff = np.diff(self.X)  
    @staticmethod
    def add_lags_series(X,nlags):
        df=[ pd.Series(np.roll(X,shift=lag)[lag:],index=np.arange(lag, X.shape[0])) for lag in range(0,nlags+1)]
        df=pd.concat(df,axis=1)
        df.columns = [f'dX(-{i})' for i in range(0,nlags+1)]                       
        return df.iloc[:,1:]
        
    def prepare_data(self,nlags,):
        if nlags>0:
            out = self.add_lags_series(self.series_diff,nlags)      
            out['X(-1)']=self.series_shift[out.index]
        else:
            out=pd.Series(self.series_shift,name='X(-1)').to_frame()
            
        
        out['dX']=self.series_diff
        out=out.dropna()
        y = out['dX'].copy()
        out=out.drop(columns=['dX'])
                              
        return out,y
    
    def collect_best_results(self):
        self.best_result = sorted(self.results,key = lambda x: x[-1])[0]
        self.best_lag=self.best_result[0]
        self.best_equation=self.best_result[1]       
        self.best_aic=self.best_result[2]
        self.unit_root_tstat = self.best_equation.tvalues.flatten()[-1]
        self.unit_root_pvalue = self.best_equation.pvalues.flatten()[-1]        
        self.H0 = r"Series has unit root"
        self.critical_values = mackinnoncrit(nobs=self.nobs,regression=self.reg)
        self.reject_null =  np.where(self.critical_values<self.unit_root_tstat, "Not Reject","Reject")
        self.is_stationary = np.where(self.critical_values<self.unit_root_tstat, "Non Stationary","Stationary")
    
    def test_results(self):
        self.summary_test_results = {
                                "Null Hypothesis":[self.H0 ],
                                "T-Statistic":[self.unit_root_tstat],
                                "P-Value (MacKinnon)":[self.unit_root_pvalue],
            
                                "Optimal Lag":[self.best_lag],
                       
                                "Confidence Level":["1%","5%","10%"],
                                "Mackinnon Critical Value":self.critical_values,
                                "Reject/Not Reject H0":self.reject_null,
                                "Stationary/Non Stationary":self.is_stationary,
                                
                               }
        
        self.summary_test_results =pd.DataFrame.from_dict(self.summary_test_results,orient='index').replace([None],'')
        self.summary_test_results.columns=['', '', '']
        
        
class EngleGranger(object):
    def __init__(self):
        pass
    
    def fit(self,X1,X2):
        self.X1=X1.copy()
        self.X2=X2.copy()
        self.ols1 = OLS()
        self.ols1.fit(self.X1,self.X2)
        self.spread1 = self.ols1.residual.flatten()
        self.adf1=ADF()
        self.adf1.evaluate(spread1)
        
        self.ols2 = OLS()
        self.ols2.fit(self.X2,self.X1)
        self.spread2 = self.ols2.residual.flatten()
        self.adf2=ADF()
        self.adf2.evaluate(self.spread2)
        
        if abs(self.adf1.unit_root_tstat)>=abs(self.adf2.unit_root_tstat):
            self.best_ols = self.ols1
            self.best_adf = self.adf1
            self.spread=self.spread1
            self.X_best= self.X1.copy()
            self.y_best= self.X2.copy()
        else:
            self.best_ols = self.ols2
            self.best_adf = self.adf2
            self.spread=self.spread2
            self.X_best= self.X2.copy()
            self.y_best= self.X1.copy()
        
        self.cointegration_params = dict(zip(self.best_ols.variable_names,self.best_ols.params.flatten()))
        self.cointegration_params = pd.Series(self.cointegration_params)
        self.fit_ecm()
        self.fit_ou_process()
        self.summarize_results()
                
            
    def fit_ecm(self):
        X_diff = np.diff(self.X_best).reshape(-1,1)
        self.y_diff = np.diff(self.y_best).flatten()
        spread_shift = np.roll(self.spread,shift=1)[1:].reshape(-1,1)
        self.X_ecm = np.concatenate([X_diff,spread_shift],axis=1)
        self.X_ecm = pd.DataFrame(self.X_ecm,columns = ['dX', 'Residual(-1)']) 
        self.y_diff = pd.Series(self.y_diff,name='dY')
        self.ecm_ols = OLS(fit_intercept=False)
        self.ecm_ols.fit(self.X_ecm,self.y_diff)
    
    
    
    def fit_ou_process(self):
        spread_shift = np.roll(self.spread,shift=1)[1:].reshape(-1,1)
        self.ar = OLS()
        self.ar.fit(spread_shift, self.spread[1:])
        
        ar_const,ar_beta = [x for x in self.ar.params.flatten()]       
        self.tau=1/252
        self.theta = -np.log(ar_beta)/self.tau
        self.half_life= np.log(2)/self.theta/self.tau
        self.mue = ar_const/(1-ar_beta)      
        sse = (self.ar.residual**2).sum()
        self.sigma=(sse*self.tau/(1-np.exp(-2*self.theta*self.tau)))**0.5
        self.ou_process_params = {
            'mue':self.mue,
            'H':self.half_life,
            'sigma':self.theta,           
            'theta':self.theta,
        }
        

    def summarize_results(self):
        self.compare_ols = pd.concat([self.ols1.table_params_out.reset_index(),
           self.ols2.table_params_out.reset_index()],axis=1)
        
        self.compare_ols = pd.concat([self.ols1.table_params_out.reset_index(),
           self.ols2.table_params_out.reset_index()],axis=1)
        eq1_str = self.ols1.ynames[0] + ' = ' + ' + '.join(self.ols1.variable_names[::-1])
        eq2_str = self.ols2.ynames[0] + ' = ' + ' + '.join(self.ols2.variable_names[::-1])
        
        self.adf1.summary_test_results.columns=['Equation 1:',eq1_str,'']
        self.adf2.summary_test_results.columns=['Equation 2:',eq2_str,'']
        self.compare_adf = pd.concat([self.adf1.summary_test_results,self.adf2.summary_test_results],axis=1)
        self.ecm_results_table = self.ecm_ols.table_params_out.copy()
        self.ou_process_summary = {
            'OU Process Parameters':None,
            r'$\mu_e$':self.mue,
            r'Half-Life (days)':self.half_life,
            r'$\sigma$':self.sigma,           
            r'$\theta$':self.theta,
        }
        self.ou_process_summary =pd.Series(self.ou_process_summary).to_frame().replace([None,np.nan],'')
        self.ou_process_summary.columns=['Value']
            
