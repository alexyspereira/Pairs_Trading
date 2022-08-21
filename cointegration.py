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

        """Class to fit multivariate multioutput linear regression via OLS method

            Argument: 
            fit_intercept (boolean): True computes the linear regression if constant term False constant term is set to zero

            returns

            intercept (1,n_target): if fit_intercept the value of the estimated intercept via OLS, otherwise return 0
            coefs (n_target,n_features): array with estimated coefficients via OLS
            params (n_target, n_features + 1): array with all the equations coefficients 


            table_params_out: Summary table with estimated Betas, standard error and p-valueu
            metric_table_out: Summary table with goodness of fit metrics such as R squared, AIC, BIC and degrees of freedom



        """

        self.fit_intercept=fit_intercept
        self.missing = 'drop'
        pass
    
    @staticmethod
    def get_variables_name(X,label='X'):

        #function to get variable names to build the tables
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
        """ fit method to estimate linear regression via OLS

        Arguments:
        X n.ndarry/pd.DataFrame/pd.Series (n_samples, n_features): independent/features vector

        Y n.ndarry/pd.DataFrame/pd.Series (n_samples, n_targets): dependent/target vector

        """

        #copies X and y variables
        self.X = X.copy()
        self.y = y.copy()

        #extract the X and y names
        self.xnames = self.get_variables_name(X)
        self.ynames = self.get_variables_name(y,label='y')
        
        #run the prepare data routine that transposes the X and y vectors, and drops variables with null values        
        self._prepare_data()

        #run the OLS procedure in fact to testimate coefficients
        self.estimate_coeff()

        #compute the residuals and R squared
        self.error()
        self.compute_rsquared()

        #computes the coefficients SEs, p-values and AIC/BIC
        self.coeff_standard_error()
        self.compute_pvalues()
        self.compute_aic_bic()

        #builds the summary tables for posterior analysis
        self.build_coeff_table()
        self.build_metrics_table()
        self.summary()
        
        
    
    def _prepare_data(self):
        """helpger function to tranpose the input array for posterior estimate"""

        #extracts X and y sahpes
        y_shape = self.y.shape if len(self.y.shape)> 1 else (len(self.y),1)
        x_shape = self.X.shape if len(self.X.shape) > 1 else (len(self.X),1)

        #converts input array to numpy array
        self.y_input = self.y if type(self.y) is np.ndarray else self.y.values
        self.X = self.X if type(self.X) is np.ndarray else self.X.values
        

        #reshaping the data
        self.y_input = self.y_input.reshape(y_shape)
        self.X = self.X.reshape(x_shape)

        #degree of freedom
        self.df = self.X.shape[0] - self.X.shape[1] - 1
        

        #adds a ones columns if fitting the intercept
        if self.fit_intercept:
            self.X_input = np.hstack( (np.ones(shape = (self.X.shape[0],1)), self.X) )
        else:
            self.X_input=self.X.copy()

        #tranposing for proper estimation
        self.X_input = self.X_input.transpose()
        self.y_input = self.y_input.transpose()
        pass
    
    
    def estimate_coeff(self):

        """ helper function that runs the OLS procedure and return the intercept and coefficients"""
        
        #inverts X matrix
        self.X_inv = np.linalg.pinv(self.X_input)

        #multiplies X and X' and then inverts, hence computes (XX')'
        self.X_X_inv = np.linalg.pinv(np.matmul(self.X_input,self.X_inv))

        #finally computes the betas as Betas = Y X'(XX')
        self.params = np.matmul(self.y_input, np.matmul(self.X_inv,self.X_X_inv))
        
        #assign coefficients and intercepts to auxiliar variables
        if self.fit_intercept:
            self.intercept = self.params[:,0]
            self.coefs = self.params[:,1:]
        else:
            self.intercept = np.array([0])
            self.coefs = self.params
            
    
    def predict(self,X):
        """Computes OLS estimates 

        argument: 
         X n.ndarry/pd.DataFrame/pd.Series (n_samples, n_features): independent/features vector

         returns 

         y np.ndarray (n_samples, n_targets): depedent variable(s) estimate(s)


        """
        return np.matmul(self.coefs,X.transpose())+self.intercept.reshape(-1,1)
        
    def compute_residuals(self,X,y):
        """
        computes residual based on OLS estimate

        Arguments:
        X n.ndarry/pd.DataFrame/pd.Series (n_samples, n_features): independent/features vector

        Y n.ndarry/pd.DataFrame/pd.Series (n_samples, n_targets): dependent/target vector

        """
        return y-self.predict(X).reshape(y.shape)
    
    def coeff_standard_error(self):

        """ helper function that computes coefficients standard error """
        self.cov_resid_matrix = np.cov(self.residual,rowvar=True).reshape(self.y_input.shape[0], self.y_input.shape[0])
        self.cov_betas=np.linalg.pinv(np.matmul(self.X_input,self.X_input.T))
        self.coeffs_se = np.kron(self.cov_betas,self.cov_resid_matrix)
        self.coeffs_se = np.sqrt(np.diag(self.coeffs_se)).reshape( self.params.shape[::-1]).T
    
    def compute_pvalues(self):

        """helper functions that computes p-value based on T-distribution"""
        self.tvalues = self.params/self.coeffs_se
        self.pvalues = 2*(1-stats.t.cdf(np.abs(self.tvalues),self.df))
    
    def compute_aic_bic(self):

        """helper function that computes AIC and BIC"""
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
        """helper function that builds the table_params_out table that summarizes the estimates of the equation"""

        #get variable names and adds constant if fit_intercept is true
        self.variable_names = self.xnames.copy()
        self.variable_names = ['Constant']+self.variable_names  if self.fit_intercept else self.variable_names
        
        #creates dataframe for beta, SE, t-stat and p-value
        self.params_t = pd.DataFrame(self.params.T.tolist(), columns = self.ynames).assign(Variable=self.variable_names,Statistic="Beta")
        self.params_se_t = pd.DataFrame(self.coeffs_se.T.tolist(),columns = self.ynames).assign(Variable=self.variable_names,Statistic='SE')       
        self.tvalues_t = pd.DataFrame(self.tvalues.T.tolist(),columns=self.ynames).assign(Variable=self.variable_names,Statistic='T-Stat')
        self.pvalues_t = pd.DataFrame(self.pvalues.T.tolist(), columns = self.ynames).assign(Variable=self.variable_names,Statistic='P-Value')
        
        #concatenates everything
        self.table_params = pd.concat([self.params_t, self.params_se_t,self.tvalues_t,self.tvalues_t,self.pvalues_t])
        
        #pivots the table
        self.table_params = self.table_params.pivot_table(index='Variable',columns='Statistic')

        #reindexes to be in order
        self.table_params=self.table_params.reindex(index= self.variable_names,)

    
    def build_metrics_table(self):

        """helper function to build metrics table"""


        depvariables = ",".join(self.ynames)
        
        #stores all the data into a dictionary and feeds into  pd.Series
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

        """computes the table parameters table and metrics table"""
        self.table_params_out = self.table_params.applymap('{:.3f}'.format)
        self.metric_table_out = self.metric_table.copy()
        self.metric_table_out.iloc[1:,:] = self.metric_table_out.iloc[1:,:].applymap('{:.3f}'.format)
        
        
class ADF(OLS):
    def __init__(self, add_constant=True, maxlags = None):
        """

        Augmented Dickey Fuller class, run the test for univariate time series and returns the assessment based on maximum number of lags specified selected
        via AIC criterion

        Argumemnts:
        add_constant boolean: True adds a constant term to the ADF equation specification
        maxlags int: defines the number of lags to be use, if None then max_lags = 12*(N OBS/100)^(0.25)
        X (n_obs,): time series to be evaluate

        returns

        unit_root_pvalue - p-value for ADF test
        unit_root_tstat - computed T-statistc for ADF Test
        critical_values - Mackinoon ADF T-statistic critical values
        summary_test_results - summary table with ADF test results


        """


        super().__init__()        
        self.maxlags=maxlags
        self.fit_intercept=add_constant
        self.reg='c' if self.fit_intercept else 'n'
        self.partial_mackinnon_p = np.vectorize(partial(mackinnonp, regression = self.reg))
    
    def evaluate(self, X):
        """ Run the ADF test

        Argument:
        X np.ndarray/pd.Series univariate time series


        """

        #copies the series
        self.X  = X
        self.nobs=len(X)

        #computes first difference
        self.compute_diff_shift()

        #computes or assigns the maximum number of lags
        self.maxlags =  12*(self.X.shape[0]/100)**(1/4) if self.maxlags is None else self.maxlags
        self.maxlags = int(self.maxlags)

        #runs OLS procedure for using the ADF LR especification for the number of maximum lags and stores all results
        self.results=[]
        for lag in range(self.maxlags+1):

            #create LR object
            lr = OLS(fit_intercept=self.fit_intercept)
            #x,y=self.prepare_data(lag)
            #prepares the data to add the required number of lagged differences
            lr.fit(*self.prepare_data(lag))

            #stores the relevant statistics
            aic=lr.aic            
            lr.pvalues = self.partial_mackinnon_p(lr.tvalues)
            lr.summary()
            self.results.append([lag,lr,aic,])
        
        #retrieves best results and build the table
        self.collect_best_results()
        self.test_results()
    
    def compute_diff_shift(self):
        """helper function that creates a shifted array and the first difference array"""
        self.series_shift = np.roll(self.X,shift=1)[1:]
        self.series_diff = np.diff(self.X)  


    @staticmethod
    def add_lags_series(X,nlags):
        """helper function that creates a dataframe with nlags new columns that are the lagged values of array X"""

        df=[ pd.Series(np.roll(X,shift=lag)[lag:],index=np.arange(lag, X.shape[0])) for lag in range(0,nlags+1)]
        df=pd.concat(df,axis=1)
        df.columns = [f'dX(-{i})' for i in range(0,nlags+1)]                       
        return df.iloc[:,1:]
        
    def prepare_data(self,nlags,):
        """helper function that creates a dataframe with nlags columns each referring to lag"""
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

        """helper function to collect the results based on the results of multiple OLS ran using various lags"""

        #sorted the results list based on the lowest AIC
        self.best_result = sorted(self.results,key = lambda x: x[-1])[0]
        
        #extracts the parameters from the equation with the lowest AIC
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
        """helper function to store the ADF results into a friendly format in a pandas DataFrame"""
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

        """
        Engle-Granger procedure to evaluate if two series X1 and X2 are cointegrated, and also estimates Ornstein-Uhlenbeck process parameters

        Arguments:
        X1, X2 np.ndarray/pd.Series representing univariate times series

        Returns:
        cointegration_params: the estimated parameters for cointegration
        ecm_ols: error correction model OLS object for assessment of error correction
        ou_process_params: estimated OU process parameters
        compare_adf: 

        """


        pass
    
    def fit(self,X1,X2):


        """Fit the Engle-Granger procedure on X1 and X2

        Arguments:
        X1, X2 np.ndarray/pd.Series representing univariate times series

        Returns:
        cointegration_params: the estimated parameters for cointegration
        ecm_ols: error correction model OLS object for assessment of error correction
        ou_process_params: estimated OU process parameters
        compare_adf: 

        """

        #copies both arrays
        self.X1=X1.copy()
        self.X2=X2.copy()

        #fits the two equations and evaluates if the residuals are stationary
        self.ols1 = OLS()
        self.ols1.fit(self.X1,self.X2)
        self.spread1 = self.ols1.residual.flatten()
        self.adf1=ADF()
        self.adf1.evaluate(self.spread1)
        
        self.ols2 = OLS()
        self.ols2.fit(self.X2,self.X1)
        self.spread2 = self.ols2.residual.flatten()
        self.adf2=ADF()
        self.adf2.evaluate(self.spread2)
        

        #selects the best equation based on the adf t-stat, assigns the best equation as the one that has the highest absolute t-statistic 
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
        
        #saves cointegration parameters
        self.cointegration_params = dict(zip(self.best_ols.variable_names,self.best_ols.params.flatten()))
        self.cointegration_params = pd.Series(self.cointegration_params)

        #fits error correction model
        self.fit_ecm()

        #fits OU process
        self.fit_ou_process()

        #generates summary tables
        self.summarize_results()
                
            
    def fit_ecm(self):

        """runs the error correction model for the best equation"""
        X_diff = np.diff(self.X_best).reshape(-1,1)
        self.y_diff = np.diff(self.y_best).flatten()
        spread_shift = np.roll(self.spread,shift=1)[1:].reshape(-1,1)
        self.X_ecm = np.concatenate([X_diff,spread_shift],axis=1)
        self.X_ecm = pd.DataFrame(self.X_ecm,columns = ['dX', 'Residual(-1)']) 
        self.y_diff = pd.Series(self.y_diff,name='dY')
        self.ecm_ols = OLS(fit_intercept=False)
        self.ecm_ols.fit(self.X_ecm,self.y_diff)
    
    
    
    def fit_ou_process(self):
        """estimates the OU parameters for the best equation by fitting an AR(1) process"""
        spread_shift = np.roll(self.spread,shift=1)[1:].reshape(-1,1)
        self.ar = OLS()
        self.ar.fit(spread_shift, self.spread[1:])
        
        ar_const,ar_beta = [x for x in self.ar.params.flatten()]       
        self.tau=1/252

        #computes theta, sigma, half-life and mue
        self.theta = -np.log(ar_beta)/self.tau
        self.half_life= np.log(2)/self.theta/self.tau
        self.mue = ar_const/(1-ar_beta)      
        sse = (self.ar.residual**2).sum()
        self.sigma=(sse*self.tau/(1-np.exp(-2*self.theta*self.tau)))**0.5
        self.ou_process_params = {
            'mue':self.mue,
            'H':self.half_life,
            'sigma':self.sigma,           
            'theta':self.theta,
        }
        

    def summarize_results(self):

        """creates tables witth summary results"""

        #summary table with the OLS results for both equations
        self.compare_ols = pd.concat([self.ols1.table_params_out.reset_index(),
           self.ols2.table_params_out.reset_index()],axis=1)
        
        self.compare_ols = pd.concat([self.ols1.table_params_out.reset_index(),
           self.ols2.table_params_out.reset_index()],axis=1)
        eq1_str = self.ols1.ynames[0] + ' = ' + ' + '.join(self.ols1.variable_names[::-1])
        eq2_str = self.ols2.ynames[0] + ' = ' + ' + '.join(self.ols2.variable_names[::-1])
        

        #creates table with ADF statistics for both equations
        self.adf1.summary_test_results.columns=['Equation 1:',eq1_str,'']
        self.adf2.summary_test_results.columns=['Equation 2:',eq2_str,'']
        self.compare_adf = pd.concat([self.adf1.summary_test_results,self.adf2.summary_test_results],axis=1)
        

        #extracts the ECM results
        self.ecm_results_table = self.ecm_ols.table_params_out.copy()
        
        #builds table with OU process parameter
        self.ou_process_summary = {
            'OU Process Parameters':None,
            r'$\mu_e$':self.mue,
            r'Half-Life (days)':self.half_life,
            r'$\sigma$':self.sigma,           
            r'$\theta$':self.theta,
        }
        self.ou_process_summary =pd.Series(self.ou_process_summary).to_frame().replace([None,np.nan],'')
        self.ou_process_summary.columns=['Value']
            
