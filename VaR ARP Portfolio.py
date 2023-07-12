import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
import statsmodels.api as sm
import scipy.optimize as opt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from arch.univariate import arch_model


### Functions
## Find Minimum Variance Weights
def MV(cov_matrix):
    def objective(W, C):
        # calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov_matrix),W)
        #objective: min vol
        util=varp**0.5
        return util
    n=len(cov_matrix)
    # initial conditions: equal weights
    W=np.ones([n])/n                     
    ### CONSTRAINTS
    # BOUNDS
    bounds=[(0.,1.) for i in range(n)]
    # BUDGETARY CONSTRAINT
    def Cons_Budgt(W):
        Cons_Budgt=sum(W)-1
        return Cons_Budgt
    c_= ({'type':'eq', 'fun': Cons_Budgt })
    optimized=opt.minimize(objective,W,(cov_matrix),
                                      method='SLSQP',constraints=c_,bounds=bounds,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x

## Find Gaussian Value-at-Risk with 99% confidence level
def var_gaussian(r, level=1):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    return (r.mean() + z*r.std(ddof=0))


### Setup for the MV and Long/Short Portfolio
data_excel=pd.read_excel('DATASET_1.xlsx', 'MONTHLY RETURNS',usecols="B:H",header=2)
benchmark=np.array(pd.read_excel('DATASET_1.xlsx', 'MONTHLY RETURNS',usecols="I",header=2))
ret=np.array(data_excel)
benchmark_ret=benchmark[:,0]
n_factors=ret.shape[1]
l_factors=data_excel.columns
n_obs=ret.shape[0]

## Compute Covariance Matrix
cov_matrix=np.cov(ret,bias=False,rowvar=0)
print(cov_matrix)

## Compute Weights for Minimum Variance Portfolio
w_MV=MV(cov_matrix)
w_MV


### Long/Short Portfolio
## Step 1: Factor Ranking
sigma_12m=np.std(ret[n_obs-12:n_obs,:],axis=0)
sigma_24m=np.std(ret[n_obs-24:n_obs,:],axis=0)
z_sigma_12=(sigma_12m-np.mean(sigma_12m))/np.std(sigma_12m)
z_sigma_24=(sigma_24m-np.mean(sigma_24m))/np.std(sigma_24m)
z_score=(z_sigma_12+z_sigma_24)/2

## Step 2: Factor Selection
select_L=z_score<np.quantile(z_score,0.2)
select_S=z_score>np.quantile(z_score,0.8)

## Step 3: Weight for Long and Short
# Long
w_L=np.zeros((n_factors))
cov_L=np.cov(ret[:,select_L],rowvar=0)
w_L[select_L]=MV(cov_L)
# Short
w_S=np.zeros((n_factors))
cov_S=np.cov(ret[:,select_S],rowvar=0)
w_S[select_S]=MV(cov_S)

## Step 4: Build Long/Short portfolio
ret_L=np.dot(ret,w_L.T)
cov_L_port=np.cov((ret_L,benchmark_ret))
beta_L=cov_L_port[0,1]/cov_L_port[1,1]
ret_S=np.dot(ret,w_S.T)
cov_S_port=np.cov((ret_S,benchmark_ret))
beta_S=cov_S_port[0,1]/cov_S_port[1,1]
# Hedge ratio for Short Portfolio
hedge_ratio=beta_L/beta_S
# ALLOCATIONS TO LONG AND SHORT PORTFOLIOS
allocation_L=3/(1+hedge_ratio)
allocation_S=hedge_ratio*allocation_L
# BETA LONG AND SHORT ALLOCATIONS
print(beta_L*allocation_L)
print(beta_S*allocation_S)


### Value-at-Risk with 99% Confidence Level
w_L_port=(w_L*allocation_L)
w_S_port=(w_S*allocation_S)
w_port=np.sum((w_L_port,w_S_port),axis=0)/3
ret_port=ret[:,[0,1,3,6]]
w_LS_port=w_port[[0,1,3,6]]
ret_port=np.prod((ret_port,w_LS_port.T))
ret_port_1=ret_port[1:,:]
var99=np.mean(ret_port,axis=0)-2.576*np.std(ret_port_1,axis=0)
var99_port=var_gaussian(ret_port)


### Non-Parametric Monte-Carlo Simulation
## Step 1.a: Original Statistics
mean=np.mean(ret_port,axis=0)
std=np.std(ret_port,axis=0)
cov_MC=np.cov(ret_port,rowvar=0)
skew=stats.skew(ret_port)
kurt=stats.kurtosis(ret_port)
# Cholesky decomposition of the Covariance Matrix
chol=np.linalg.cholesky(cov_MC)
## Step 1.b: Simulation Setup
n_var_MC=len(cov_MC)
n_obs_MC=len(ret_port)
n_sim=1000
## Step 1.c: Storage in Matrices
out_mean_mc=np.zeros((n_sim,n_var_MC))
out_std_mc=np.zeros((n_sim,n_var_MC))
out_skew_mc=np.zeros((n_sim,n_var_MC))
out_kurt_mc=np.zeros((n_sim,n_var_MC))
var99_mc=np.zeros((n_sim,n_var_MC))
out_std_port=np.zeros((n_sim,1))
var99_port_mc=np.zeros((n_sim,1))
out_sim=np.zeros((n_sim,n_obs_MC,n_var_MC))
## Step 1.d: Simulations with Bootstrap
for i in range(n_sim):
    print(i)
    sim=resample(ret_port, replace=True)
    out_mean_mc[i,:]=np.mean(sim,axis=0).T
    out_std_mc[i,:]=(np.std(sim,axis=0)).T
    out_skew_mc[i,:]=(stats.skew(sim)).T
    out_kurt_mc[i,:]=(stats.kurtosis(sim)).T
    var99_mc[i,:]=out_mean_mc[i,:]-2.576*out_std_mc[i,:]
    out_std_port[i]=np.sqrt(np.dot(w_LS_port.T,np.dot(np.cov(sim,rowvar=0),w_LS_port))).T
    var99_port_mc[i]=np.sum(out_mean_mc[i],axis=0)-2.576*out_std_port[i]
    out_sim[i,:,:]=sim
# Plot Value-at-Risk 99% and Standard Deviation of the Portfolio
plt.plot(var99_port_mc)
plt.show()
plt.plot(out_std_port)
plt.show()
# Plot Distribution of the Value-at-Risk 99%
sns.distplot(var99_port_mc,hist=True,kde=True,bins=50)

## Step 2: Check Normality Assumption with Jarque-Bera
out_jb=np.zeros((n_var_MC,2))
for i in range(n_var_MC):
    out_jb[i,:]=stats.jarque_bera(sim[:,i])

## Step 3: Comparing the Simulated and Original Correlation Matrix
cor_real=np.corrcoef(ret_port,rowvar=0)
cor_sim=np.corrcoef(sim,rowvar=0)

## Step 4: Computing Empirical Confidence Intervals 99% for Mean and Variance Estimation Errors
mean_sim=np.percentile(out_mean_mc-mean,(1,99),axis=0).T
variance_sim=np.percentile(out_std_mc**2-std**2,(1,99),axis=0).T

## Step 5: Comparing Simulated and Theoretical CI of Mean and Variance Estimation Errors
mean_theor=2.576/len(ret_port)**0.5*std
variance_theor=2.576/len(ret_port)**0.5*(2*std**4)**0.5


### ARMA GARCH model
## Step 1: ARMA Model for the Mean with Return Diagnostic (PACF->p; ACF->q)
ret_tot=np.sum(ret_port,axis=1)
lags=np.arange(1,10,1)
plot_pacf(ret_tot,lags=lags)
plot_acf(ret_tot,lags=lags)
plt.show()
garch_qtest=sm.stats.acorr_ljungbox(ret_tot, lags=[10], boxpierce=True)
print('p-value Q-test=', garch_qtest[3])
at=ret_tot-np.mean(ret_tot)

## Step 2: GARCH Model for the Variance
# Step 2.a: PACF Squared Residuals
plot_pacf(at**2,lags=lags)
plt.show()
plot_acf(at**2,lags=lags)
plt.show()
# Step 2.b: Model Selection with Information Criteria: AIC + BIC
out_garch_aic=np.zeros((2,2))
out_garch_bic=np.zeros((2,2))
for order_s in range(1,3,1):
    for order_q in range(1,3,1):
        print(order_q)
        garch=arch_model(100*at, mean='Zero', vol='GARCH', p=order_s, o=0, q=order_q, rescale=False)
        garch_result=garch.fit()
        out_garch_aic[order_s-1,order_q-1]=garch_result.aic/len(ret_tot)
        out_garch_bic[order_s-1,order_q-1]=garch_result.bic/len(ret_tot)
garch=arch_model(at*100,p=1,q=1,o=0, mean='Zero', vol='GARCH')
garch_result=garch.fit()
print(garch_result.summary())
garch_result.plot(annualize="D")
resid=garch_result._resid

## Step 3: Residual Diagnostic
# Step 3.1: Residuals and Standardized Residuals
plt.plot(resid)
plt.show()
garch_std_resid=garch_result._resid/garch_result._volatility
plt.plot(garch_std_resid)
plt.show()
# Step 3.2: Normality Test of Standardized Residuals with JB
print(stats.jarque_bera(garch_std_resid))
print(stats.kurtosis(garch_std_resid))
# Step 3.3: Normality Test of Standardized Residuals with Ljung-Box
garch_qtest_resid=sm.stats.acorr_ljungbox(garch_std_resid, lags=[10], boxpierce=True)
print('p-value Q-test=', garch_qtest_resid[3])
garch_qtest_squared_resid=sm.stats.acorr_ljungbox(garch_std_resid**2, lags=[10], boxpierce=True)
print('p-value Q-test=', garch_qtest_resid[3])
yh=garch_result._resid
plot_pacf(yh,lags=lags)
plt.show()
plot_acf(yh,lags=lags)
plt.show()
plot_pacf(garch_std_resid**2,lags=lags)
plt.show()
plot_acf(garch_std_resid**2,lags=lags)
plt.show()

std_garch=garch_result._volatility/100
mean_garch=np.mean(ret_tot)

## Step 4: Computing Time Varying VaR 99% (Distribution)
var99_garch = mean_garch -2.576 * std_garch
plt.plot(var99_garch)
plt.show()
plt.plot(std_garch)
plt.show()

sns.distplot(var99_garch, hist =True, kde = True, bins = 50)

## Step 5: Accuracy of Estimated VaR 99%
count = np.zeros(ret_tot.shape)
for i in range (len(ret_tot)):
    count[i] = ret_tot[i]<var99_garch[i]

print(sum(count)/len(ret_tot))


### Estimating 95% Confidence Interval of Time Varying VaR 99%
ci_low_tot= np.percentile(var99_garch, 5)
ci_high_tot= np.percentile(var99_garch, 95)


###############################################################################

#### Optional: factors must be selected individually (ex. from row 265 to 271, from row 274 to 280, …)

### Plot Value Factor VaR 99% and Standard Deviation
plt.plot(var99_mc[:,0])
plt.show()
plt.plot(out_std_mc[:,0])
plt.show()

sns.distplot(var99_mc[:,0],hist=True,kde=True,bins=50)


### Plot Quality Factor VaR 99% and Standard Deviation
plt.plot(var99_mc[:,1])
plt.show()
plt.plot(out_std_mc[:,1])
plt.show()

sns.distplot(var99_mc[:,1],hist=True,kde=True,bins=50)


### Plot Momentum Factor VaR 99% and Standard Deviation
plt.plot(var99_mc[:,2])
plt.show()
plt.plot(out_std_mc[:,2])
plt.show()

sns.distplot(var99_mc[:,2],hist=True,kde=True,bins=50)


### Plot Short Volatility Factor VaR 99% and Standard Deviation
plt.plot(var99_mc[:,3])
plt.show()
plt.plot(out_std_mc[:,3])
plt.show()

sns.distplot(var99_mc[:,3],hist=True,kde=True,bins=50)

###############################################################################