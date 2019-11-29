#######################################################
# Python program name	: 
#Description	: JADr_paper.py
#Args           : Code for JADr paper:Feature Selection to Build the Vallecas Index                                                                                     
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#pyenv install 3.7.0
#pyenv local 3.7.0
#python3 -V
# To use ipython3 debes unset esta var pq contien old version
#PYTHONPATH=/usr/local/lib/python2.7/site-packages
#unset PYTHONPATH
# $ipython3
# To use ipython2 /usr/local/bin/ipython2
#/Library/Frameworks/Python.framework/Versions/3.7/bin/ipython3
#pip install rfpimp. (only for python 3)
#######################################################
# -*- coding: utf-8 -*-
import os, sys, pdb, operator
import time
import numpy as np
import pandas as pd
import importlib
#importlib.reload(module)
import sys
import statsmodels.api as sm
import time
import datetime
import importlib
import random
import pickle
from scipy import stats
#import rfpimp
from rfpimp import *
#sys.path.append('/Users/jaime/github/code/tensorflow/production')
#import descriptive_stats as pv
#sys.path.append('/Users/jaime/github/papers/EDA_pv/code')
import warnings
#from subprocess import check_output
#import area_under_curve 
import matplotlib
matplotlib.use('Agg')
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
# from pprint import pprint
# from sklearn import metrics
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score,\
# confusion_matrix,roc_auc_score, roc_curve, auc, classification_report,precision_recall_curve,\
# make_scorer, average_precision_score
sys.path.append('/Users/jaime/github/papers/JADr_VallecasIndex/code')
from JADr_paper import vallecas_features_dictionary, select_PVdictio, dataset_cleanup, encode_in_quartiles, rename_esp_eng_pv_columns, feature_selection_filtering, split_training_test_sets


def compound_interest(principle, rate, years): 
	# Calculates compound interest 
	#principle = 100; rate <0, time number of years 
	brain_left= principle * (pow((1 + rate / 100), years)) 
	print("The brain left is :", brain_left) 

def scatterplot_atrophy_i_iplus1(df, label=None, figures_dir=None):
	"""scatterplot_atrophy_i_iplus1: Scatter plot all points x: 21,32,43 y: 32, 43, 54 
	2 consecutive years
	"""
	fig, ax = plt.subplots(figsize=(9,9))
	c1 = df['conversionmci']
	c1 = pd.concat([c1,c1,c1], axis=0,ignore_index=True)
	if label is None:
		label=''
		s1 = df['siena_12']; s2 = df['siena_23'];s3 = df['siena_34'];s4 = df['siena_45']
	elif label is 'vel_':
		s1 = df['siena_vel_12']; s2 = df['siena_vel_23'];s3 = df['siena_vel_34'];s4 = df['siena_vel_45']
	elif label is 'accloss_':
		s1 = df['siena_accloss_12']; s2 = df['siena_accloss_23'];s3 = df['siena_accloss_34'];s4 = df['siena_accloss_45']
	elif label is 'velsquared_':
		s1 = df['siena_velsquared_12']; s2 = df['siena_velsquared_23'];s3 = df['siena_velsquared_34'];s4 = df['siena_velsquared_45']
		
	delta_i = pd.concat([s1, s2, s3], axis=0, ignore_index=True)
	delta_plusi = pd.concat([s2, s3, s4], axis=0, ignore_index=True)

	frame = {'i':delta_i, 'iplus1':delta_plusi}
	frame = {'i':delta_i, 'iplus1':delta_plusi, 'mci':c1}
	
	df = pd.DataFrame(frame)
	df = df.apply(pd.to_numeric, errors='coerce')
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df = df.dropna()
	#regression part
	
	slope, intercept, r_value, p_value, std_err = stats.linregress(df.i,df.iplus1)
	line = slope*df.i+intercept
	plt.plot(df.i, line, 'r', label='$y={:.2f}x + {:.2f}'.format(slope,intercept))
	#scatter plot part

	#plt.scatter(df.i,df.iplus1, s=3, c='conversionmci', alpha=0.5)
	col = df.mci.map({0:'b', 1:'brown'})
	plt.scatter(df.i,df.iplus1, c=col, s=3, alpha=0.8)
	plt.title(r'$Scatter plot \Delta_i \Delta_{i+1}$' + label)
	plt.xlabel(r'$ \Delta_{i,i+1} (i=1..3) $')
	plt.ylabel(r'$ \Delta_{i,i+1} (i=2...4)$')
	plt.legend(fontsize=9)
	fig_file = os.path.join(figures_dir, 'scatter_' + label + '.png')
	plt.savefig(fig_file)
	return



def boxplot_ts(df, label, figures_dir):
	"""
	"""
	if label is None:
		label=''
		s1 = df['siena_12']; s2 = df['siena_23'];s3 = df['siena_34'];s4 = df['siena_45']
	elif label is 'vel_':
		s1 = df['siena_vel_12']; s2 = df['siena_vel_23'];s3 = df['siena_vel_34'];s4 = df['siena_vel_45']
	elif label is 'accloss_':
		s1 = df['siena_accloss_12']; s2 = df['siena_accloss_23'];s3 = df['siena_accloss_34'];s4 = df['siena_accloss_45']
	elif label is 'velsquared_':
		s1 = df['siena_velsquared_12']; s2 = df['siena_velsquared_23'];s3 = df['siena_velsquared_34'];s4 = df['siena_velsquared_45']
	atrophyconcat = pd.concat([s1, s2, s3, s4], axis=0, ignore_index=True)
	nrows = s1.shape[0]
	y12 = pd.Series(['12']); y12=y12.repeat(nrows)
	y23 = pd.Series(['23']); y23=y23.repeat(nrows) 
	y34 = pd.Series(['34']); y34=y34.repeat(nrows)
	y45 = pd.Series(['45']); y45=y45.repeat(nrows)
	yearsconcat = pd.concat([y12, y23, y34, y45], axis=0, ignore_index=True)
	frame = {'years':yearsconcat, 'atrophy':atrophyconcat}
	df2boxplot = pd.DataFrame(frame)
	df2boxplot = df2boxplot.dropna()
	# plot boxplot same figure with sns and pandas 
	fig, ax = plt.subplots(figsize=(9,9))
	sns.boxplot(x='years', y='atrophy', data=df2boxplot,width=0.5,palette="colorblind")
	plt.title('Boxplot atrophy:' + label)
	fig_file = os.path.join(figures_dir, 'boxplot_' + label + '_sns_ts'+ '.png')
	plt.savefig(fig_file)

	sns.stripplot(x='years', y='atrophy',data=df2boxplot, jitter=True, marker='o', alpha=0.3, color='black')
	plt.title('Boxplot atrophy:' + label)
	fig_file = os.path.join(figures_dir, 'stripplot_' + label + '_ts'+ '.png')
	plt.savefig(fig_file)
	
	df2boxplot.boxplot(by='years',column=['atrophy'], grid=True).set_title(label)
	fig_file = os.path.join(figures_dir, 'boxplot_' + label + '_pd_ts'+ '.png')
	plt.savefig(fig_file)
	return
	

def get_loss_timeseries_full(df, yi, ye):
	"""get_loss_timeseries_full: Build DataFrame index(0..5) yearvisit(rounded) values
	"""
	df2ts = pd.DataFrame()

	vol2eITV= df.iloc[0]['BrainSegVol_to_eTIV_y1']
	y12vol = vol2eITV*df.iloc[0]['siena_accloss_12'];y23vol = y12vol*df.iloc[0]['siena_accloss_23'];
	y34vol = y23vol*df.iloc[0]['siena_accloss_34'];y45vol = y34vol*df.iloc[0]['siena_accloss_45']
	
	agev1 = df.iloc[0]['edad_visita1']; agev2 = df.iloc[0]['edad_visita2'];
	agev3 = df.iloc[0]['edad_visita3']; agev4 = df.iloc[0]['edad_visita4'];
	agev5 = df.iloc[0]['edad_visita5']; y19 = 19; 
	tsfull = [[y19, 1.000], [agev1, vol2eITV], [agev2, y12vol],[agev3, y23vol], [agev4, y34vol], [agev5, y45vol]]
	df2r = pd.DataFrame(data=tsfull, columns = ['age', 'bvol'])
	# Round age to int
	df2r.age = df2r.age.round()
	return df2r 


def get_timeseries(df, label, yi, ye):
	"""
	"""
	df2ts = pd.DataFrame()
	for yy in np.arange(yi, ye):
		pairyears = str(yy)+str(yy+1)
		atrophy_label = 'siena_' + label + pairyears
		sij =  df[atrophy_label]
		df2ts = df2ts.append(sij)
	return df2ts

def get_timeseries_siena(df):
	"""
	"""
	s12 = df['siena_vel_12'];s23 = df['siena_vel_23'];s34 = df['siena_vel_34'];s45 = df['siena_vel_45']
	return pd.concat([s12,s23,s34,s45], axis=1)

def plot_timeseries_siena(df, label=None, figures_dir=None):
	"""plot_timeseries_siena
	Args:label = 'accloss_' , 'vel_'
	Out:
	"""
	yi=1; ye=5
	if label is None:label=''
	df2plot = pd.DataFrame()
	for yy in np.arange(yi, ye):
		pairyears = str(yy)+str(yy+1)
		atrophy_label = 'siena_' + label + pairyears
		sij =  df[atrophy_label]
		#ts_gr1 = pd.concat([sij,sall], axis=1)
		df2plot = df2plot.append(sij)
	fig, ax = plt.subplots(figsize=(15,7))
	ax = plt.gca()
	ax.grid(True)
	plt.plot(df2plot)
	plt.title(r'$Time series:$' + label)
	fig_file = os.path.join(figures_dir, 'ts_' + label + '.png')
	plt.savefig(fig_file)
	return df2plot


def random_walk_model_df(df_ts):
	"""
	"""
	print('dickey fuller test for dataframe...\n')
	df_ts['stationary'] = pd.Series()
	indices_sub = df_ts.index.values.tolist()
	j=0
	for i in range(len(df_ts)):
		print('ADF for Series i=%s\n' % str(i))
		series = df_ts.iloc[i,:][0:4]
		result = stationarity_test(series, i)
		arima_model(series,i)
		# Base Level +x Trend +x Seasonality +x Error
		#components= ts_components(series,i)
		print('Sample Entropy:')

		print(SampEn(series.values, m=1, r=2.5*np.std(series.values)))
		df_ts['stationary'] = df_ts['stationary'].set_value(indices_sub[j], result)
		j = j + 1
	pdb.set_trace()
	return df_ts

def SampEn(U, m, r):
    """Compute Sample entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))


def ts_components(series, label=None):
	"""
	"""
	figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/longit/adftest/components'
	if label is None:label=' '
	from statsmodels.tsa.seasonal import seasonal_decompose
	dates = np.array(['2012', '2014','2016','2018'], dtype=np.datetime64)
	frame = {'ts': series.values}
	df = pd.DataFrame(frame, index=dates)
	#result seasonal, trend, resid
	result = seasonal_decompose(df, model='additive', freq=1)
	result.plot()
	fig_file = os.path.join(figures_dir,'ts22_components_' + str(label)+ '.png')
	plt.savefig(fig_file)
	return result


def arima_model(series, label=None):
	"""to forecast stationary time series
	"""
	from statsmodels.tsa.arima_model import ARIMA

	# 1,1,2 ARIMA Model
	model = ARIMA(series.values, order=(1,1,0))
	model_fit = model.fit(disp=0)
	print(model_fit.summary())


def stationarity_test(series, label=None,figures_dir=None):
	"""stationarity_test adfuller
	"""
	from statsmodels.tsa.stattools import adfuller
	from numpy import log
	if label is None:
		label = ' '
	# check is series is stationary
	
	if series.shape[0] > 1:
		# ND array convert into 1D
		series_arr = series.values.ravel()
	else:
		series_arr = series.values 

	result = adfuller(series_arr, maxlag=2)
	# H0: series is nonstationary, HA: Alternative hypothesis is usually stationarity or trend-stationarity
	#If H0 is rejected, the alternative hypothesis (Ha) can be accepted: the residue series is stationary.
	#print('ADF Statistic H0:a unit root is present in an autoregressive model (or (non stationary)) : %f' % result[0])
	print('p-value: %s used lags %s nobs %s' % (str(result[1]),str(result[2]),str(result[3])))
	thr = 0.05
	if result[1]<thr:
		print('Reject H0 ergo time series is stationary and does not have an unit root.')
		plot_ts_stationairy_test(series, label, figures_dir)
	else:
		print('Failed to reject H0:time series is non-stationary. Needs differencing')
	return result

def plot_ts_stationairy_test(series, label=None, figures_dir=None):	
	# if the p-value of the test is less than the significance level (0.05) then you reject the null hypothesis and infer that the time series is indeed stationary.
	# Otherwise, we need to do differencing,m find d, to make the series stationary
	# Original Series
	
	from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
	if label is None:label=' '
	#figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/longit/adftest'
	fig, axes = plt.subplots(3, 2, sharex=True)
	axes[0, 0].plot(series); axes[0, 0].set_title('Original Series')
	plot_acf(series, ax=axes[0, 1])

	# 1st Differencing
	axes[1, 0].plot(series.diff()); axes[1, 0].set_title('1st Order Differencing')
	plot_acf(series.diff().dropna(), ax=axes[1, 1])

	# 2nd Differencing
	axes[2, 0].plot(series.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
	plot_acf(series.diff().diff().dropna(), ax=axes[2, 1])
	fig_file = os.path.join(figures_dir, 'timseseries_diif_i_' + str(label) + '.png')
	plt.savefig(fig_file)

def accumulated_loss(df):
	"""accumulated_loss: add columns with he accumulated loss across years
	Args:dataframe
	Out:dataframe with new columns siena_accloss_ij
	"""
	yi=1; ye=7
	atrophy = 'siena_' + str(yi)+str(ye)
	s1 = df['siena_12']; s2 = df['siena_23'];s3 = df['siena_34'];s4 = df['siena_45'];s5 = df['siena_56'];s6 = df['siena_67']
	pairs = []; accum_loss=1; accum_loss_list = []
	df['siena_accloss_01'] = 1; df['siena_accloss_12'] = df['siena_accloss_01']; df['siena_accloss_23']=df['siena_accloss_12'];df['siena_accloss_34']=df['siena_accloss_23'];df['siena_accloss_45']=df['siena_accloss_34'];df['siena_accloss_56']=df['siena_accloss_45'];df['siena_accloss_67']=df['siena_accloss_56']
	#pdb.set_trace()
	for yy in np.arange(yi, ye):
		pairyears = str(yy)+str(yy+1)
		atrophy_label = 'siena_' + pairyears
		accum_loss_label_prev = 'siena_accloss_' + str(yy-1)+str(yy)
		accum_loss_label = 'siena_accloss_' + pairyears
		accum_loss = (1 + df[atrophy_label]/100.)*accum_loss
		df[accum_loss_label] = np.where(df[atrophy_label].isnull(), df[accum_loss_label_prev], (1 + df[atrophy_label]/100.)*df[accum_loss_label_prev])
		pairs.append(pairyears)
		accum_loss_list.append(accum_loss)
	return df

def compare_groups_ttest(grp1, grp2,the_file,class2cmp):
	"""
	"""
	from scipy.stats import ttest_ind
	tstat, pval = ttest_ind(grp1, grp2)
	print('tstat and pval %s %s' %(tstat, pval))
	the_file.write('ttest for classes:' + class2cmp + ' tstat=' + str(tstat) + ' pval=' + str(pval) + '\n')
	return 


def get_conditions_in_df(df, condition):
	"""get_conditions_in_df
	Example:  get_conditions_in_df(df, 'apoe')
	Args:
	Out: return list of dataframes one for each condition selection
	"""
	consitionslist = {} 
	if condition is 'apoe':
		# apoe mask
		mask_a0 = df['apoe']==0; df_a0 = df[mask_a0];condkey = condition + '0';consitionslist[condkey] = df_a0
		mask_a1 = df['apoe']==1; df_a1 = df[mask_a1];condkey = condition + '1';consitionslist[condkey] = df_a1
		mask_a2 = df['apoe']==2; df_a2 = df[mask_a2];condkey = condition + '2';consitionslist[condkey] = df_a2
		mask_a12 = df['apoe']>=1; df_a12 = df[mask_a12];condkey = condition + '12';consitionslist[condkey] = df_a12
		#consitionslist[condition] = [df_a0,df_a1,df_a2,df_a12]
	elif condition is 'conversionmci':
		mask_mci0 = df['conversionmci']==0; df_mci0 = df[mask_mci0];condkey = condition + '0';consitionslist[condkey] = df_mci0
		mask_mci1 = df['conversionmci']==1; df_mci1 = df[mask_mci1];condkey = condition + '1';consitionslist[condkey] = df_mci1
	elif condition is 'sexo':
		mask_male = df['sexo']==0; df_male = df[mask_male];condkey = condition + 'M';consitionslist[condkey] = df_male
		mask_female = df['sexo']==1; df_female = df[mask_female];condkey = condition + 'F';consitionslist[condkey] = df_female
	elif condition is 'familial_ad':
		mask_fam0 = df['familial_ad']==0; df_fam0 = df[mask_fam0];condkey = condition + '0';consitionslist[condkey] = df_fam0
		mask_fam1 = df['familial_ad']==1; df_fam1 = df[mask_fam1];condkey = condition + '1';consitionslist[condkey] = df_fam1
	elif condition is 'nivel_educativo':
		mask_educa01 = df['nivel_educativo']<=1; df_educa01 = df[mask_educa01];condkey = condition + '01';consitionslist[condkey] = df_educa01
		mask_educa23 = df['nivel_educativo']>1; df_educa23 = df[mask_educa23];condkey = condition + '23';consitionslist[condkey] = df_educa23
	elif condition is 'edad_visita1':
		mask_edad70 = df['edad_visita1']<80; df_edad70 = df[mask_edad70];condkey = condition + '70';consitionslist[condkey] = df_edad70
		mask_edad80 = df['edad_visita1']>=80; df_edad80 = df[mask_edad80];condkey = condition + '80';consitionslist[condkey] = df_edad80
	elif condition is 'edad_ultimodx':
		mask_edad70 = df['edad_ultimodx']<78; df_edad70 = df[mask_edad70];condkey = condition + '78';consitionslist[condkey] = df_edad70
		mask_edad80 = df['edad_ultimodx']>=78; df_edad80 = df[mask_edad80];condkey = condition + '88';consitionslist[condkey] = df_edad80
	return consitionslist

# def compare_conditions_ttest(dfgr1, dfgr2, xvars, class2cmp, figures_dir=None):
# 	"""
# 	"""
	

# 	# compare groups ttest 
# 	#reportfile = '/Users/jaime/github/papers/atrophy_long/figures/longit/reportfile.txt'
# 	reportfile = os.path.join(figures_dir, 'reportfile.txt');
# 	file_hld= open(reportfile,"w+")
# 	for siena in xvars:
# 		class2cmp = 'APOE:0,12-' + siena
# 		compare_groups_ttest(df_a0[siena], df_a12[siena],file_hld, class2cmp)
# 		class2cmp = 'MCI:0,1-' + siena
# 		compare_groups_ttest(df_mci0[siena], df_mci1[siena],file_hld, class2cmp)
# 		class2cmp = 'Sex:0,1-' + siena
# 		compare_groups_ttest(df_male[siena], df_female[siena],file_hld, class2cmp)
# 		class2cmp = 'Educa:01,23-' + siena
# 		compare_groups_ttest(df_educa01[siena], df_educa23[siena],file_hld, class2cmp)
# 		class2cmp = 'Age:<80,>=80-' + siena
# 		compare_groups_ttest(df_edad70[siena], df_edad80[siena],file_hld, class2cmp)
# 		file_hld.write('\n')
# 	file_hld.close() 


def mask_visits(df, yi, ye, condition_label=None):
	"""mask_visits
	Args: yi, ye, condition_label 
	Out: dataframe of longitudinal from yi to ye AND condition_label
	mask_siena_or for 2 atrophies (4 tie points) we shoudl have siean_15 (!!)
	"""
	mask_siena = df['siena_12'].notnull() & df['siena_23'].notnull() & df['siena_34'].notnull() &  df['siena_45'].notnull()
	#mask_siena_or = df['siena_12'].notnull() &  df['siena_45'].notnull()
	mask_siena_condition = mask_siena
	print('Final Mask SIENA nb of Rows AND: %s' %(sum(mask_siena))) #,sum(mask_siena_or)))
	if condition_label is not None:
		if condition_label is 'healthy':
			#mask_condition = (df['dx_corto_visita1']==0) & (df['dx_corto_visita2']==0) & (df['dx_corto_visita3']==0) & (df['dx_corto_visita4']==0) & (df['dx_corto_visita5']==0)
			mask_condition = (df['dx_corto_visita1']==0) & (df['dx_corto_visita2']==0) & (df['dx_corto_visita3']==0) & (df['dx_corto_visita4']==0) & (df['dx_corto_visita5']==0)
		elif condition_label is 'mci':
			mask_condition = (df['dx_corto_visita1']==1) & (df['dx_corto_visita2']==1) & (df['dx_corto_visita3']==1) & (df['dx_corto_visita4']==1) & (df['dx_corto_visita5']==1)
		mask_siena_condition = mask_siena & mask_condition
	return df[mask_siena_condition]

def remove_outliers(df):
	"""
	"""
	df_orig = df.copy()
	low = .01
	high = .99
	#siena_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	siena_cols = ['siena_12','siena_23','siena_34','siena_45','siena_56','siena_67']
	df_siena = df[siena_cols]
	quant_df = df_siena.quantile([low, high])
	
	print('Outliers: low= %.3f high= %.3f \n %s' %(low, high, quant_df))
	df_nooutliers = df_siena[(df_siena > quant_df.loc[low, siena_cols]) & (df_siena < quant_df.loc[high, siena_cols])]
	df_outs = df_siena[(df_siena <= quant_df.loc[low, siena_cols]) | (df_siena >= quant_df.loc[high, siena_cols])]
	df[siena_cols] = df_nooutliers.to_numpy()
	# List of outliers
	reportfile = '/Users/jaime/github/papers/atrophy_long/figures/longit/outliersLH.txt'
	file_h= open(reportfile,"w+")
	print('Outliers Low : High %s %s' %(low, high))
	file_h.write('Outliers: low= %.3f high= %.3f \n %s \n' %(low, high, quant_df))
	for year in siena_cols:
		outliers_y = df_outs.index[df_outs[year].notna() == True].tolist()
		file_h.write('\tOutliers Years :' + year + str(outliers_y) + '\n')
	return df


def plot_joint_pairs(dataframe, columns=None, label=None, hue=None, figures_dir=None):
	"""
	"""
	if columns is None:columns = ['conversionmci','apoe', 'siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45','siena_accloss_56','siena_accloss_67']
	if label is None:label='accloss_'
	if hue is None:hue='conversionmci'

	df2plt = dataframe[columns]
	try:
		print('Calling to pairplot atrophy %s' % (columns))
		sns_plot = sns.pairplot(df2plt, hue=hue, size=2.5)
		#sns_plot = sns.pairplot(df2plt, size=2.5)
		fig_file = os.path.join(figures_dir, label + '_' + hue + '_pairplot.png')
		print('Saving sns pairplot atrophy at %s' % (fig_file))
		sns_plot.savefig(fig_file)
	except ZeroDivisionError:
		print("pairplot empty for selection  !!! \n\n")


def plot_by_pair(dataframe, x_vars=None, y_var=None, figures_dir=None):
	"""plot_by_pair plot x axis atrophy and y axis phenotype of interest, first element in xvars is the hue
	"""
	if x_vars is None: x_vars = ['siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45']
	if y_var is None: y_var = ["edad_ultimodx"]
	#columns.append('edad_ultimodx')
	sns_plot = sns.pairplot(dataframe, hue='conversionmci', x_vars=x_vars, y_vars=y_var,height=5, aspect=.8);
	fig_file = os.path.join(figures_dir, y_var + x_vars[-1].split('_')[1] + '_pairplot.png')	
	sns_plot.savefig(fig_file)


def scatterplot_2variables_in_df(df2plt,xvar,yvar,figures_dir):
	"""scatterplot_2variables_in_df: scatter plot of 2 variables in dataframe
	Example: scatterplot_2variables_in_df(df,'siena_vel_12','siena_vaccloss_23')
	Args:df, xvar, yvar. 
	Out:
	"""
	def r2(x, y):
		return stats.pearsonr(x, y)[0] ** 2
	yearsx = xvar.split('_')[-1][-1]; yearsy =  yvar.split('_')[-1][-1]
	labelx = xvar.split('_')[0]; labely = yvar.split('_')[0]
	titlelabel = labelx + '-' + yearsx + '_' + labely + '-' + yearsy
	fig, ax = plt.subplots(figsize=(15,7))
	snsp = sns.jointplot(x=xvar, y=yvar, data=df2plt.replace([np.inf, -np.inf], np.nan), kind="reg", stat_func=r2);
	fig_file = os.path.join(figures_dir, 'joint_' + titlelabel + '.png')
	snsp.savefig(fig_file)

def plot_distribution_variable_in_df(df2plt, xvar, figures_dir=None):
	"""plot_distribution_variable_in_df distribution of xvar with mean (k) and median (r)
	"""

	plt.figure(figsize=(8, 9))
	sns.distplot(df2plt[xvar], hist=False, rug=True)
	fig_file = os.path.join(figures_dir, 'hist_' + xvar + '.png')
	plt.axvline(df2plt[xvar].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(x=1.00, color='green', linestyle='dashed', linewidth=0.5)
	plt.axvline(df2plt[xvar].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt[xvar].mean()*1.0, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt[xvar].mean(),df2plt[xvar].std()))
	plt.savefig(fig_file)

def plot_correlation_in_df(df2plt, xvars, yvars, figures_dir=None):
	"""plot_correlation_in_df heatmap of xvars vs yvars of df
	Args:df, xvars,yvars . xvars can be == yvars in this case symmetric squared matrix plot as triangular
	Example:  plot_correlation_in_df(df2plt, ['siena_accloss_12',..'siena_accloss_45'] , ['siena_accloss_12',..'siena_accloss_45'], figures_dir=None)
	"""
	labelx = xvars[0].split('_')[-2]; labely = yvars[0].split('_')[-2]
	if xvars == yvars:
		# symmetric (and squared) matrix mask to print only triangular
		atrophy_corr = df2plt[xvars].corr()
		titlelabel =  '_' + labelx + '-'
		mask = np.zeros_like(atrophy_corr)
		mask[np.triu_indices_from(mask)] = True
	else:
		atrophy_corr = df2plt[xvars + yvars].corr().iloc[0:len(xvars),len(yvars)-1:]
		titlelabel =  '_' + labelx + '-' + labely
		mask = np.zeros_like(atrophy_corr)
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(atrophy_corr,mask=mask,annot=True, center=0,square=True, linewidths=.5)
	#heatmap = sns.heatmap(atrophy_corr,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heatcorr' + titlelabel + '.png')
	plt.savefig(fig_file)
	return atrophy_corr
		

def acceleration_interannual(dataframe):
	"""acceleration_interannual: add columns with acceleration interannual
	call this function only for consecutive time points (datarame_longit)
	"""

	#veloc_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	dataframe['siena_velsquared_12'] = dataframe['siena_12']/ pow(dataframe['tpo1.2'],2)
	dataframe['siena_velsquared_23'] = dataframe['siena_23']/ pow((dataframe['tpo1.3']-dataframe['tpo1.2']),2)
	dataframe['siena_velsquared_34'] = dataframe['siena_34']/ pow((dataframe['tpo1.4']-dataframe['tpo1.3']),2)
	dataframe['siena_velsquared_45'] = dataframe['siena_45']/ pow((dataframe['tpo1.5']-dataframe['tpo1.4']),2)
	dataframe['siena_velsquared_56'] = dataframe['siena_56']/ pow((dataframe['tpo1.6']-dataframe['tpo1.5']),2)
	dataframe['siena_velsquared_67'] = dataframe['siena_67']/ pow((dataframe['tpo1.7']-dataframe['tpo1.6']),2)
	return dataframe

def velocity_interannual(dataframe):
	"""velocity_interannual: add columns with velocity interannual
	call this function only for consecutive time points (datarame_longit)
	"""

	#veloc_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	dataframe['siena_vel_12'] = dataframe['siena_12']/ (dataframe['tpo1.2'])
	dataframe['siena_vel_23'] = dataframe['siena_23']/ (dataframe['tpo1.3']-dataframe['tpo1.2'])
	dataframe['siena_vel_34'] = dataframe['siena_34']/ (dataframe['tpo1.4']-dataframe['tpo1.3'])
	dataframe['siena_vel_45'] = dataframe['siena_45']/ (dataframe['tpo1.5']-dataframe['tpo1.4'])
	dataframe['siena_vel_56'] = dataframe['siena_56']/ (dataframe['tpo1.6']-dataframe['tpo1.5'])
	dataframe['siena_vel_67'] = dataframe['siena_67']/ (dataframe['tpo1.7']-dataframe['tpo1.6'])
	return dataframe

def convert_stringtofloat(dataframe):
	"""convert_stringtofloat: cast edad_ultimodx, edad_visita1,tpoi.j and siena_ij to float 
	Args:dataframe
	Out:dataframe
	"""
	# Change cx_cortov1 to float because all other dx corto are float (not strictly necessary)
	dataframe.dx_corto_visita1 = dataframe.dx_corto_visita1.astype(float)
	dataframe['edad_ultimodx'] = dataframe['edad_ultimodx'].str.replace(',','.').astype(float)
	dataframe['edad_visita1'] = dataframe['edad_visita1'].str.replace(',','.').astype(float)
	sv_toconvert = ['siena_12','siena_23','siena_34','siena_45','siena_56','siena_67','viena_12','viena_23','viena_34','viena_45','viena_56','viena_67']
	for ix in sv_toconvert:
		print('converting str to float in %s' % ix)
		dataframe[ix] = dataframe[ix].str.replace(',','.').astype(float)
	tpo_toconvert = ['tpo1.2','tpo1.3', 'tpo1.4','tpo1.5','tpo1.6','tpo1.7']
	for ix in tpo_toconvert:
		print('converting str to float in %s' % ix)
		dataframe[ix] = dataframe[ix].str.replace(',','.').astype(float)
	return dataframe
def plot_groupby(df, figures_dir):
	"""
	"""
	fig, ax = plt.subplots(figsize=(15,7))
	df.groupby('conversionmci')['siena_12'].agg(['mean','median']).plot(kind='bar')
	fig_file = os.path.join(figures_dir, 'groupby.png')	
	plt.savefig(fig_file)

	multicol_sum = df.groupby([pd.cut(df['edad_visita1'],2)]).mean()
	pdb.set_trace()

def save_df_csvfile(df, typestudy=None):
	"""save_df_csvfile save df as .csv with actual date in the filename
	Args: df, typestudy= Longit|Crosssectional only for file name purposes
	"""
	from datetime import date
	if typestudy is None:typestudy='longit'
	today = date.today()
	d1 = today.strftime("%d-%m-%Y")
	print("d1 =", d1)
	csv_dir = '/Users/jaime/github/papers/atrophy_long/'
	csv_fname = os.path.join(csv_dir, 'siena_lossandvelacc_' + d1 + typestudy + '.csv')
	excel_fname = os.path.join(csv_dir, 'siena_lossandvelacc_' + d1 + typestudy+ '.xlsx')
	columnsofinterest = ['idpv', 'edad_visita1', 'edad_ultimodx','ultimavisita', 'sexo','nivel_educativo','apoe',\
	'familial_ad','conversionmci','tpo1.2','tpo1.3','tpo1.4','tpo1.5','tpo1.6','tpo1.7',\
	'siena_12','siena_23','siena_34','siena_45','siena_56','siena_67',\
	'siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45','siena_accloss_56', 'siena_accloss_67',\
	'siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67',\
	'siena_velsquared_12','siena_velsquared_23','siena_velsquared_34','siena_velsquared_45','siena_velsquared_56','siena_velsquared_67',\
	'dx_corto_visita1','dx_corto_visita2','dx_corto_visita3','dx_corto_visita4','dx_corto_visita5','dx_corto_visita6','dx_corto_visita7',\
	'fcsrtlibdem_visita1','fcsrtlibdem_visita2','fcsrtlibdem_visita3','fcsrtlibdem_visita4','fcsrtlibdem_visita5','fcsrtlibdem_visita6', 'fcsrtlibdem_visita7',\
	'mmse_visita1','mmse_visita2','mmse_visita3','mmse_visita4','mmse_visita5','mmse_visita6','mmse_visita7']
	dfshort = df[columnsofinterest]
	dfshort.to_csv(csv_fname,sep=',')
	dfshort.to_excel(excel_fname, float_format="%.6f")
	return dfshort

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################


### BEGIN TEST ###

def plot_rollingmean(df_ts, subid=None):
	"""https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788
	FIX!! interpolate abd plot curve with  all points 19,20,21,.....80 	
	"""
	fig, ax = plt.subplots(1,figsize=(12, 9))
	#ax.plot(df_ts.index, df_ts, label='raw data')
	ax.plot(df_ts.age, df_ts.bvol, label='raw data')
	ax.plot(df_ts.age, df_ts.bvol.rolling(window=2).mean(), label="rolling mean");
	ax.plot(df_ts.age, df_ts.bvol.rolling(window=2).std(), label="rolling std");
	#label = df_ts.columns[0]
	fig_file = os.path.join(figures_dir, 'rolling_id' + str(subid) + '.png')
	#plt.grid(axis='x')
	ax.grid(axis='both')
	ax.legend()
	plt.savefig(fig_file)

def df_ts_set_subjects(df, listsubjects):
	"""
	"""
	df_set_subjects = pd.DataFrame()
	for s_id in listsubjects:
		df_s = timeseries_df_per_subject(df, s_id)
		df_s['id'] = s_id
		df_set_subjects = df_set_subjects.append(df_s)
	return df_set_subjects

def timeseries_df_per_subject(df, subject_id):
	""" Get time series dataframe ts for one subject 
	columns: age bvol idpv conversionmci
	"""
	if type(df.index) == pd.RangeIndex:
		df.set_index("idpv", inplace=True,verify_integrity=False)
	#df.loc[subject_id] Series df.loc[[6,12]] DataFrame
	columns_names = ['edad_visita1','edad_visita2','edad_visita3',\
	'edad_visita4','edad_visita5', 'apoe', 'familial_ad',\
	'conversionmci','dx_corto_visita5', 'siena_accloss_12','siena_accloss_23',\
	'siena_accloss_34','siena_accloss_45']
	columns_names.append('BrainSegVol_to_eTIV_y1')
	df_row = df.loc[[subject_id],columns_names] #df
	df_ts = get_loss_timeseries_full(df_row, 1, 5)
	# Add columns with constant values inthe time series eg: pvid, conversion etc
	df_ts['idpv'] = subject_id
	df_ts['conversionmci'] = df_row['conversionmci'].values[0]
	df_ts['dx_corto_visita5'] = df_row['dx_corto_visita5'].values[0]
	# df_ts = get_timeseries(df_row, 'accloss_', 1, 5)
	# plot_rollingmean(df_ts,subject_id)
	return df_ts

def plot_exponential_smoothing(ts, fit1, fit2, fit3, pred1, pred2, pred3, label, title, figures_dir):
	"""
	"""
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(np.arange(4), ts, label="ts")
	for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('#ff7823','#3c763d','c')):
		label2plot = "alpha="+str(f.params['smoothing_level'])[:3]
		if title.split(' ')[0] == 'Holt':
			label2plot="alpha="+str(f.params['smoothing_level'])[:4]+", beta="+str(f.params['smoothing_slope'])[:4]
		ax.plot(np.arange(4), f.fittedvalues, color=c,label=label2plot)
	plt.title(title) 
	plt.xticks(np.arange(4), ('y1-y2', 'y2-y3', 'y3-y4', 'y4-y5'), rotation=0)
	plt.legend()
	fig_file = os.path.join(figures_dir, title + label + '.png')
	plt.grid(axis='x')
	plt.savefig(fig_file)

def exponential_smoothing(ts, label=None):
	"""Exponential smoothing methods assign exponentially 
	decreasing weights for past observations.
	"""
	from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
	locs, labels = plt.xticks()

	model = SimpleExpSmoothing(np.asarray(ts))
	#  auto optimization 
	fit1 = model.fit()
	pred1 = fit1.forecast(2)
	fit2 = model.fit(smoothing_level=.25)
	pred2 = fit2.forecast(2)
	fit3 = model.fit(smoothing_level=.5)
	pred3 = fit3.forecast(2)
	plot_exponential_smoothing(ts, fit1, fit2, fit3, pred1, pred2, pred3, label, "Simple Exponential Smoothing",figures_dir)

	model = ExponentialSmoothing(np.asarray(ts), trend='add')
	#  auto optimization 
	fit1 = model.fit()
	pred1 = fit1.forecast(2)
	fit2 = model.fit(smoothing_level=.9)
	pred2 = fit2.forecast(2)
	fit3 = model.fit(smoothing_level=.5)
	pred3 = fit3.forecast(2)
	plot_exponential_smoothing(ts, fit1, fit2, fit3, pred1, pred2, pred3, label, "Exponential Smoothing",figures_dir)

	model = Holt(np.asarray(ts))
	#  auto optimization 
	fit1 = model.fit(smoothing_level=.3, smoothing_slope=.05)
	pred1 = fit1.forecast(2)
	fit2 = model.fit(optimized=True)
	pred2 = fit2.forecast(2)
	fit3 = model.fit(smoothing_level=.3, smoothing_slope=.2)
	pred3 = fit3.forecast(2)
	plot_exponential_smoothing(ts, fit1, fit2, fit3, pred1, pred2, pred3, label, "Holt Exponential Smoothing",figures_dir)
	return fit3
	
def print_exp_smoothing_results(fit_res):
	"""
	"""
	print('Exponential smoothing results:\n')
	print('ExpSmoothing params: %s' % fit_res.params)
	print('SSE: %f' % fit_res.sse)
	print('AIC: %f BIC %f' % (fit_res.aic,fit_res.aic) )
	print('Residuals: %s ' % fit_res.resid)
	print('fitted values: %s and predicted %s' % (fit_res.fittedvalues,fit_res.fittedfcast))

def qcurvefitting(x, y, subject_id=None):
	"""quadratic curvefitting
	"""
	from scipy.optimize import curve_fit
	def func(x, a, b, c):return a + b * x + c * x ** 2
	def func3(x, a, b, c,d):return a + b * x + c * x ** 2 + d * x ** 3 
	zfull = np.polyfit(x, y, 2, full=True); residuals = zfull[1]
	print('Fitted poly a + bx + cx^2: %s' %(zfull[0]));print('Residuals^2 %f' %(np.polyfit(x, y, 2,full=True)[1]))
	z3full = np.polyfit(x, y, 3, full=True); residuals3 = z3full[1]
	print('Fitted poly a + bx + cx^2 + dx^3: %s' %(z3full[0]));print('Residuals^3 %f' %(np.polyfit(x, y, 3,full=True)[1]))
	
	z = np.polyfit(x, y, 2)
	z3 = np.polyfit(x, y, 3)
	pol = np.poly1d(z) #pol(70)
	pol3 = np.poly1d(z3)
	
	popt, pcov  = curve_fit(func, x, y)
	print('Curve fit: %s' %popt)
	xnew = np.linspace(x.iloc[0], x.iloc[-1], x.iloc[-1]-x.iloc[0]+ 1)
	_ = plt.plot(x, y, '.', xnew, pol(xnew), '-')
	#plt.plot(x, y, 'bo')
	#plt.plot(xnew, func(xnew, *popt), 'r--')
	fig_file = os.path.join(figures_dir, 'fittedcurve2_'+str(subject_id)+'.png')
	plt.savefig(fig_file)
	# Cubic works badly, use Quadratic
	# popt3, pcov3  = curve_fit(func3, x, y)
	# print('Curve fit: %s' %popt3)
	# _ = plt.plot(x, y, '.', xnew, pol3(xnew), '-')
	# fig_file = os.path.join(figures_dir, 'fittedcurve3_'+str(subject_id)+'.png')
	# plt.savefig(fig_file)
	return zfull


def solve_ana_r(yearv1, remaining_vol, subject_id=None):
	"""
	"""
	max_yy = 19
	years = yearv1 - max_yy
	r = 1 - remaining_vol**(1/years)
	return r


def plot_curvebrainvol(yearv1, rate,subject_id=None):
	"""
	"""
	# Brain vol when y years
	
	at_yy_vol = (1-rate)**(yearv1-19)
	print("The brain vol remaining at age r = %d is %f " % (yearv1,at_yy_vol))
	year_ticks = np.arange(1,yearv1-19)
	e_curve = (1-rate)**year_ticks
	# Plot
	fig, ax = plt.subplots(figsize=(9,9))
	ax.set_ylim([0,1.1])
	label2plot = "r="+str(round(rate,4))
	plt.plot(year_ticks+19, e_curve, label=label2plot)
	plt.xlabel("Age ")

	plt.title("Estimated Brain Vol age 19 = %d subject_id %d " % (yearv1,subject_id))
	plt.ylabel("remaining brain vol. 19-v1")
	plt.grid()
	ax.legend()
	fig_file = os.path.join(figures_dir, 'ecurve_' + str(subject_id) + '.png')
	plt.savefig(fig_file)

def solve_num_r(yearv1, remaining_vol, subject_id=None):
	"""
	"""
	from scipy.optimize import fsolve
	#figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/healthy_only'
	max_yy = 19
	years = yearv1 - max_yy

	func =  lambda x: (1-x)**years - remaining_vol
	# Plot
	# x = np.linspace(0.0, 0.5, 10000)
	# label2plot = "Brain vol. loss:"+str(subject_id)
	# plt.plot(x, func(x),label=label2plot)
	# plt.xlabel("r (annual atrophy rate)")
	# plt.ylabel("remaining brain vol.")
	# plt.grid()
	# fig_file = os.path.join(figures_dir, 'r_' + str(subject_id) + '.png')
	# plt.grid(axis='x')
	# plt.savefig(fig_file)

	# Use the numerical solver to find the roots
	x_initial_guess = 0.0001
	x_solution = fsolve(func, x_initial_guess)
	print("The solution is r = %f  (func=%f), remaining vol %f" % (x_solution,func(x_solution),remaining_vol))

	# Brain vol when y years
	at_yy = 70
	at_yy_vol = (1-x_solution)**(at_yy-19)
	print("The brain vol remaining at age r = %d is %f " % (at_yy,at_yy_vol))
	year_ticks = np.arange(1,yearv1-max_yy)
	e_curve = (1-x_solution)**year_ticks
	# Plot
	fig, ax = plt.subplots(figsize=(9,9))
	ax.set_ylim([0,1.1])
	label2plot = "r="+str(x_solution[0])[0:7]
	plt.plot(year_ticks+max_yy, e_curve, label=label2plot)
	plt.xlabel("Age ")

	plt.title("Estimated Brain Vol age 19 = %d subject_id %d " % (yearv1,subject_id))
	plt.ylabel("remaining brain vol. 19-v1")
	plt.grid()
	ax.legend()
	fig_file = os.path.join(figures_dir, 'ecurve_' + str(subject_id) + '.png')
	plt.savefig(fig_file)
	return x_solution

def plot_ts_allsubjects(data, subject_id=None):
	"""plot_ts_allsubjects ploot time series from dataframe age bvol id as tiem series
	"""
	from pandas.plotting import andrews_curves

	if subject_id == None:
		# Many subjects	
		title = 'Est. Brain Vol. Loss'
		print('Plotting all ts subjects....\n')  
		fig_file = os.path.join(figures_dir, 'megadf_.png')
		data.to_csv(os.path.join(figures_dir, 'megadf_.csv'))
	else:
		# 1 subject
		title = 'Est. Brain Vol. Loss id:' + str(subject_id)
		print('Plotting 1 ts subject: %s \n' %str(subject_id))
		fig_file = os.path.join(figures_dir, 'loss_ts_id_' + str(subject_id) +'.png')

	fig, ax = plt.subplots(figsize=(12, 12))

	# multiline plot with group by
	ax.set_xlabel('Age')
	ax.set_title(title)
	ax.set_ylabel('Brain Vol. [0,1]')
	ax.axvline(x=19,linewidth=1, color='r', linestyle = '--')
	ax.grid()
	c={0.0: 'blue', 1.0: 'orange', 2.0:'red'};
	for key, grp in data.groupby(['idpv']): 
		ax.plot(grp['age'], grp['bvol'], color=c[grp['dx_corto_visita5'][0]]) # label ="Temp in {0:02d}".format(key))
		ax.axvline(x=grp['age'].iloc[1],  linewidth=0.5, color='grey', linestyle = '--')
		ax.axvline(x=grp['age'].iloc[-1], linewidth=0.5, color='grey', linestyle = '--')
		#plt.legend(loc='best')  

	plt.savefig(fig_file)
	
def select_one_randonsubject(df, condition):
	"""
	"""
	gotid_longseg = True
	while gotid_longseg == True:
		#df_ts = get_timeseries(df, label, 1, 5)
		seednow = datetime.datetime.now().second
		subject = df.idpv.sample(1,random_state=seednow).values[0] # Select random idpv e.g. 6
		print('Trying subject id:%d ...' %subject)
		df_row = df.loc[df['idpv'] == subject]
		gotid_longseg = df_row[condition].isnull().values.any()
		time.sleep(1) 
	return subject


def timseseries_analysis():
	"""
	"""

	#figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/healthy_only'
	csv_path = '/Users/jaime/github/papers/atrophy_long/siena_lossandvelacc_18-10-2019longit_and_healthy.csv'
	csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019.csv'
	df = pd.read_csv(csv_path, sep=';')
	#df_orig = df.copy()
	df = accumulated_loss(df)
	df_orig = df.copy()
	label = 'accloss_'
	# Get list of all subjects with all time points
	listofsubjects = []
	megadf = pd.DataFrame()
	for subject in df.idpv:
		print('Trying subject id:%d ...' %subject)
		df_row = df.loc[df['idpv'] == subject]
		gotid_longseg = df_row[['edad_visita1','edad_visita2','edad_visita3','edad_visita4','edad_visita5','BrainSegVol_to_eTIV_y1']].isna().values.any()
		if gotid_longseg == False:
			listofsubjects.append(subject)
	for ix in listofsubjects:
		df_tsi = timeseries_df_per_subject(df, ix)
		#df_tsj = timeseries_df_per_subject(df, listofsubjects[2])
		megadf = pd.concat([megadf,df_tsi])
	# Plot figure all subjects  brain loss 
	plot_ts_allsubjects(megadf)

	# Select Randomly one subject with all segmentation	
	condition = ['edad_visita1','edad_visita2','edad_visita3','edad_visita4','edad_visita5','BrainSegVol_to_eTIV_y1']
	subject = select_one_randonsubject(df.reset_index(), condition)	
	print('Time Series analysis subject id= %d' % subject)
	# Plot figure 1 subject brain loss 
	df_ts = timeseries_df_per_subject(df.reset_index(), subject)
	plot_ts_allsubjects(df_ts, subject)
	
	subject = select_one_randonsubject(df.reset_index(), condition)	
	print('Time Series analysis subject id= %d' % subject)
	df_ts = timeseries_df_per_subject(df.reset_index(), subject)
	plot_ts_allsubjects(df_ts, subject)

	pdb.set_trace()

	# Curve fitting 19, visit1, siena years
	years = df_ts.age; ts = df_ts.bvol
	print('Calling to Qcurvefitting years %s bvol %s' %(years,ts))
	zfull = qcurvefitting(years, ts, subject)
	# Solve for r (interannual loss rate)
	yearv1 = df_ts.loc[1].age; remaining_vol= df_ts.loc[1].bvol
	#rnum = solve_num_r(yearv1, remaining_vol, subject)
	rana = solve_ana_r(yearv1, remaining_vol, subject)
	#print('r (annual loss) analytical= %f numerical = %f' %(rana, rnum))
	print('r (annual loss) analytical= %f ' %(rana))
	plot_curvebrainvol(yearv1, rana, subject)
	pdb.set_trace()

	fit_res = exponential_smoothing(df_ts, str(subject))
	print_exp_smoothing_results(fit_res)
	result_stat = stationarity_test(df_ts, subject, figures_dir)
	print('ADF test if p-value > 0.05 is non-stationary'); print('p-value: %f' % result[1])
	pdb.set_trace()
	ts = df_ts[subject]

	pdb.set_trace()
### END TEST ###

def main():
	np.random.seed(42)
	global figures_dir
	figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/healthy_only'
	#figures_dir = '/Users/jaime/github/papers/atrophy_long/figures'
	# #importlib.reload(JADr_paper);import atrophy_long; atrophy_long.main()
	print('Code for Atrophy Siena Longitudinal paper\n')
	plt.close('all')
	timseseries_analysis()
	return

	#csv_path = '~/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-07June2019.csv'
	csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-15October2019.csv'
	#csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/SienaViena-short-07102019-Test.csv'
	dataframe = pd.read_csv(csv_path, sep=';')
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()

	dataframe = convert_stringtofloat(dataframe)
	# remove outliers low, high 0.01, 0.99
	print('Prior to removing outliers %s', sum(dataframe['siena_12'].notnull()))
	dataframe = remove_outliers(dataframe)
	print('Post to removing outliers %s', sum(dataframe['siena_12'].notnull()))

	# Select only subjects with all visits (dx and siena) years 1 to 5
	#dataframe_longit = mask_visits(dataframe, 1, 5)
	dataframe_longit = mask_visits(dataframe, 1, 5, None)
	dataframe_longit_h = mask_visits(dataframe, 1, 5,'healthy')
	# Build accumulated loss columns only for & visits
	dataframe_longit = accumulated_loss(dataframe_longit)
	# Build accumulated loss columns only for all visits AND healthy
	dataframe_longit_h = accumulated_loss(dataframe_longit_h)	
	# Build accumulated loss columns only for all visits
	dataframe_all = accumulated_loss(dataframe)

	# Build atrophy velocity columns only call for dataframe_longit 'siena_vel_ij'
	dataframe_longit = velocity_interannual(dataframe_longit)
	dataframe_longit_h = velocity_interannual(dataframe_longit_h)
	dataframe_all = velocity_interannual(dataframe_all)

	#Build atrophy acceleration columns only call for dataframe_longit 'siena_velsquared_ij'
	dataframe_longit = acceleration_interannual(dataframe_longit)
	dataframe_longit_h = acceleration_interannual(dataframe_longit_h)
	dataframe_all = acceleration_interannual(dataframe_all)
	
	# save short (mmse, busche and demo)version dataframe with accloss and vel|acc columns 
	dfshort = save_df_csvfile(dataframe_longit, 'longit')
	dfshort_h = save_df_csvfile(dataframe_longit_h, 'longit_and_healthy')
	dfshort_all = save_df_csvfile(dataframe_all, 'crosssec')
	# When csv created we don't need the previous steps. Make sure convert_stringtofloat for accloss and vel
	# csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/siena_lossandvel_14-10-2019.csv'
	# dfshort = pd.read_csv(csv_path, sep=';')


	# Not call this block for dataframe_longit_h because there is a condition already
	condition = ['apoe', 'conversionmci', 'edad_visita1', 'nivel_educativo','edad_ultimodx','familial_ad']
	ix = 0
	df_cond_apoe = get_conditions_in_df(dfshort, condition[ix])
	df_cond_mci = get_conditions_in_df(dfshort, condition[1])
	df_cond_age1 = get_conditions_in_df(dfshort, condition[2])
	df_cond_edu = get_conditions_in_df(dfshort, condition[3])
	df_cond_agex = get_conditions_in_df(dfshort, condition[4])
	df_cond_fam = get_conditions_in_df(dfshort, condition[5])
	#df_cond_apoe['apoe0'] ...  df_cond['apoe12'].  df_cond_agex.keys()

	## Plot distribution single variable
	plot_distribution_variable_in_df(dfshort_h, 'siena_accloss_12', figures_dir)
	plot_distribution_variable_in_df(dfshort_h, 'siena_accloss_23', figures_dir)
	plot_distribution_variable_in_df(dfshort_h, 'siena_accloss_34', figures_dir)
	plot_distribution_variable_in_df(dfshort_h, 'siena_accloss_45', figures_dir)
	#figures_dir_nHn ='/Users/jaime/github/papers/atrophy_long/figures/not_healthy_only'
	# plot_distribution_variable_in_df(dfshort, 'siena_accloss_12', figures_dir_nHn)
	# plot_distribution_variable_in_df(dfshort, 'siena_accloss_23', figures_dir_nHn)
	# plot_distribution_variable_in_df(dfshort, 'siena_accloss_34', figures_dir_nHn)
	# plot_distribution_variable_in_df(dfshort, 'siena_accloss_45', figures_dir_nHn)

	## Scatter Plots
	# scatterplot i,i+1 siena dfshort_all(select columns all)  or dfshort (select columns longi)
	scatterplot_atrophy_i_iplus1(dfshort_h, None, figures_dir)
	scatterplot_atrophy_i_iplus1(dfshort_h, 'vel_', figures_dir)
	scatterplot_atrophy_i_iplus1(dfshort_h, 'accloss_', figures_dir)
	scatterplot_atrophy_i_iplus1(dfshort_h, 'velsquared_', figures_dir)
	#figures_dir_nHn ='/Users/jaime/github/papers/atrophy_long/figures/not_healthy_only'
	scatterplot_atrophy_i_iplus1(dfshort, None, figures_dir_nHn)
	scatterplot_atrophy_i_iplus1(dfshort, 'vel_', figures_dir_nHn)
	scatterplot_atrophy_i_iplus1(dfshort, 'accloss_', figures_dir_nHn)
	scatterplot_atrophy_i_iplus1(dfshort, 'velsquared_', figures_dir_nHn)

	print('Scatter plot of 2 variables in df...\n')
	scatterplot_2variables_in_df(dfshort_h,'siena_accloss_12','siena_accloss_23',figures_dir)
	scatterplot_2variables_in_df(dfshort_h,'siena_accloss_23','siena_accloss_34',figures_dir)
	scatterplot_2variables_in_df(dfshort_h,'siena_accloss_34','siena_accloss_45',figures_dir)
	scatterplot_2variables_in_df(dfshort_h,'siena_accloss_12','siena_accloss_45',figures_dir)
	scatterplot_2variables_in_df(dfshort_h,'siena_velsquared_12','siena_vel_23',figures_dir)

	## Joint Plots
	columns = ['apoe', 'conversionmci', 'siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45','siena_accloss_56','siena_accloss_67']
	label = 'accloss_'; hue = 'conversionmci'
	plot_joint_pairs(dfshort, columns[1:], label, hue, figures_dir)
	# plot_by_pair x_vars= [atrophy measure], y_var= phenotypic
	x_vars =  ['siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45','siena_accloss_56','siena_accloss_67']
	y_var = 'edad_ultimodx'
	plot_by_pair(dataframe, x_vars, y_var, figures_dir)
	y_var = 'sexo'
	plot_by_pair(dataframe, x_vars, y_var, figures_dir)
	y_var = 'apoe'
	plot_by_pair(dfshort, x_vars, y_var, figures_dir)


	## Plot correlation matrix

	xvars = ['siena_accloss_12','siena_accloss_23','siena_accloss_34','siena_accloss_45']
	yvars = xvars
	corr_matrix = plot_correlation_in_df(dfshort_h, xvars, yvars, figures_dir)
	yvars = ['fcsrtlibdem_visita1','fcsrtlibdem_visita2','fcsrtlibdem_visita3','fcsrtlibdem_visita4','fcsrtlibdem_visita5']
	corr_matrix = plot_correlation_in_df(dfshort_h, xvars, yvars, figures_dir)
	yvars = ['dx_corto_visita1','dx_corto_visita2','dx_corto_visita3','dx_corto_visita4','dx_corto_visita5']
	corr_matrix = plot_correlation_in_df(dfshort_h, xvars, yvars, figures_dir)

	## Time series
	# Longitudinal plot_timeseries_siena 
	plot_timeseries_siena(dfshort_h, None, figures_dir)
	dfts = plot_timeseries_siena(dfshort_h, 'accloss_', figures_dir)
	#accloss in 12 dfts.iloc[0].describe(). in 45 dfts.iloc[3].describe()
	plot_timeseries_siena(dfshort_h, 'vel_', figures_dir)
	plot_timeseries_siena(dfshort_h, 'velsquared_', figures_dir)
	# Box plot of times series	siena
	boxplot_ts(dfshort_h, None, figures_dir)
	boxplot_ts(dfshort_h, 'vel_', figures_dir)
	boxplot_ts(dfshort_h, 'accloss_', figures_dir)
	boxplot_ts(dfshort_h, 'velsquared_', figures_dir)
	pdb.set_trace()
	print('Estimated Volume brain left at 0.5 percent loss and years is:')
	compound_interest(100, -0.5, 5)
	
	###################################################

	
	
	## signal Processing: Random Walk Stationary modeling
	series = dfshort_h['siena_accloss_12']
	#series = dataframe_longit['siena_vel_12']
	random_walk_model_df(series)

	print('\n\n\n END!!!!')	 
if __name__ == "__name__":
	
	main()