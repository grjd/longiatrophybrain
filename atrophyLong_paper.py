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

figures_dir = '/Users/jaime/github/papers/atrophy_long/figures/'


def scatter_plot_all(df):
	"""Scatter plot all points x: 21,32,43 y: 32, 43, 54 
	"""
	fig, ax = plt.subplots(figsize=(9,9))
	s1 = df['siena_vel_12']; s2 = df['siena_vel_23'];s3 = df['siena_vel_34'];s4 = df['siena_vel_45']

	delta_i = pd.concat([s1, s2, s3], axis=0, ignore_index=True)
	delta_plusi = pd.concat([s2, s3, s4], axis=0, ignore_index=True)
	frame = {'i':delta_i, 'iplus1':delta_plusi}
	df = pd.DataFrame(frame)
	df = df.apply (pd.to_numeric, errors='coerce')
	df = df.dropna()
	#regression part
	slope, intercept, r_value, p_value, std_err = stats.linregress(df.i,df.iplus1)
	line = slope*df.i+intercept
	plt.plot(df.i, line, 'r', label='y={:.2f}x {:.2f}'.format(slope,intercept))
	#scatter plot part
	plt.scatter(df.i,df.iplus1, s=3, alpha=0.5)
	plt.title(r'$Scatter plot \Delta_i \Delta_{i+1}$')
	plt.xlabel(r'$ \Delta_{i,i+1} (i=1..3) $')
	plt.ylabel(r'$ \Delta_{i,i+1} (i=2...4)$')
	plt.legend(fontsize=9)
	fig_file = os.path.join(figures_dir, 'scatter_all.png')
	plt.savefig(fig_file)
	return


def plot_timeseries_siena(df1,df2, labels):
	"""
	"""
	fig, ax = plt.subplots()
	s12 = df1['siena_vel_12'];s23 = df1['siena_vel_23'];s34 = df1['siena_vel_34'];s45 = df1['siena_vel_45']
	t12 = df2['siena_vel_12'];t23 = df2['siena_vel_23'];t34 = df2['siena_vel_34'];t45 = df2['siena_vel_45']
	ts_gr1 = pd.concat([s12,s23,s34,s45], axis=1); ts_gr2 = pd.concat([t12,t23,t34,t45], axis=1)
	for i in range(len(ts_gr1)):plt.plot(ts_gr1.iloc[i,:])
	fig_file = os.path.join(figures_dir, labels[0])
	plt.savefig(fig_file)
	fig, ax = plt.subplots()
	for i in range(len(ts_gr2)):plt.plot(ts_gr2.iloc[i,:])
	fig_file = os.path.join(figures_dir, labels[1])
	plt.savefig(fig_file)
	
	#ts_gr2.iloc[0].plot(kind='line', legend=False)


def compare_groups_ttest(grp1, grp2,the_file,class2cmp):
	"""
	"""
	from scipy.stats import ttest_ind
	tstat, pval = ttest_ind(grp1, grp2)
	print('tstat and pval %s %s' %(tstat, pval))
	the_file.write('ttest for classes:' + class2cmp + ' tstat=' + str(tstat) + ' pval=' + str(pval) + '\n')
	return 

def compare_conditions(df):
	"""
	"""
	
	# apoe mask
	mask_a0 = df['apoe']==0; df_a0 = df[mask_a0]
	mask_a1 = df['apoe']==1; df_a1 = df[mask_a1]
	mask_a2 = df['apoe']==2; df_a2 = df[mask_a2]
	mask_a12 = df['apoe']>=1; df_a12 = df[mask_a12]
	plot_distribution_atrophy(df_a0, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/apoe0')
	plot_distribution_atrophy(df_a1, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/apoe1')
	plot_distribution_atrophy(df_a2, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/apoe2')
	plot_distribution_atrophy(df_a12, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/apoe12')
	#mci mask 
	mask_mci0 = df['conversionmci']==0; df_mci0 = df[mask_mci0]
	mask_mci1 = df['conversionmci']==1; df_mci1 = df[mask_mci1]

	plot_distribution_atrophy(df_mci0, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/mci0')
	plot_distribution_atrophy(df_mci1, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/mci1')
	#sexo mask
	mask_male = df['sexo']==0; df_male = df[mask_male]
	mask_female = df['sexo']==1; df_female = df[mask_female]
	plot_distribution_atrophy(df_male, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/male')
	plot_distribution_atrophy(df_female, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/female')
	#education mask 
	mask_educa01 = df['nivel_educativo']<=1; df_educa01 = df[mask_educa01]
	mask_educa23 = df['nivel_educativo']>1; df_educa23 = df[mask_educa23]
	plot_distribution_atrophy(df_educa01, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/educa01')
	plot_distribution_atrophy(df_educa23, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/educa23')
	#age mask 
	mask_edad70 = df['edad_visita1']<80; df_edad70 = df[mask_edad70]
	mask_edad80 = df['edad_visita1']>=80; df_edad80 = df[mask_edad80]
	plot_distribution_atrophy(df_edad70, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/edad70')
	plot_distribution_atrophy(df_edad80, figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit/edad80')	


	# compare groups ttest 
	reportfile = '/Users/jaime/github/papers/atrophy_long/figures/longit/reportfile.txt'
	file_hld= open(reportfile,"w+")
	sienas = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45']
	for siena in sienas:
		class2cmp = 'APOE:0,12-' + siena
		compare_groups_ttest(df_a0[siena], df_a12[siena],file_hld, class2cmp)
		class2cmp = 'MCI:0,1-' + siena
		compare_groups_ttest(df_mci0[siena], df_mci1[siena],file_hld, class2cmp)
		class2cmp = 'Sex:0,1-' + siena
		compare_groups_ttest(df_male[siena], df_female[siena],file_hld, class2cmp)
		class2cmp = 'Educa:01,23-' + siena
		compare_groups_ttest(df_educa01[siena], df_educa23[siena],file_hld, class2cmp)
		class2cmp = 'Age:<80,>=80-' + siena
		compare_groups_ttest(df_edad70[siena], df_edad80[siena],file_hld, class2cmp)
	file_hld.close() 

	# compare groups Bayesian

	# Plot time series
	plot_timeseries_siena(df_a0, df_a12, ['time_series_apoe0', 'time_series_apoe12'])
	plot_timeseries_siena(df_mci0, df_mci1, ['time_series_mci0', 'time_series_mci12'])
	plot_timeseries_siena(df_male, df_female, ['time_series_male', 'time_series_female'])
	plot_timeseries_siena(df_educa01, df_educa23, ['time_series_educa01', 'time_series_educa23'])
	plot_timeseries_siena(df_edad70, df_edad80, ['time_series_70', 'time_series_80'])
	return

def mask_visits(df, yi, ye):
	"""mask_visits(df,1,5) : select rows with dx_corto_visita_yi...ye notnull
	Select subjects with all visits rfom yi to y3, e (df,1,5) df[visits at 1&2&3&4&4&5)
	"""
	dx_yi = 'dx_corto_visita' + str(yi)

	maski = df[dx_yi].notnull()
	years = np.arange(yi+1,ye+1)
	for yy in years:
		label_yy = 'dx_corto_visita' + str(yy)
		mask = df[label_yy].notnull()
		print('Selecting 1 &...%s' %yy)
		maski = mask & maski
		print('Mask total Rows %s' %sum(maski))
	print('Final Mask DX nb of Rows: %s' %sum(maski))
	print('Shape masked df all visits::', df[maski].shape)
	# Mask subjects all first 5 visits (hard coded)
	mask_siena = df['siena_vel_12'].notnull() & df['siena_vel_23'].notnull() & df['siena_vel_34'].notnull() &  df['siena_vel_45'].notnull()
	print('Final Mask SIENA nb of Rows: %s' %sum(mask_siena))
	mask_dx_siena = maski & mask_siena
	# return dataframe of mask dx, siena or dx and sienas return df[maski]
	return df[mask_dx_siena]

def remove_outliers(df):
	"""
	"""
	df_orig = df.copy()
	low = .05
	high = .95
	siena_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	df_siena = df[siena_cols]
	quant_df = df_siena.quantile([low, high])
	
	print('Outliers: low= %.3f high= %.3f \n %s' %(low, high, quant_df))
	df_nooutliers = df_siena[(df_siena > quant_df.loc[low, siena_cols]) & (df_siena < quant_df.loc[high, siena_cols])]
	df[siena_cols] = df_nooutliers.to_numpy()
	return df


def plot_distribution_atrophy(dataframe, figures_dir=None):
	"""
	"""
	#Check 'siena_vel_34' there must be a 0 divide

	veloc_cols = ['conversionmci','siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	df2plt = dataframe[veloc_cols]
	try:
		print('Calling to pairplot atrophy velocity %s' % (veloc_cols))
		sns_plot = sns.pairplot(df2plt, hue='conversionmci', size=2.5)
		fig_file = os.path.join(figures_dir, 'veloc_pairplot.png')
		print('Saving sns pairplot atrophy velocity %s' % (fig_file))
		sns_plot.savefig(fig_file)

		veloc_cols = ['apoe','siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
		df2plt = dataframe[veloc_cols]
		print('Calling to pairplot atrophy velocity %s' % (veloc_cols))
		sns_plot = sns.pairplot(df2plt, hue='apoe', size=2.5)
		fig_file = os.path.join(figures_dir, 'apoeveloc_pairplot.png')
		print('Saving sns pairplot atrophy velocity %s' % (fig_file))
		sns_plot.savefig(fig_file)
	except ZeroDivisionError:
		print("pairplot empty for selection  !!! \n\n")


	age_veloc_cols = ['edad_ultimodx', 'conversionmci','siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	# pairplot age siena
	sns_plot = sns.pairplot(dataframe[age_veloc_cols], hue='conversionmci', x_vars=["siena_vel_12", "siena_vel_23", "siena_vel_34", "siena_vel_45", "siena_vel_56"], y_vars=["edad_ultimodx"],height=5, aspect=.8);
	fig_file = os.path.join(figures_dir, 'age_veloc_pairplot.png')	
	sns_plot.savefig(fig_file)
	# pairplot apoe siena
	apoe_veloc_cols = ['apoe', 'conversionmci','siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	sns_plot = sns.pairplot(dataframe[apoe_veloc_cols], hue='conversionmci', x_vars=["siena_vel_12", "siena_vel_23", "siena_vel_34", "siena_vel_45", "siena_vel_56"], y_vars=["apoe"],height=5, aspect=.8);
	fig_file = os.path.join(figures_dir, 'apoe_veloc_pairplot.png')	
	sns_plot.savefig(fig_file)
	# pairplot sex siena
	sex_veloc_cols = ['sexo', 'conversionmci','siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	sns_plot = sns.pairplot(dataframe[sex_veloc_cols], hue='conversionmci', x_vars=["siena_vel_12", "siena_vel_23", "siena_vel_34", "siena_vel_45", "siena_vel_56"], y_vars=["sexo"],height=5, aspect=.8);
	fig_file = os.path.join(figures_dir, 'sex_veloc_pairplot.png')	
	sns_plot.savefig(fig_file)


	# for ix in veloc_cols[1:]:
	# 	print('Plotting distplot %s' %(ix))
	# 	ax = sns.distplot(df2plt[ix].dropna(), kde=False, fit=stats.gamma)
	# 	fig = ax.get_figure()
	# 	fig_file = os.path.join(figures_dir, 'veloc_' + str(ix) + '.png')
	# 	fig.savefig(fig_file)
	#Scatter plots
	snsp = sns.jointplot(x='siena_vel_12', y='siena_vel_23', data=df2plt, kind="reg");
	fig_file = os.path.join(figures_dir, 'joint12-23.png')
	snsp.savefig(fig_file)
	snsp = sns.jointplot(x='siena_vel_23', y='siena_vel_34', data=df2plt, kind="reg");
	fig_file = os.path.join(figures_dir, 'joint23-34.png')
	snsp.savefig(fig_file)
	snsp = sns.jointplot(x='siena_vel_34', y='siena_vel_45', data=df2plt, kind="reg");
	fig_file = os.path.join(figures_dir, 'joint34-45.png')
	snsp.savefig(fig_file)
	snsp = sns.jointplot(x='siena_vel_45', y='siena_vel_56', data=df2plt, kind="reg");
	fig_file = os.path.join(figures_dir, 'joint45-56.png')
	snsp.savefig(fig_file)
	snsp = sns.jointplot(x='siena_vel_56', y='siena_vel_67', data=df2plt, kind="reg");
	fig_file = os.path.join(figures_dir, 'joint56-67.png')
	plt.savefig(fig_file)
	
	fig2, ax2 = plt.subplots()
	sns.distplot(df2plt['siena_vel_12'], hist=False, rug=True);
	fig_file = os.path.join(figures_dir, 'hist_12' + '.png')
	plt.axvline(df2plt['siena_vel_12'].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(df2plt['siena_vel_12'].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt['siena_vel_12'].mean()*1.1, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt['siena_vel_12'].mean(),df2plt['siena_vel_12'].std()))
	plt.savefig(fig_file)
	
	fig3, ax3 = plt.subplots()
	sns.distplot(df2plt['siena_vel_23'], hist=False, rug=True);
	fig_file = os.path.join(figures_dir, 'hist_23' + '.png')
	plt.axvline(df2plt['siena_vel_23'].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(df2plt['siena_vel_23'].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt['siena_vel_23'].mean()*1.1, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt['siena_vel_23'].mean(),df2plt['siena_vel_23'].std()))
	plt.savefig(fig_file)
	
	fig4, ax4 = plt.subplots()
	sns.distplot(df2plt['siena_vel_34'], hist=False, rug=True);
	fig_file = os.path.join(figures_dir, 'hist_34' + '.png')
	plt.axvline(df2plt['siena_vel_34'].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(df2plt['siena_vel_34'].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt['siena_vel_34'].mean()*1.1, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt['siena_vel_34'].mean(),df2plt['siena_vel_34'].std()))
	plt.savefig(fig_file)
	
	fig5, ax5 = plt.subplots()
	sns.distplot(df2plt['siena_vel_45'], hist=False, rug=True);
	fig_file = os.path.join(figures_dir, 'hist_45' + '.png')
	plt.axvline(df2plt['siena_vel_45'].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(df2plt['siena_vel_45'].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt['siena_vel_45'].mean()*1.1, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt['siena_vel_45'].mean(),df2plt['siena_vel_45'].std()))
	plt.savefig(fig_file)
	
	fig6, ax6 = plt.subplots()
	sns.distplot(df2plt['siena_vel_56'], hist=False, rug=True);
	fig_file = os.path.join(figures_dir, 'hist_56' + '.png')
	plt.axvline(df2plt['siena_vel_56'].mean(), color='k', linestyle='dashed', linewidth=1)
	plt.axvline(df2plt['siena_vel_56'].median(), color='r', linestyle='dashed', linewidth=.7)
	min_ylim, max_ylim = plt.ylim()
	plt.text(df2plt['siena_vel_56'].mean()*1.1, max_ylim*0.9, r'$\mu \pm \sigma: {:.3f} \pm {:.3f}$'.format(df2plt['siena_vel_56'].mean(),df2plt['siena_vel_56'].std()))
	plt.savefig(fig_file)

	atrophies = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56']
	conversions = ['dx_corto_visita1','dx_corto_visita2','dx_corto_visita3','dx_corto_visita4','dx_corto_visita5','dx_corto_visita6']
	
	corr_matrix = dataframe[atrophies+conversions].corr()
	# velocity atrophy matrix
	atrophy_corr = corr_matrix.iloc[0:5,0:5]
	mask = np.zeros_like(atrophy_corr)
	mask[np.triu_indices_from(mask)] = True
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(atrophy_corr,mask=mask,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heat_atrophy_corr' + '.png')
	plt.savefig(fig_file)
	# dx vs dx
	dx_corr = corr_matrix.iloc[5:,5:]
	mask = np.zeros_like(dx_corr)
	mask[np.triu_indices_from(mask)] = True
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(dx_corr,mask=mask,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heat_dx_corr' + '.png')
	plt.savefig(fig_file)
	# atrophy vs dx
	atro_cog_corr = corr_matrix.iloc[0:5,5:]
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(atro_cog_corr,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heat_atrophydx_corr' + '.png')
	plt.savefig(fig_file)
	# remove year 6
	atro_cog5_corr = corr_matrix.iloc[0:4,5:10]
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(atro_cog5_corr,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heat_atrophydx1-5_corr' + '.png')
	plt.savefig(fig_file)
	return
		


def convert_stringtofloat(dataframe):
	"""
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
		#pdb.set_trace()
		dataframe[ix] = dataframe[ix].str.replace(',','.').astype(float)
	veloc_cols = ['siena_vel_12','siena_vel_23','siena_vel_34','siena_vel_45','siena_vel_56','siena_vel_67']
	dataframe['siena_vel_12'] = dataframe['siena_12']/dataframe['tpo1.2']
	dataframe['siena_vel_23'] = dataframe['siena_23']/ (dataframe['tpo1.3']-dataframe['tpo1.2'])
	dataframe['siena_vel_34'] = dataframe['siena_34']/ (dataframe['tpo1.4']-dataframe['tpo1.3'])
	dataframe['siena_vel_45'] = dataframe['siena_45']/ (dataframe['tpo1.5']-dataframe['tpo1.4'])
	dataframe['siena_vel_56'] = dataframe['siena_56']/ (dataframe['tpo1.6']-dataframe['tpo1.5'])
	dataframe['siena_vel_67'] = dataframe['siena_67']/ (dataframe['tpo1.7']-dataframe['tpo1.6'])

	return dataframe


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
def main():
	np.random.seed(42)
	#importlib.reload(JADr_paper);import atrophy_long; atrophy_long.main()
	print('Code for Atrohy Siena Longitudinal paper\n')
	# open csv with pv databse
	plt.close('all')
	#csv_path = '~/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-07June2019.csv'
	csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-01October2019.csv'
	dataframe = pd.read_csv(csv_path, sep=';')
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	# Build atrophy velocity = atrophy/ year
	dataframe = convert_stringtofloat(dataframe)
	# remove outliers low, high 0.05, 0.95
	dataframe = remove_outliers(dataframe)
	# scatter plot all
	scatter_plot_all(dataframe)
	# Select only subjets with all visits (dx and siena) years 1 to 5
	dataframe_longit = mask_visits(dataframe, 1, 5)

	plot_distribution_atrophy(dataframe_longit,figures_dir='/Users/jaime/github/papers/atrophy_long/figures/longit')
	plot_distribution_atrophy(dataframe, figures_dir)

	compare_conditions(dataframe_longit)

	print('\n\n\n END!!!!')	 
if __name__ == "__name__":
	
	main()