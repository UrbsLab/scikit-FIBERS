import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import accuracy_score
sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
from scipy.stats import wilcoxon


def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--o', dest='outputpath', help='', type=str, default = 'myOutputPath') #full path/filename
    parser.add_argument('--cv', dest='cv', help='cross validation partitions', type=int, default= 10)

    options=parser.parse_args(argv[1:])
    outputpath = options.outputpath
    cv = options.cv
    loci_list = 'A,B,C,DRB1,DRB345,DQA1,DQB1'
    loci_list = loci_list.split(',')
    p_val = 0.05

    significance_metrics = ['Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residual','Threshold','Group Ratio','Count At/Below Threshold','Count Above Threshold','Bin Size','Birth Iteration','Runtime']


    # CORE EXPERIMENTS ***************************************
    #_DP3
    #_DP3_B10
    #_DP3_B5
    #_DP3_T0
    #_DP3_T0_B10
    #_DP3_T0_B5

    #UNFILTERED ****************************************
    # AT *******
    #Table - FIBERS2 Fit compare _DP3_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_B10_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_B10_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_B10_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_B5_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_B5_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B5_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter','Fibers2.0_hla_7locus_product_DP3_B5_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    # T=0 *******
    #Table - FIBERS2 Fit compare _DP3_T0_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_T0_B10_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0_B10_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_T0_B5_nofilter
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0_B5_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B5_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #UNFILTERED COMPARE CONFIGS ****************************************
    #Table - FIBERS2 log-rank compare unfiltered
    table_name = 'Real_7Locus_FIBERS2_logrank_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_nofilter','Fibers2.0_hla_7locus_logrank_DP3_B10_nofilter','Fibers2.0_hla_7locus_logrank_DP3_B5_nofilter','Fibers2.0_hla_7locus_logrank_DP3_T0_nofilter','Fibers2.0_hla_7locus_logrank_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_logrank_DP3_T0_B5_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 residuals compare unfiltered
    table_name = 'Real_7Locus_FIBERS2_residuals_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product compare unfiltered
    table_name = 'Real_7Locus_FIBERS2_product_nofilter'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_B5_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #FILTERED ****************************************
    # AT *******
    #Table - FIBERS2 Fit compare _DP3 
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_product_DP3']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_B10 
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_B10'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B10','Fibers2.0_hla_7locus_residuals_DP3_B10','Fibers2.0_hla_7locus_product_DP3_B10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_B5 
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B5','Fibers2.0_hla_7locus_residuals_DP3_B5','Fibers2.0_hla_7locus_product_DP3_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    # T=0 *******
    #Table - FIBERS2 Fit compare _DP3_T0
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_T0_B10
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0_B10'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B10','Fibers2.0_hla_7locus_residuals_DP3_T0_B10','Fibers2.0_hla_7locus_product_DP3_T0_B10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare _DP3_T0_B5
    table_name = 'Real_7Locus_FIBERS2_Fit_DP3_T0_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B5','Fibers2.0_hla_7locus_residuals_DP3_T0_B5','Fibers2.0_hla_7locus_product_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #FILTERED COMPARE CONFIGS ****************************************
    #Table - FIBERS2 log-rank compare filtered
    table_name = 'Real_7Locus_FIBERS2_logrank'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_logrank_DP3_B10','Fibers2.0_hla_7locus_logrank_DP3_B5','Fibers2.0_hla_7locus_logrank_DP3_T0','Fibers2.0_hla_7locus_logrank_DP3_T0_B10','Fibers2.0_hla_7locus_logrank_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 residuals compare filtered
    table_name = 'Real_7Locus_FIBERS2_residuals'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_residuals_DP3_B10','Fibers2.0_hla_7locus_residuals_DP3_B5','Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0_B10','Fibers2.0_hla_7locus_residuals_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product compare filtered
    table_name = 'Real_7Locus_FIBERS2_product'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_7locus_product_DP3_B10','Fibers2.0_hla_7locus_product_DP3_B5','Fibers2.0_hla_7locus_product_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0_B10','Fibers2.0_hla_7locus_product_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #NoAg UNFILTERED ****************************************
    #_DP3_nofilter_NoAg
    #_DP3_B10_nofilter_NoAg
    #_DP3_B5_nofilter_NoAg
    #_DP3_T0_nofilter_NoAg
    #_DP3_T0_B10_nofilter_NoAg
    #_DP3_T0_B5_nofilter_NoAg

    #Table - FIBERS2 _residuals_DP3_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_B10_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_B10_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_B10_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_B10_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_B10_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_B5_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_B5_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter','Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_B5_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_B5_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_B5_nofilter','Fibers2.0_hla_7locus_product_DP3_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_B10_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_B10_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_B10_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_B10_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_B5_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_B5_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_B5_nofilter_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_B5_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #NoAg UNFILTERED Config Compare ****************************************
    #_DP3_nofilter_NoAg
    #_DP3_B10_nofilter_NoAg
    #_DP3_B5_nofilter_NoAg
    #_DP3_T0_nofilter_NoAg
    #_DP3_T0_B10_nofilter_NoAg
    #_DP3_T0_B5_nofilter_NoAg

    #Table - FIBERS2 residuals compare unfiltered NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_B10_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_B5_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_nofilter_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product compare unfiltered NoAg
    table_name = 'Real_7Locus_FIBERS2_product_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_B10_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_B5_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_B10_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_B5_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_nofilter_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #NoAG FILTERED ****************************************
    #_DP3_NoAg
    #_DP3_B10_NoAg
    #_DP3_B5_NoAg
    #_DP3_T0_NoAg
    #_DP3_T0_B10_NoAg
    #_DP3_T0_B5_NoAg


    #Table - FIBERS2 _residuals_DP3_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_residuals_DP3_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_7locus_product_DP3_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_B10_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_B10','Fibers2.0_hla_7locus_residuals_DP3_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_B10_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_B10','Fibers2.0_hla_7locus_product_DP3_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_B5_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_B5_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_B5','Fibers2.0_hla_7locus_residuals_DP3_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_B5_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_B5_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_B5','Fibers2.0_hla_7locus_product_DP3_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_B10_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_B10','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_B10_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_B10','Fibers2.0_hla_7locus_product_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP3_T0_B5_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP3_T0_B5_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_B5','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_T0_B5_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP3_T0_B5_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_B5','Fibers2.0_hla_7locus_product_DP3_T0_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #NoAg FILTERED COMPARE CONFIGS ****************************************

    #Table - FIBERS2 residuals compare filtered NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_NoAg','Fibers2.0_hla_7locus_residuals_DP3_B10_NoAg','Fibers2.0_hla_7locus_residuals_DP3_B5_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product compare filtered NoAg
    table_name = 'Real_7Locus_FIBERS2_product_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_NoAg','Fibers2.0_hla_7locus_product_DP3_B10_NoAg','Fibers2.0_hla_7locus_product_DP3_B5_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_B10_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_B5_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    """
    #Table - FIBERS2 residuals_T0_nofilter compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3_T0_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product_T0_nofilter compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP_3_T0_nofilter_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_nofilter_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_nofilter'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 _residuals_DP_3_NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_residuals_DP3_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 _product_DP3_NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP_3_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_7locus_product_DP3_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 residuals_T0_B10 compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3_T0_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0_B10','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product_T0_B10 compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP_3_T0_B10_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0_B10','Fibers2.0_hla_7locus_product_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 residuals_T0 compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3_T0_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product_T0 compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP_3_T0_NoAg'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 residuals scenarios compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3_NoAgCompare' 
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_NoAg','Fibers2.0_hla_7locus_residuals_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_NoAg', 'Fibers2.0_hla_7locus_residuals_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_residuals_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product scenarios compare DP 3 NoAg
    table_name = 'Real_7Locus_FIBERS2_product_DP_3_NoAgCompare' 
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_NoAg','Fibers2.0_hla_7locus_product_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_NoAg', 'Fibers2.0_hla_7locus_product_DP3_nofilter_NoAg','Fibers2.0_hla_7locus_product_DP3_T0_B10_NoAg']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_NoAg'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 Fit compare DP 3 T0_B5
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_T0_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B5','Fibers2.0_hla_7locus_residuals_DP3_T0_B5','Fibers2.0_hla_7locus_product_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 3 B5
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B5','Fibers2.0_hla_7locus_residuals_DP3_B5','Fibers2.0_hla_7locus_product_DP3_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 3 T0_B5 NoAg
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_T0_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B5','Fibers2.0_hla_7locus_residuals_DP3_T0_B5','Fibers2.0_hla_7locus_product_DP3_T0_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 3 B5 NoAg
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_B5'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_B5','Fibers2.0_hla_7locus_residuals_DP3_B5','Fibers2.0_hla_7locus_product_DP3_B5']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_B5'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - FIBERS2 Fit compare DP 3
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_product_DP3']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 3 T0
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_T0'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 3 T0_B10
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_T0_B10'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T0_B10','Fibers2.0_hla_7locus_residuals_DP3_T0_B10','Fibers2.0_hla_7locus_product_DP3_T0_B10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T0_B10'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 log-rank compare DP 3
    table_name = 'Real_7Locus_FIBERS2_logrank_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_logrank_DP3_nofilter','Fibers2.0_hla_7locus_logrank_DP3_T0','Fibers2.0_hla_7locus_logrank_DP3_T0_nofilter','Fibers2.0_hla_7locus_logrank_DP3_T0_B10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 residuals compare DP 3
    table_name = 'Real_7Locus_FIBERS2_residuals_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0','Fibers2.0_hla_7locus_residuals_DP3_T0_nofilter','Fibers2.0_hla_7locus_residuals_DP3_T0_B10']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 product compare DP 3
    table_name = 'Real_7Locus_FIBERS2_product_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_7locus_product_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3_T0','Fibers2.0_hla_7locus_product_DP3_T0_nofilter','Fibers2.0_hla_7locus_product_DP3_T0_B10']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Test Adjusted HR','Test NoAg Adjusted HR','Test Unadjusted HR','Train Adjusted HR','Train NoAg Adjusted HR','Train Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)
    """


def run_analysis(outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val):
    dataframe_stat_list = []
    raw_dataframes = []
    baseline_index = variable_element.index(baseline)

    for var in variable_element:
        #master_summary = writepath+var+'/'+var+'_master_summary.csv'
        summary = outputpath+'/'+var+'/'+var+'_CV_summary.csv'

        # Load the stats summary CSV file into a pandas DataFrame
        df_sum = pd.read_csv(summary)
        raw_dataframes.append(df_sum)
        formated_df = format_data(df_sum,stat_list_columns, var)
        dataframe_stat_list.append(formated_df) # Add a row of results for each experiment

    #now have the basic stats collected for each experiment and dataset
    # Determine statistical significance differences for significance_metrics 
    # add '* next to any non-baseline results where a significant difference observed on contrast with baseline
    for metric in significance_metrics:
        #Get baseline data for metric
        base_col = raw_dataframes[baseline_index][metric]
        for i in range(len(variable_element)):
            #Get comparison data for metric
            if i != baseline_index: #don't include baseline data
                compare_col = raw_dataframes[i][metric]
                #Apply Wilcoxon Significance comparison
                try:
                    is_sig = wilcoxon_sig(base_col,compare_col,p_val)
                    if is_sig: # indicate significance within dataframe stat_list
                        # Find appropriate metric
                        #cell_value = str(dataframe_stat_list[i][metric])
                        #dataframe_stat_list[i][metric] = str(dataframe_stat_list[i][metric])+'*'
                        dataframe_stat_list[i][metric] += '*'
                except ValueError as e:
                    print(f"Error while performing the wilcoxon test: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

    # combine experiment results into a single dataframe
    combined_df = pd.concat(dataframe_stat_list, ignore_index=False)
    #transpose for easy copying
    combined_df_T = combined_df.T
    combined_df_T.to_csv(outputpath+'/'+str(table_name)+'_Table.csv', index=False)

def wilcoxon_sig(col1,col2,p_val):
    statistic, p_value = wilcoxon(col1, col2)
    print('Comparison: '+str(statistic)+ ' '+ str(p_value))
    if p_value <= p_val:
        return True

def format_data(df,stat_list_columns, var):
    #significance_metrics = ['Adjusted HR','Unadjusted HR','Log-Rank Score','Residual','Threshold','Group Ratio','Count At/Below Threshold','Count Above Threshold','Bin Size','Birth Iteration','Runtime']
    experiment = []
    experiment.append(var)
    experiment.append(str(round(df['Test Adjusted HR'].mean(),3))+' ('+str(round(df['Test Adjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Test NoAg Adjusted HR'].mean(),3))+' ('+str(round(df['Test NoAg Adjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Test Unadjusted HR'].mean(),3))+' ('+str(round(df['Test Unadjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Train Adjusted HR'].mean(),3))+' ('+str(round(df['Train Adjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Train NoAg Adjusted HR'].mean(),3))+' ('+str(round(df['Train NoAg Adjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Train Unadjusted HR'].mean(),3))+' ('+str(round(df['Train Unadjusted HR'].std(),3))+')') #adj HR                      
    experiment.append(str(round(df['Log-Rank Score'].mean(),1))+' ('+str(round(df['Log-Rank Score'].std(),1))+')') #log rank 
    experiment.append(str(round(df['Residual'].mean(),2))+' ('+str(round(df['Residual'].std(),2))+')') #Residual
    experiment.append(str(round(df['Threshold'].mean(),1))+' ('+str(round(df['Threshold'].std(),1))+')') #Threshold
    experiment.append(str(round(df['Group Ratio'].mean(),2))+' ('+str(round(df['Group Ratio'].std(),2))+')') #Group Ratio
    experiment.append(str(round(df['Count At/Below Threshold'].mean(),0))+' ('+str(round(df['Count At/Below Threshold'].std(),0))+')') #Count At/Below Threshold
    experiment.append(str(round(df['Count Above Threshold'].mean(),0))+' ('+str(round(df['Count Above Threshold'].std(),0))+')') #Count Above Threshold
    experiment.append(str(round(df['Bin Size'].mean(),1))+' ('+str(round(df['Bin Size'].std(),1))+')') #Bin Size
    experiment.append(str(round(df['Birth Iteration'].mean(),1))+' ('+str(round(df['Birth Iteration'].std(),1))+')') #Birth Iteration
    experiment.append(str(round(df['Runtime'].mean()/60,1))+' ('+str(round(df['Runtime'].std()/60,1))+')') #Runtime

    new_df = pd.DataFrame([experiment], columns=stat_list_columns)
    return new_df


if __name__=="__main__":
    sys.exit(main(sys.argv))
