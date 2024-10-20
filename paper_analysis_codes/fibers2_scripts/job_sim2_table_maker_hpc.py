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
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputpath', help='', type=str, default = None) #full path/filename
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)

    options=parser.parse_args(argv[1:])

    writepath = options.writepath +'output/'
    outputpath = options.outputpath
    random_seeds = options.random_seeds
    p_val = 0.05

    #Get names of all experiment folders
    experiment_folder_names = [name for name in os.listdir(writepath) if os.path.isdir(os.path.join(writepath, name))]
    print(experiment_folder_names)
    #Get names of all dataset folders (used within each experiment)
    dataset_folder_names = [name for name in os.listdir(writepath+experiment_folder_names[0]) if os.path.isdir(os.path.join(writepath+experiment_folder_names[0], name))]
    print(dataset_folder_names)

    significance_metrics = ['Accuracy','Number of P','Number of R','Ideal Iteration','Log-Rank Score','Adjusted HR','Group Ratio','Runtime']
    count_metrics = ['Ideal Bin','Ideal Threshold','TC1 Present']


    # PC vs NC EXPERIMENTS ***************************************
    #Table - PC DP 0
    table_name = 'Sim2_PC_DP_0'
    fixed_element = 'BasePC_i_10000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - PC DP 3
    table_name = 'Sim2_PC_DP_3'
    fixed_element = 'BasePC_i_10000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - NC DP 0
    table_name = 'Sim2_NC_DP_0'
    fixed_element = 'BaseNC_i_10000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_True'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - NC DP 3
    table_name = 'Sim2_NC_DP_3'
    fixed_element = 'BaseNC_i_10000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_True'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    # FEATURES EXPERIMENTS ***************************************
    #Table - Features 200 DP 0
    table_name = 'Sim2_Features_200_DP_0'
    fixed_element = 'Features_i_10000_tf_200_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Features 500 DP 0
    table_name = 'Sim2_Features_500_DP_0'
    fixed_element = 'Features_i_10000_tf_500_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Features 1000 DP 0
    table_name = 'Sim2_Features_1000_DP_0'
    fixed_element = 'Features_i_10000_tf_1000_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Features 200 DP 3
    table_name = 'Sim2_Features_200_DP_3'
    fixed_element = 'Features_i_10000_tf_200_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Features 500 DP 3
    table_name = 'Sim2_Features_500_DP_3'
    fixed_element = 'Features_i_10000_tf_500_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Features 1000 DP 3
    table_name = 'Sim2_Features_1000_DP_3'
    fixed_element = 'Features_i_10000_tf_1000_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    # BASENOISE EXPERIMENTS ***************************************
    #Table - BaseNoise 0.1 DP 0
    table_name = 'Sim2_BaseNoise_0.1_DP_0'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.1_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.2 DP 0
    table_name = 'Sim2_BaseNoise_0.2_DP_0'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.2_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.3 DP 0
    table_name = 'Sim2_BaseNoise_0.3_DP_0'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.3_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.4 DP 0
    table_name = 'Sim2_BaseNoise_0.4_DP_0'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.4_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.5 DP 0
    table_name = 'Sim2_BaseNoise_0.5_DP_0'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.5_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)


    #Table - BaseNoise 0.1 DP 3
    table_name = 'Sim2_BaseNoise_0.1_DP_3'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.1_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.2 DP 3
    table_name = 'Sim2_BaseNoise_0.2_DP_3'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.2_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.3 DP 3
    table_name = 'Sim2_BaseNoise_0.3_DP_3'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.3_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.4 DP 3
    table_name = 'Sim2_BaseNoise_0.4_DP_3'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.4_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - BaseNoise 0.5 DP 3
    table_name = 'Sim2_BaseNoise_0.5_DP_3'
    fixed_element = 'BaseNoise_i_10000_tf_100_p_6_t_0_n_0.5_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    # INSTANCES EXPERIMENTS ***************************************
    #Table - Instances 1000 DP 0
    table_name = 'Sim2_Instances_1000_DP_0'
    fixed_element = 'Instances_i_1000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Instances 500 DP 0
    table_name = 'Sim2_Instances_500_DP_0'
    fixed_element = 'Instances_i_500_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Instances 1000 DP 3
    table_name = 'Sim2_Instances_1000_DP_3'
    fixed_element = 'Instances_i_1000_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Instances 500 DP 3
    table_name = 'Sim2_Instances_500_DP_3'
    fixed_element = 'Instances_i_500_tf_100_p_6_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    # THRESHOLD EXPERIMENTS ***************************************
    #Table - Threshold 1 DP 0
    table_name = 'Sim2_Threshold_1_DP_0'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_1_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 2 DP 0
    table_name = 'Sim2_Threshold_2_DP_0'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_2_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 3 DP 0
    table_name = 'Sim2_Threshold_3_DP_0'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_3_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 4 DP 0
    table_name = 'Sim2_Threshold_4_DP_0'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_4_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 5 DP 0
    table_name = 'Sim2_Threshold_5_DP_0'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_5_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default','Fibers2.0_sim2_residuals','Fibers2.0_sim2_log_rank_residuals']
    baseline = 'Fibers2.0_sim2_default'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)


    #Table - Threshold 1 DP 3
    table_name = 'Sim2_Threshold_1_DP_3'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_1_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 2 DP 3
    table_name = 'Sim2_Threshold_2_DP_3'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_2_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 3 DP 3
    table_name = 'Sim2_Threshold_3_DP_3'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_3_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 4 DP 3
    table_name = 'Sim2_Threshold_4_DP_3'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_4_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)

    #Table - Threshold 5 DP 3
    table_name = 'Sim2_Threshold_5_DP_3'
    fixed_element = 'Threshold_i_10000_tf_100_p_6_t_5_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim2_default_DP','Fibers2.0_sim2_residuals_DP','Fibers2.0_sim2_log_rank_residuals_DP']
    baseline = 'Fibers2.0_sim2_default_DP'
    var_element_is_experiment = True
    stat_list_columns = ['Variable Element','Fixed Element','TC1 Present','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Adjusted HR','Group Ratio','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val)


def run_analysis(writepath,outputpath,significance_metrics,count_metrics,table_name,fixed_element,variable_element,baseline,var_element_is_experiment,stat_list_columns,p_val):
    dataframe_stat_list = []
    raw_dataframes = []
    baseline_index = variable_element.index(baseline)

    for var in variable_element:
        if var_element_is_experiment:
            master_summary = writepath+var+'/'+fixed_element+'/summary/'+fixed_element+'_master_summary.csv'
            summary = writepath+var+'/'+fixed_element+'/summary/'+fixed_element+'_summary.csv'
        else:
            master_summary = writepath+fixed_element+'/'+var+'/summary/'+var+'_master_summary.csv'
            summary = writepath+fixed_element+'/'+var+'/summary/'+var+'_summary.csv'
        # Load the stats summary CSV file into a pandas DataFrame
        df_master = pd.read_csv(master_summary)
        formated_df = format_data(df_master,stat_list_columns, var, fixed_element)
        dataframe_stat_list.append(formated_df)
        #Load the 30 random seed data
        df_sum = pd.read_csv(summary)
        raw_dataframes.append(df_sum)

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

def format_data(df,stat_list_columns, var, fixed_element):
    experiment = []
    #experiment.append(df.loc[0,'Dataset'])
    experiment.append(var)
    experiment.append(fixed_element)
    experiment.append("'"+str(df.loc[0,'TC1 Present'])+'/30') #Top bin
    experiment.append(str(round(df.loc[0,'Accuracy'],3))+' ('+str(round(df.loc[0,'Accuracy (SD)'],3))+')') #Accuracy
    experiment.append(str(round(df.loc[0,'Number of P'],3))+' ('+str(round(df.loc[0,'Number of P (SD)'],3))+')') #Num Pred
    experiment.append(str(round(df.loc[0,'Number of R'],3))+' ('+str(round(df.loc[0,'Number of R (SD)'],3))+')') #Num Rand
    experiment.append("'"+str(df.loc[0,'Ideal Bin'])+'/30') #Top bin
    experiment.append(str(round(df.loc[0,'Iteration of Ideal Bin'],2))+' ('+str(round(df.loc[0,'Iteration of Ideal Bin (SD)'],2))+')') #Ideal Iter
    experiment.append("'"+str(df.loc[0,'Ideal Threshold'])+'/30') #True Thresh
    experiment.append(str(round(df.loc[0,'Threshold'],1))+' ('+str(round(df.loc[0,'Threshold (SD)'],1))+')') #log rank
    experiment.append(str(round(df.loc[0,'Log-Rank Score'],1))+' ('+str(round(df.loc[0,'Log-Rank Score (SD)'],1))+')') #log rank
    experiment.append(str(round(df.loc[0,'Residual'],2))+' ('+str(round(df.loc[0,'Residual (SD)'],2))+')') #residual
    experiment.append(str(round(df.loc[0,'Adjusted HR'],3))+' ('+str(round(df.loc[0,'Adjusted HR (SD)'],3))+')') #adj HR
    experiment.append(str(round(df.loc[0,'Group Ratio'],2))+' ('+str(round(df.loc[0,'Group Ratio (SD)'],2))+')') #Group ratio
    experiment.append(str(round(df.loc[0,'Runtime'],2))+' ('+str(round(df.loc[0,'Runtime (SD)'],2))+')') #runtime

    new_df = pd.DataFrame([experiment], columns=stat_list_columns)
    return new_df


if __name__=="__main__":
    sys.exit(main(sys.argv))
