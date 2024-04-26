import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN
def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--d', dest='dataset', help='name of data path (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputPath', help='', type=str, default = 'myOutputPath') #full path/filename
    parser.add_argument('--r', dest='random_seeds', help='random seeds in experiment', type=str, default='None')

    options=parser.parse_args(argv[1:])
    dataset = options.dataset
    outputPath = options.outputPath
    random_seeds = int(options.random_seeds)

    ideal_count = 10
    ideal_threshold = 0
    algorithm = "FIBERS 2.0"
    experiment = "TestAnalysis"

    # Get dataset Name
    filename = os.path.basename(dataset)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(dataset)

    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    #Define columns for replicate results summary:
    columns = ["Bin Features", "Threshold", "Fitness", "Pre-Fitness", "Log-Rank Score","Log-Rank p-value",
               "Bin Size", "Group Ratio", "Count At/Below Threshold", "Count Above Threshold", "Birth Iteration", 
               "Deletion Probability", "Cluster", "Residual", "Residual p-value", "Unadjusted HR", "Unadjusted HR CI",
               "Unadjusted HR p-value", "Adjusted HR", "Adjusted HR CI", "Adjusted HR p-value", "Number of P", 
               "Number of R", "Ideal Iteration", "Accuracy", "Runtime", "Dataset Filename"]
    df = pd.DataFrame(columns=columns)

    #Make intial lists to store metrics across replications
    accuracy = []
    num_P = []
    num_R = []
    ideal = 0
    ideal_iter = []
    ideal_thresh = 0
    threshold = []
    log_rank = []
    residuals = []
    unadj_HR = []
    adj_HR = []
    group_balance = []
    runtime = []
    tc = 0
    bin_size = []
    birth_iteration = []

    #Create top bin summary across replicates
    for random_seed in range(0, random_seeds):  #for each replicate
        #Unpickle FIBERS Object
        with open(outputPath+'/'+dataset_name+'_'+str(random_seed)+'_fibers.pickle', 'rb') as f:
            fibers = pickle.load(f)
        #Get top bin object for current fibers population
        bin_index = 0 #top bin
        bin = fibers.set.bin_pop[bin_index]
        results_list = [bin.feature_list, bin.group_threshold, bin.fitness, bin.pre_fitness, bin.log_rank_score,
                        bin.log_rank_p_value, bin.bin_size, bin.group_strata_prop, bin.count_bt, bin.count_at, 
                        bin.birth_iteration, bin.deletion_prop, bin.cluster, bin.residuals_score, bin.residuals_p_value,
                        bin.HR, bin.HR_CI, bin.HR_p_value, bin.adj_HR, bin.adj_HR_CI, bin.adj_HR_p_value, 
                        str(bin.feature_list).count('P'), str(bin.feature_list).count('R'), 
                        ideal_iteration(ideal_count, bin.feature_list, bin.birth_iteration),
                        accuracy_score(fibers.predict(data,bin_number=bin_index),true_risk_group) if true_risk_group is not None else None,
                        fibers.elapsed_time, dataset_name] 
        df.loc[len(df)] = results_list

        #Update metric lists
        accuracy.append(accuracy_score(fibers.predict(data,bin_number=bin_index),true_risk_group) if true_risk_group is not None else None)
        num_P.append(str(bin.feature_list).count('P'))
        num_R.append(str(bin.feature_list).count('R'))
        if ideal_iteration(ideal_count, bin.feature_list, bin.birth_iteration) != None:
            ideal += 1
            ideal_iter.append(bin.birth_iteration)
        if bin.group_threshold == ideal_threshold:
            ideal_thresh += 1
        threshold.append(bin.group_threshold)
        if bin.log_rank_score != None:
            log_rank.append(bin.log_rank_score)
        if bin.residuals_score != None:
            residuals.append(bin.residuals_score)
        if bin.HR != None:
            unadj_HR.append(bin.HR)
        if bin.adj_HR != None:
            adj_HR.append(bin.adj_HR)
        group_balance.append(bin.group_strata_prop)
        runtime.append(fibers.elapsed_time)
        if str(bin.feature_list).count('T') > 0:
            tc += 1
        bin_size.append(bin.bin_size)
        birth_iteration.append(bin.birth_iteration)

    #Save replicate results as csv
    df.to_csv(outputPath+'/'+dataset_name+'_summary'+'.csv', index=False)

    #Generate experiment summary 'master list'
    master_columns = ["Algorithm","Experiment", "Dataset", 
                      "Accuracy", "Accuracy (SD)", 
                      "Number of P", "Number of P (SD)",
                      "Number of R", "Number of R (SD)", "Ideal Bin", 
                      "Iteration of Ideal Bin", "Iteration of Ideal Bin (SD)", "Ideal Threshold", 
                      "Threshold", "Threshold (SD)",
                      "Log-Rank Score", "Log-Rank Score (SD)", 
                      "Residual", "Residual (SD)", 
                      "Unadjusted HR", "Unadjusted HR (SD)", 
                      "Adjusted HR", "Adjusted HR (SD)", 
                      "Group Ratio", "Group Ratio (SD)",
                      "Runtime", "Runtime (SD)", "TC1 Present", 
                      "Bin Size", "Bin Size (SD)", 
                      "Birth Iteration", "Birth Iteration (SD)"]
    
    df_master = pd.DataFrame(columns=master_columns)
    master_results_list = [algorithm,experiment,dataset_name,
                           np.mean(accuracy),np.std(accuracy),
                           np.mean(num_P),np.std(num_P),
                           np.mean(num_R),np.std(num_R), ideal, 
                           np.mean(ideal_iter),np.std(ideal_iter), ideal_thresh,
                           np.mean(threshold),np.std(threshold), 
                           None if len(log_rank) == 0 else np.mean(log_rank), None if len(log_rank) == 0 else np.std(log_rank) ,
                           None if len(residuals) == 0 else np.mean(residuals), None if len(residuals) == 0 else np.std(residuals), 
                           None if len(unadj_HR) == 0 else np.mean(unadj_HR), None if len(unadj_HR) == 0 else np.std(unadj_HR),
                           None if len(adj_HR) == 0 else np.mean(adj_HR), None if len(adj_HR) == 0 else np.std(adj_HR), 
                           np.mean(group_balance),np.std(group_balance),
                           np.mean(runtime),np.std(runtime), tc,
                           np.mean(bin_size),np.std(bin_size), 
                           np.mean(birth_iteration),np.std(birth_iteration)]
    
    df_master.loc[len(df_master)] = master_results_list
    #Save master results as csv
    df_master.to_csv(outputPath+'/'+dataset_name+'_master_summary'+'.csv', index=False)
    
def ideal_iteration(ideal_count, feature_list, birth_iteration):
    if str(feature_list).count('P') == ideal_count and str(feature_list).count('R') == 0:
        return birth_iteration
    else:
        return None

if __name__=="__main__":
    sys.exit(main(sys.argv))
