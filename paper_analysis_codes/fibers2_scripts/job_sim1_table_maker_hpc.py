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

    significance_metrics = ['Accuracy','Number of P','Number of R','Ideal Iteration','Log-Rank Score','Unadjusted HR','Group Ratio','Runtime']
    count_metrics = ['Ideal Bin','Ideal Threshold']

    #Table - FIBERS2 Base - Mutation rate compare
    table_name = 'MutationRateComparison'
    fixed_element = 'BasePC_i_10000_tf_100_p_10_t_0_n_0.0_c_0.2_nc_False'
    variable_element = ['Fibers2.0_sim_mutation_rate_0_1','Fibers2.0_sim_mutation_rate_0_2','Fibers2.0_sim_mutation_rate_0_3',
                             'Fibers2.0_sim_mutation_rate_0_4','Fibers2.0_sim_mutation_rate_0_5']
    baseline = 'Fibers2.0_sim_mutation_rate_0_1'
    var_element_is_experiment = True
    stat_list_columns = ['Dataset','Accuracy','Number of P','Number of R','Ideal Bin','Iteration of Ideal Bin',
                         'Ideal Threshold','Threshold','Log-Rank Score','Residual','Unadjusted HR','Group Ratio','Runtime']

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
        formated_df = format_data(df_master,stat_list_columns)
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
                is_sig = wilcoxon_sig(base_col,compare_col,p_val)
                if is_sig: # indicate significance within dataframe stat_list
                    # Find appropriate metric
                    dataframe_stat_list[i][metric] = str(dataframe_stat_list[i][metric])+'*'

    # combine experiment results into a single dataframe
    combined_df = pd.concat(dataframe_stat_list, ignore_index=True)
    combined_df.to_csv(outputpath+'/'+str(table_name)+'_Table.csv', index=False)





    #df = pd.DataFrame(experiment)
    #df.to_csv(outputpath+'/MutationRate_Table.csv', index=True)

def wilcoxon_sig(col1,col2,p_val):
    statistic, p_value = wilcoxon(col1, col2)
    print('Comparison: '+str(statistic)+ ' '+ str(p_value))
    if p_value <= p_val:
        return True

def format_data(df,stat_list_columns):
    experiment = []
    experiment.append(df.loc[0,'Dataset'])
    experiment.append(str(round(df.loc[0,'Accuracy'],3))+' ('+str(round(df.loc[0,'Accuracy (SD)'],3))+')') #Accuracy
    experiment.append(str(round(df.loc[0,'Number of P'],3))+' ('+str(round(df.loc[0,'Number of P (SD)'],3))+')') #Num Pred
    experiment.append(str(round(df.loc[0,'Number of R'],3))+' ('+str(round(df.loc[0,'Number of R (SD)'],3))+')') #Num Rand
    experiment.append("'"+str(df.loc[0,'Ideal Bin'])+'/30') #Top bin
    experiment.append(str(round(df.loc[0,'Iteration of Ideal Bin'],2))+' ('+str(round(df.loc[0,'Iteration of Ideal Bin (SD)'],2))+')') #Ideal Iter
    experiment.append("'"+str(df.loc[0,'Ideal Threshold'])+'/30') #True Thresh
    experiment.append(str(round(df.loc[0,'Threshold'],1))+' ('+str(round(df.loc[0,'Threshold (SD)'],1))+')') #log rank
    experiment.append(str(round(df.loc[0,'Log-Rank Score'],1))+' ('+str(round(df.loc[0,'Log-Rank Score (SD)'],1))+')') #log rank
    experiment.append(str(round(df.loc[0,'Residual'],2))+' ('+str(round(df.loc[0,'Residual (SD)'],2))+')') #residual
    experiment.append(str(round(df.loc[0,'Unadjusted HR'],2))+' ('+str(round(df.loc[0,'Unadjusted HR (SD)'],2))+')') #adj HR
    experiment.append(str(round(df.loc[0,'Group Ratio'],2))+' ('+str(round(df.loc[0,'Group Ratio (SD)'],2))+')') #Group ratio
    experiment.append(str(round(df.loc[0,'Runtime'],2))+' ('+str(round(df.loc[0,'Runtime (SD)'],2))+')') #runtime

    new_df = pd.DataFrame([experiment], columns=stat_list_columns)
    return new_df


if __name__=="__main__":
    sys.exit(main(sys.argv))
