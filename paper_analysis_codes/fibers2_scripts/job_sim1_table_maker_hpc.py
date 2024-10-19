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

    #Get names of all experiment folders
    experiment_folder_names = [name for name in os.listdir(writepath) if os.path.isdir(os.path.join(writepath, name))]
    print(experiment_folder_names)
    #Get names of all dataset folders (used within each experiment)
    dataset_folder_names = [name for name in os.listdir(writepath) if os.path.isdir(os.path.join(writepath+experiment_folder_names[0], name))]
    print(dataset_folder_names)

    significance_metrics = ['Accuracy','Number of P','Number of R','Ideal Iteration','Log-Rank Score','Unadjusted HR','Group Ratio','Runtime']
    count_metrics = ['Ideal Bin','Ideal Threshold']

    #Table - FIBERS2 Base - Mutation rate compare
    experiment_folder_names = ['Fibers2.0_sim_mutation_rate_0_1','Fibers2.0_sim_mutation_rate_0_2','Fibers2.0_sim_mutation_rate_0_3',
                               'Fibers2.0_sim_mutation_rate_0_4','Fibers2.0_sim_mutation_rate_0_5']
    dataset_folder_names = ['BasePC_i_10000_tf_100_p_10_t_0_n_0.0_c_0.2_nc_False']

    # Gather baseline stats
    baseline_experiment = ['Fibers2.0_sim_mutation_rate_0_1']

    master_summary = writepath+baseline_experiment+'/'+str(dataset_folder_names[0])+'/summary/'+str(dataset_folder_names[0])+'_master_summary.csv'
    summary = writepath+baseline_experiment+'/'+str(dataset_folder_names[0])+'/summary/'+str(dataset_folder_names[0])+'_summary.csv'

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(master_summary)
    experiment = format_data(df)

    df = pd.DataFrame(experiment)
    T_df = df.T
    T_df.to_csv(outputpath+'/MutationRate_Table.csv', index=False)

    significance_metrics = ['Accuracy','Number of P','Number of R','Ideal Iteration','Threshold','Log-Rank Score','Unadjusted HR','Group Ratio','Runtime']
    count_metrics = ['Ideal Bin','Ideal Threshold']

def format_data(df):
    experiment = []
    experiment.append(df.iloc[0,'Experiment']+'_'+df.iloc[0,'Dataset'])
    experiment.append(str(round(df.iloc[0,'Accuracy'],3))+' ('+str(round(df.iloc[0,'Accuracy (SD)'],3))+')') #Accuracy
    experiment.append(str(round(df.iloc[0,'Number of P'],3))+' ('+str(round(df.iloc[0,'Number of P (SD)'],3))+')') #Num Pred
    experiment.append(str(round(df.iloc[0,'Number of R'],3))+' ('+str(round(df.iloc[0,'Number of R (SD)'],3))+')') #Num Rand
    experiment.append("'"+str(df.iloc[0,'Ideal Bin'])+'/30') #Top bin
    experiment.append(str(round(df.iloc[0,'Iteration of Ideal Bin'],2))+' ('+str(round(df.iloc[0,'Iteration of Ideal Bin (SD)'],2))+')') #Ideal Iter
    experiment.append("'"+str(df.iloc[0,'Ideal Threshold'])+'/30') #True Thresh
    experiment.append(str(round(df.iloc[0,'Threshold'],1))+' ('+str(round(df.iloc[0,'Threshold (SD)'],1))+')') #log rank
    experiment.append(str(round(df.iloc[0,'Log-Rank Score'],1))+' ('+str(round(df.iloc[0,'Log-Rank Score (SD)'],1))+')') #log rank
    experiment.append(str(round(df.iloc[0,'Residual'],2))+' ('+str(round(df.iloc[0,'Residual (SD)'],2))+')') #residual
    experiment.append(str(round(df.iloc[0,'Unadjusted HR'],2))+' ('+str(round(df.iloc[0,'Unadjusted HR (SD)'],2))+')') #adj HR
    experiment.append(str(round(df.iloc[0,'Group Ratio'],2))+' ('+str(round(df.iloc[0,'Group Ratio (SD)'],2))+')') #Group ratio
    experiment.append(str(round(df.iloc[0,'Runtime'],2))+' ('+str(round(df.iloc[0,'Runtime (SD)'],2))+')') #runtime
    return experiment


if __name__=="__main__":
    sys.exit(main(sys.argv))
