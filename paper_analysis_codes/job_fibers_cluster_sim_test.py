import os
import sys
import argparse
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from skfibers.fibers import FIBERS

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--d', dest='dataset', help='name of data path (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputPath', help='', type=str, default = 'myOutputPath') #full path/filename
    parser.add_argument('--r', dest='random_seed', help='random seed', type=str, default='None')

    options=parser.parse_args(argv[1:])

    dataset = options.dataset
    outputPath = options.outputPath
    random_seed = int(options.random_seed)

    # Get dataset Name
    filename = os.path.basename(dataset)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(dataset)

    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    #Job Definition
    fibers = FIBERS(outcome_label="Duration", outcome_type="survival", iterations=50, pop_size=50, tournament_prop=0.2, crossover_prob=0.5, min_mutation_prob=0.1, max_mutation_prob=0.5, merge_prob=0.1, 
                    new_gen=1.0, elitism=0.1, diversity_pressure=0, min_bin_size=1, max_bin_size=None, max_bin_init_size=10, fitness_metric="log_rank", log_rank_weighting=None, censor_label="Censoring", 
                    group_strata_min=0.2, penalty=0.5, group_thresh=None, min_thresh=0, max_thresh=5, int_thresh=True, thresh_evolve_prob=0.5, manual_bin_init=None, covariates=None, pop_clean = 'group_strata',  
                    report=None, random_seed=random_seed, verbose=False)

    fibers = fibers.fit(data)
    bin_index = 0 #top bin
    summary = fibers.get_cox_prop_hazard(data, bin_index)

    #Save bin population as csv
    pop_df = fibers.get_pop()
    pop_df.to_csv(outputPath+'/'+'Pop_'+dataset_name+'_'+str(random_seed)+'.csv', index=False)

    #Pickle FIBERS trained object
    with open(outputPath+'/'+'Pop_'+dataset_name+'_'+str(random_seed)+'.pickle', 'wb') as f:
        pickle.dump(fibers, f)

if __name__=="__main__":
    sys.exit(main(sys.argv))
