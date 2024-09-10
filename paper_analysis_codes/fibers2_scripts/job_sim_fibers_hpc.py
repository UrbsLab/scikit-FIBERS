import os
import sys
import argparse
import pickle
import pandas as pd
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datapath', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='Duration')  
    parser.add_argument('--ot', dest='outcome_type', help='outcome type', type=str, default='survival')
    parser.add_argument('--i', dest='iterations', help='iterations', type=int, default=100)
    parser.add_argument('--ps', dest='pop_size', help='population size', type=int, default=50)
    parser.add_argument('--tp', dest='tournament_prop', help='trournament probability', type=float, default=0.2)
    parser.add_argument('--cp', dest='crossover_prob', help='crossover probability', type=float, default=0.5)
    parser.add_argument('--mi', dest='min_mutation_prob', help='minimum mutation probability', type=float, default=0.1)
    parser.add_argument('--ma', dest='max_mutation_prob', help='maximum mutation probability', type=float, default=0.5)
    parser.add_argument('--mp', dest='merge_prob', help='merge probability', type=float, default=0.1)
    parser.add_argument('--ng', dest='new_gen', help='proportion of max population used to deterimine offspring population size', type=float, default=1.0)
    parser.add_argument('--e', dest='elitism', help='elite proportion of population protected from deletion', type=float, default=0.1)
    parser.add_argument('--dp', dest='diversity_pressure', help='diversity pressure (K in k-means)', type=int, default=0)
    parser.add_argument('--bi', dest='min_bin_size', help='minimum bin size', type=int, default=1)
    parser.add_argument('--ba', dest='max_bin_size', help='maximum bin size', type=str, default='None')
    parser.add_argument('--ib', dest='max_bin_init_size', help='maximum bin intitilize size', type=int, default=10)
    parser.add_argument('--f', dest='fitness_metric', help='fitness metric', type=str, default='log_rank')
    parser.add_argument('--we', dest='log_rank_weighting', help='log-rank test weighting', type=str, default='None')
    parser.add_argument('--c', dest='censor_label', help='censor column label', type=str, default='Censoring')
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.2)
    parser.add_argument('--p', dest='penalty', help='group strata min penalty', type=float, default=0.5)
    parser.add_argument('--t', dest='group_thresh', help='group threshold', type=str, default=0)
    parser.add_argument('--it', dest='min_thresh', help='minimum threshold', type=int, default=0)
    parser.add_argument('--at', dest='max_thresh', help='maximum threshold', type=int, default=5)
    #int_thresh
    parser.add_argument('--te', dest='thresh_evolve_prob', help='threshold evolution probability', type=float, default=0.5)
    parser.add_argument('--cl', dest='pop_clean', help='clean population', type=str, default='None')
    parser.add_argument('--cov', dest='covariates_used', help='covariates used', type=str, default='None')
    parser.add_argument('--r', dest='random_seed', help='random seed', type=int, default='None')

    options=parser.parse_args(argv[1:])

    datapath= options.datapath
    outputpath = options.outputpath
    if options.manual_bin_init == 'None':
        manual_bin_init = None
    else:
        manual_bin_init = pd.read_csv(manual_bin_init,low_memory=False)

    outcome_label = options.outcome_label
    outcome_type = options.outcome_type
    iterations = options.iterations
    pop_size = options.pop_size
    tournament_prop = options.tournament_prop
    crossover_prob = options.crossover_prob
    min_mutation_prob = options.min_mutation_prob 
    max_mutation_prob = options.max_mutation_prob
    merge_prob = options.merge_prob
    new_gen = options.new_gen
    elitism = options.elitism
    diversity_pressure = options.diversity_pressure
    min_bin_size = options.min_bin_size
    if options.max_bin_size == 'None':
        max_bin_size = None
    else:
        max_bin_size = int(options.max_bin_size)
    max_bin_init_size = options.max_bin_init_size
    fitness_metric = options.fitness_metric
    if options.log_rank_weighting == 'None':
        log_rank_weighting = None
    else:
        log_rank_weighting = str(options.log_rank_weighting)
    censor_label = options.censor_label
    group_strata_min = options.group_strata_min
    penalty = options.penalty
    if options.group_thresh == 'None':
        group_thresh = None
    else:
        group_thresh = int(options.group_thresh)
    min_thresh = options.min_thresh 
    max_thresh = options.max_thresh 
    #int_thresh = options.int_thresh
    thresh_evolve_prob = options.thresh_evolve_prob
    if options.pop_clean == 'None':
        pop_clean = None
    else:
        pop_clean = str(options.pop_clean)
    covariates_used = options.covariates_used
    if covariates_used == 'Adv':
        covariates = ['AFRICAN-AMERICAN','ASIAN','HISPANIC','WHITE','OTHER','FDFR','FDMR','MDFR','MDMR']
    elif covariates_used == 'Simple':
        covariates = ['C_1', 'C_2']
    else:
        covariates = None
    random_seed = options.random_seed

    # Get Dataset Name
    filename = os.path.basename(datapath)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(datapath)

    if covariates == None:
        true_risk_group = data[['TrueRiskGroup']]
        data = data.drop('TrueRiskGroup', axis=1)

    #Job Definition
    fibers = FIBERS(outcome_label=outcome_label, outcome_type=outcome_type, iterations=iterations, pop_size=pop_size, tournament_prop=tournament_prop, 
                    crossover_prob=crossover_prob, min_mutation_prob=min_mutation_prob, max_mutation_prob=max_mutation_prob, merge_prob=merge_prob, 
                    new_gen=new_gen, elitism=elitism, diversity_pressure=diversity_pressure, min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                    max_bin_init_size=max_bin_init_size, fitness_metric=fitness_metric, log_rank_weighting=log_rank_weighting, censor_label=censor_label, 
                    group_strata_min=group_strata_min, penalty=penalty, group_thresh=group_thresh, min_thresh=min_thresh, max_thresh=max_thresh,
                    int_thresh=True, thresh_evolve_prob=thresh_evolve_prob, manual_bin_init=manual_bin_init, covariates=covariates, pop_clean=pop_clean,  
                    report=None, random_seed=random_seed, verbose=False)

    fibers = fibers.fit(data)
    bin_index = 0 #top bin
    try:
        summary = fibers.get_cox_prop_hazard_unadjust(data, bin_index)
        summary.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_coxph_unadj_bin_'+str(bin_index)+'.csv', index=True)
        if covariates != None:
            summary = fibers.get_cox_prop_hazard_adjusted(data, bin_index)
            summary.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_coxph_adj_bin_'+str(bin_index)+'.csv', index=True)
    except:
        pass
    #Save bin population as csv
    pop_df = fibers.get_pop()
    pop_df.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_pop'+'.csv', index=False)

    #Pickle FIBERS trained object
    with open(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_fibers.pickle', 'wb') as f:
        pickle.dump(fibers, f)
    
    fibers.save_run_params(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_run_parameters.txt')

if __name__=="__main__":
    sys.exit(main(sys.argv))
