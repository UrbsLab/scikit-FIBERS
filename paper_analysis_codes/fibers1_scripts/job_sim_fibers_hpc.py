import os
import sys
import argparse
import pickle
import pandas as pd
from lifelines import CoxPHFitter
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src_archive.skfibersv1.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

covariates = None #Manually included in script

def save_run_params(fibers, filename):
    with open(filename, 'w') as file:
        file.write(f"outcome_label: {fibers.duration_name}\n")
        file.write(f"iterations: {fibers.iterations}\n")
        file.write(f"pop_size: {fibers.set_number_of_bins}\n")
        file.write(f"crossover_prob: {fibers.crossover_probability}\n")
        file.write(f"mutation_prob: {fibers.mutation_probability}\n")
        file.write(f"elitism: {fibers.elitism_parameter}\n")
        file.write(f"min_features_per_group: {fibers.min_features_per_group}\n")
        file.write(f"max_number_of_groups_with_feature: {fibers.max_number_of_groups_with_feature}\n")
        file.write(f"censor_label: {fibers.label_name}\n")
        file.write(f"group_strata_min: {fibers.informative_cutoff}\n")
        file.write(f"random_seed: {fibers.random_seed}\n")

def prepare_data(df, outcome_label, censor_label, covariates):
    # Make list of feature names (i.e. columns that are not outcome, censor, or covariates)
    feature_names = list(df.columns)
    if covariates != None:
        exclude = covariates + [outcome_label,censor_label]
    else:
        exclude = [outcome_label,censor_label]
    feature_names = [item for item in feature_names if item not in exclude]

    # Remove invariant feature columns (data cleaning)
    cols_to_drop = []
    for col in feature_names:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True)
    feature_names = [item for item in feature_names if item not in cols_to_drop]
    print("Dropped "+str(len(cols_to_drop))+" invariant feature columns.")

    return df, feature_names

def cox_prop_hazard(bin_df, outcome_label, censor_label): #make bin variable beetween 0 and 1
    cph = CoxPHFitter()
    cph.fit(bin_df,outcome_label,event_col=censor_label, show_progress=False)
    return cph.summary

def get_cox_prop_hazard_unadjust(fibers,x, y=None, bin_index=0, use_bin_sums=False):
    if not fibers.hasTrained:
        raise Exception("FIBERS must be fit first")
    
    # PREPARE DATA ---------------------------------------
    df = fibers.check_x_y(x, y)
    df, feature_names = prepare_data(df, fibers.duration_name, fibers.label_name, covariates)

    # Sum instance values across features specified in the bin
    sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    feature_sums = df.loc[:,feature_names][fibers.bins[sorted_bin_list[bin_index]]].sum(axis=1)
    bin_df = pd.DataFrame({'Bin_'+str(bin_index):feature_sums})

    if not use_bin_sums:
        # Transform bin feature values according to respective bin threshold
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= fibers.bins[sorted_bin_list[bin_index]].threshold else 1)

    bin_df = pd.concat([bin_df,df.loc[:,fibers.duration_name],df.loc[:,fibers.label_name]],axis=1)
    summary = None
    try:
        summary = cox_prop_hazard(bin_df,fibers.duration_name,fibers.label_name)
        # fibers.set.bin_pop[bin_index].HR = summary['exp(coef)'].iloc[0]
        # fibers.set.bin_pop[bin_index].HR_CI = str(summary['exp(coef) lower 95%'].iloc[0])+'-'+str(summary['exp(coef) upper 95%'].iloc[0])
        # fibers.set.bin_pop[bin_index].HR_p_value = summary['p'].iloc[0]
    except:
        # fibers.set.bin_pop[bin_index].HR = 0
        # fibers.set.bin_pop[bin_index].HR_CI = None
        # fibers.set.bin_pop[bin_index].HR_p_value = None
        pass

    df = None
    return summary


def get_cox_prop_hazard_adjusted(fibers,x, y=None, bin_index=0, use_bin_sums=False):
    if not fibers.hasTrained:
        raise Exception("FIBERS must be fit first")

    # PREPARE DATA ---------------------------------------
    df = fibers.check_x_y(x, y)
    df, feature_names = prepare_data(df, fibers.duration_name, fibers.label_name, covariates)

    # Sum instance values across features specified in the bin
    
    sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    feature_sums = df.loc[:,feature_names][fibers.bins[sorted_bin_list[bin_index]]].sum(axis=1)
    bin_df = pd.DataFrame({'Bin_'+str(bin_index):feature_sums})

    if not use_bin_sums:
        # Transform bin feature values according to respective bin threshold
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= fibers.set.bin_pop[bin_index].group_threshold else 1)

    bin_df = pd.concat([bin_df,df.loc[:,fibers.outcome_label],df.loc[:,fibers.label_name]],axis=1)
    summary = None
    try:
        bin_df = pd.concat([bin_df,df.loc[:,covariates]],axis=1)
        summary = cox_prop_hazard(bin_df,fibers.outcome_label,fibers.label_name)
        # fibers.set.bin_pop[bin_index].adj_HR = summary['exp(coef)'].iloc[0]
        # fibers.set.bin_pop[bin_index].adj_HR_CI = str(summary['exp(coef) lower 95%'].iloc[0])+'-'+str(summary['exp(coef) upper 95%'].iloc[0])
        # fibers.set.bin_pop[bin_index].adj_HR_p_value = summary['p'].iloc[0]
    except:
        # fibers.set.bin_pop[bin_index].adj_HR = 0
        # fibers.set.bin_pop[bin_index].adj_HR_CI = None
        # fibers.set.bin_pop[bin_index].adj_HR_p_value = None
        pass

    df = None
    return summary
    

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datapath', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='Duration')  
    parser.add_argument('--i', dest='iterations', help='iterations', type=int, default=100)
    parser.add_argument('--ps', dest='pop_size', help='population size', type=int, default=50)
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--cp', dest='crossover_prob', help='crossover probability', type=float, default=0.5)
    parser.add_argument('--mup', dest='mutation_prob', help='mutation probability', type=float, default=0.5)
    parser.add_argument('--e', dest='elitism', help='elite proportion of population protected from deletion', type=float, default=0.1)
    parser.add_argument('--bi', dest='min_features_per_group', help='mininum features in a bin', type=int, default=2)
    parser.add_argument('--ba', dest='max_number_of_groups_with_feature', help='maximum number of bin with said features', type=int, default=2)
    parser.add_argument('--c', dest='censor_label', help='censor column label', type=str, default='Censoring')
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.2)
    parser.add_argument('--r', dest='random_seed', help='random seed', type=int, default='None')

    options=parser.parse_args(argv[1:])

    datapath= options.datapath
    outputpath = options.outputpath
    if options.manual_bin_init == 'None':
        manual_bin_init = None
        assert(manual_bin_init is None)
    else:
        # manual_bin_init = pd.read_csv(manual_bin_init,low_memory=False)
        raise NotImplementedError

    outcome_label = options.outcome_label
    iterations = options.iterations
    pop_size = options.pop_size
    crossover_prob = options.crossover_prob
    mutation_prob = options.mutation_prob 
    elitism = options.elitism
    min_features_per_group = options.min_features_per_group
    max_number_of_groups_with_feature = int(options.max_number_of_groups_with_feature)
    censor_label = options.censor_label
    group_strata_min = options.group_strata_min
    random_seed = options.random_seed

    # Get Dataset Name
    filename = os.path.basename(datapath)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(datapath)

    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    #Job Definition
    # fibers = FIBERS(outcome_label=outcome_label, outcome_type=outcome_type, iterations=iterations, pop_size=pop_size, tournament_prop=tournament_prop, 
    #                 crossover_prob=crossover_prob, min_mutation_prob=min_mutation_prob, max_mutation_prob=max_mutation_prob, merge_prob=merge_prob, 
    #                 new_gen=new_gen, elitism=elitism, diversity_pressure=diversity_pressure, min_bin_size=min_bin_size, max_bin_size=max_bin_size,
    #                 max_bin_init_size=max_bin_init_size, fitness_metric=fitness_metric, log_rank_weighting=log_rank_weighting, censor_label=censor_label, 
    #                 group_strata_min=group_strata_min, penalty=penalty, group_thresh=group_thresh, min_thresh=min_thresh, max_thresh=max_thresh,
    #                 int_thresh=True, thresh_evolve_prob=thresh_evolve_prob, manual_bin_init=manual_bin_init, covariates=covariates, pop_clean=pop_clean,  
    #                 report=None, random_seed=random_seed, verbose=False)
    fibers = FIBERS(label_name=censor_label, duration_name=outcome_label,
                    given_starting_point=False, amino_acid_start_point=None, amino_acid_bins_start_point=None,
                    iterations=iterations, set_number_of_bins=pop_size, 
                    min_features_per_group=min_features_per_group, max_number_of_groups_with_feature=max_number_of_groups_with_feature,
                    informative_cutoff=group_strata_min, crossover_probability=crossover_prob, mutation_probability=mutation_prob, 
                    elitism_parameter=elitism,
                    random_seed=random_seed)

    fibers = fibers.fit(data)
    bin_index = 0 #top bin
    try:
        summary = get_cox_prop_hazard_unadjust(fibers, data, bin_index)
        summary.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_coxph_unadj_bin_'+str(bin_index)+'.csv', index=True)
        if covariates != None:
            summary = get_cox_prop_hazard_adjusted(fibers, data, bin_index)
            summary.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_coxph_adj_bin_'+str(bin_index)+'.csv', index=True)
    except Exception as e:
        print("Exception")
        print(e)
    #Save bin population as csv
    # pop_df = fibers.get_pop()
    # pop_df.to_csv(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_pop'+'.csv', index=False)

    #Pickle FIBERS trained object
    with open(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_fibers.pickle', 'wb') as f:
        pickle.dump(fibers, f)
    
    save_run_params(fibers, outputpath+'/'+dataset_name+'_'+str(random_seed)+'_run_parameters.txt')



if __name__=="__main__":
    sys.exit(main(sys.argv))
