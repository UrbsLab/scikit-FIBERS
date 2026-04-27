import os
import sys
import argparse
import pickle
import pandas as pd

script_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_path, '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Fallback shared cluster checkout.
if '/project/kamoun_shared/code_shared/scikit-FIBERS-protective' not in sys.path:
    sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS-protective')
# sys.path.append('/project/kamoun_shared/amy_fibers_project/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'None')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0)
    parser.add_argument('--cv', dest='cv', help='current cv split', type=int, default='None')

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
    parser.add_argument('--de', dest='desired_bin_effect', help='desired bin effect mode', type=str, default='default')
    parser.add_argument('--cl', dest='pop_clean', help='clean population', type=str, default='None')
    parser.add_argument('--r', dest='random_seed', help='random seed', type=int, default='None')

    options=parser.parse_args(argv[1:])

    datafolder= options.datafolder
    outputpath = options.outputpath
    if options.manual_bin_init == 'None':
        manual_bin_init = None
    else:
        manual_bin_init = pd.read_csv(options.manual_bin_init,low_memory=False)
    loci_list = options.loci_list.split(',')
    if options.cov_list == 'None':
        cov_list = None
    else:
        cov_list = options.cov_list.split(',')
    rare_filter = options.rare_filter
    cv = options.cv

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
    desired_bin_effect = options.desired_bin_effect
    if desired_bin_effect == "highrisk":
        desired_bin_effect = "high_risk"
    covariates = None #Manually included in script
    if options.pop_clean == 'None':
        pop_clean = None
    else:
        pop_clean = str(options.pop_clean)
    random_seed = options.random_seed

    if desired_bin_effect not in ["default", "protective", "high_risk", "permissive"]:
        raise Exception("'desired_bin_effect' must be one of: 'default', 'protective', 'high_risk', 'permissive'")

    #Hard Coded Covariate Information
    #If there is a colinearity issue with calculating residuals, Keith indicated that we can remove dcadcodoth and/or PKPRA_MS from covariate list
    covariates = [
              'shared', 'DCD', 'DON_AGE', 'donage_slope_ge18', 'dcadcodanox', 'dcadcodcva', 'dcadcodcnst', 'dcadcodoth', 'don_cmv_negative',
              'don_htn_0c', 'ln_don_wgt_kg_0c', 'ln_don_wgt_kg_0c_s55', 'don_ecd', 'age_ecd', 'yearslice', 'REC_AGE_AT_TX',
              'rec_age_spline_35', 'rec_age_spline_50', 'rec_age_spline_65', 'diab_noted', 'age_diab', 'dm_can_age_spline_50',
              'can_dgn_htn_ndm', 'can_dgn_pk_ndm', 'can_dgn_gd_ndm', 'rec_prev_ki_tx', 'rec_prev_ki_tx_dm', 'rbmi_0c', 'rbmi_miss',
              'rbmi_gt_20', 'rbmi_DM', 'rbmi_gt_20_DM', 'ln_c_hd_m', 'ln_c_hd_0c', 'ln_c_hd_m_ptx', 'PKPRA_MS', 'PKPRA_1080',
              'PKPRA_GE80', 'hispanic', 'CAN_RACE_BLACK', 'CAN_RACE_asian', 'CAN_RACE_WHITE', 'Agmm0']
    
    #DRB345 has redundant covariates with DRB1, so both should not be specified together
    #Also, Malek indicated we will not corret for DPA1 or DPB1 for the time being.
    cov_typ_dict = {
        'A': ['AgmmA0', 'AgmmA1'],
        'B': ['AgmmB0', 'AgmmB1'],
        'C': ['Agmmc0', 'Agmmc1'],
        'DRB1':  ['AgmmDR0', 'AgmmDR1'],
        'DRB345':  ['AgmmDR0', 'AgmmDR1'],
        'DQA1':  ['Agmmdqa10', 'Agmmdqa11'],
        'DQB1':  ['Agmmdqb10', 'Agmmdqb11'],
        'DPA1':  ['Agmmdpa10', 'Agmmdpa11'],
        'DPB1':  ['Agmmdpb10', 'Agmmdpb11']} 
    
    #hard coded specific AA-MM positions to include in these analyses
    locus_range_dict = {
        'A': [1,182],
        'B': [1,182],
        'C': [1,182],
        'DRB1': [6,94],
        'DRB345': [6,94],
        'DQA1':  [6,94],
        'DQB1':  [6,95],
        'DPA1':  [6,94],
        'DPB1':  [6,94]}
    
    final_covariates = covariates[:]
    Ag_covariates = []
    #Create Final Covariate List
    if cov_list != None:
        for covariate in cov_list:
            cov_sub_list = cov_typ_dict[covariate]
            for each in cov_sub_list:
                final_covariates.append(each) #add selected Ag covariate to primary covariates
                Ag_covariates.append(each)
    print(final_covariates) #temporary

    # Get Dataset Name
    files = [f for f in os.listdir(datafolder) if os.path.isfile(os.path.join(datafolder, f))]
    filename = os.path.splitext(files[0])[0]
    nameparts = filename.split('_')
    filename = '_'.join(nameparts[:3])

    # Get Dataset Name
    data_full_train = datafolder+'/'+filename+ '_'+str(cv)+'_Train.csv'

    #Load/Process Dataset
    train_data = pd.read_csv(data_full_train)

    #Identify MM features to include as independent features
    MM_feature_list = []
    for locus in loci_list: #each specified locus to evaluate as independent features
        for j in range(locus_range_dict[locus][0],locus_range_dict[locus][1]+1):
            MM_feature_list.append('MM_'+str(locus)+'_'+str(j))

    features = MM_feature_list + final_covariates + [outcome_label] + [censor_label]
    print(features) #temporary
    train_data = train_data[features]

    #Missing data values check
    missing_sum = train_data.isna().sum().sum()
    if missing_sum > 0:
        print("Sum of data missing values:", missing_sum)

    #Data Cleaning
    if rare_filter > 0.0: #filter out rare features and invariant features
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = train_data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        print(percentages)
        columns_to_remove = percentages[percentages < rare_filter].index.tolist()
        train_data = train_data.drop(columns=columns_to_remove)
    else: #filter out invariant features only
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = train_data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        print(percentages)
        columns_to_remove = percentages[percentages == 0.0].index.tolist()
        train_data = train_data.drop(columns=columns_to_remove)

    #Report filtering
    count_list = []
    total_count = 0
    for locus in loci_list:
        count = sum(['MM_'+str(locus) in col for col in train_data.columns])
        total_count += count
        count_list.append(str(locus)+":"+str(count))

    with open(outputpath+'/'+str(cv)+'_post_filter_counts.txt', 'w') as file:
        for item in count_list:
            file.write(f"{item}\n")
        file.write('Total:'+str(total_count))

    #Job Definition
    fibers = FIBERS(outcome_label=outcome_label, outcome_type=outcome_type, iterations=iterations, pop_size=pop_size, tournament_prop=tournament_prop, 
                    crossover_prob=crossover_prob, min_mutation_prob=min_mutation_prob, max_mutation_prob=max_mutation_prob, merge_prob=merge_prob, 
                    new_gen=new_gen, elitism=elitism, diversity_pressure=diversity_pressure, min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                    max_bin_init_size=max_bin_init_size, fitness_metric=fitness_metric, log_rank_weighting=log_rank_weighting, censor_label=censor_label, 
                    group_strata_min=group_strata_min, penalty=penalty, group_thresh=group_thresh, min_thresh=min_thresh, max_thresh=max_thresh,
                    int_thresh=True, thresh_evolve_prob=thresh_evolve_prob, manual_bin_init=manual_bin_init, covariates=final_covariates, pop_clean=pop_clean,  
                    report=None, random_seed=random_seed, verbose=False, desired_bin_effect=desired_bin_effect)

    fibers = fibers.fit(train_data)
    bin_index = 0 #top bin
    y = None
    use_bin_sums = False
    show_progress = True

    # Save core run artifacts even if no bins survive cleanup/filtering.
    pop_df = fibers.get_pop()
    pop_df.to_csv(outputpath+'/'+str(cv)+'_pop'+'.csv', index=False)

    with open(outputpath+'/'+str(cv)+'_fibers.pickle', 'wb') as f:
        pickle.dump(fibers, f)
    
    fibers.save_run_params(outputpath+'/'+str(cv)+'_run_parameters.txt')

    if len(fibers.set.bin_pop) == 0:
        with open(outputpath+'/'+str(cv)+'_no_valid_bins.txt', 'w') as file:
            file.write('No bins remained after training/cleanup.\n')
            file.write('desired_bin_effect: '+str(desired_bin_effect)+'\n')
            file.write('pop_clean: '+str(pop_clean)+'\n')
            file.write('group_strata_min: '+str(group_strata_min)+'\n')
        return

    summary = fibers.get_cox_prop_hazard_unadjust(train_data, y, bin_index, use_bin_sums, show_progress)
    summary.to_csv(outputpath+'/'+str(cv)+'_coxph_unadj_bin_train_'+str(bin_index)+'.csv', index=True)

    #Kaplan Meir Plot
    fibers.get_kaplan_meir(train_data,bin_index,save=True,show=False, output_folder=outputpath,data_name=str(cv)+'_train')

    if final_covariates != None:
        summary = fibers.get_cox_prop_hazard_adjusted(train_data, y, bin_index, use_bin_sums, show_progress)
        summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_train_'+str(bin_index)+'.csv', index=True)
        if final_covariates != covariates:
            train_data = train_data.drop(columns=Ag_covariates)
            summary = fibers.get_cox_prop_hazard_adjusted(train_data, y, bin_index, use_bin_sums, show_progress, covariates)
            summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_train_'+str(bin_index)+'_NoAg.csv', index=True)



    # Get Dataset Name
    data_full_test = datafolder+'/'+filename+ '_'+str(cv)+'_Test.csv'
    #Load/Process Dataset
    original_test_data = pd.read_csv(data_full_test)
    features = MM_feature_list + final_covariates + [outcome_label] + [censor_label]
    test_data = original_test_data[features]

    summary = fibers.get_cox_prop_hazard_unadjust(test_data, y, bin_index, use_bin_sums, show_progress)
    summary.to_csv(outputpath+'/'+str(cv)+'_coxph_unadj_bin_test_'+str(bin_index)+'.csv', index=True)

    #Kaplan Meir Plot
    fibers.get_kaplan_meir(test_data,bin_index,save=True,show=False, output_folder=outputpath,data_name=str(cv)+'_test')

    if final_covariates != None:
        try:
            summary = fibers.get_cox_prop_hazard_adjusted(test_data, y, bin_index, use_bin_sums, show_progress)
            summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_test_'+str(bin_index)+'.csv', index=True)
        except:
            test_data_new = test_data.drop(columns=['PKPRA_MS'])
            temp_covariates = covariates[:]
            temp_covariates.remove('PKPRA_MS')
            summary = fibers.get_cox_prop_hazard_adjusted(test_data_new, y, bin_index, use_bin_sums, show_progress,temp_covariates)
            summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_test_'+str(bin_index)+'.csv', index=True)

        if final_covariates != covariates:
            test_data = test_data.drop(columns=Ag_covariates)
            try:
                summary = fibers.get_cox_prop_hazard_adjusted(test_data, y, bin_index, use_bin_sums, show_progress, covariates)
                summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_test_'+str(bin_index)+'_NoAg.csv', index=True)
            except:
                test_data_new = test_data.drop(columns=['PKPRA_MS'])
                temp_covariates = covariates[:]
                temp_covariates.remove('PKPRA_MS')
                summary = fibers.get_cox_prop_hazard_adjusted(test_data_new, y, bin_index, use_bin_sums, show_progress, temp_covariates)
                summary.to_csv(outputpath+'/'+str(cv)+'_coxph_adj_bin_test_'+str(bin_index)+'_NoAg.csv', index=True)

if __name__=="__main__":
    sys.exit(main(sys.argv))
