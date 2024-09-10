import os
import sys
import argparse
import pickle
import pandas as pd
from lifelines import CoxPHFitter
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src_archive.skfibersv1.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

covariates = [
              'shared', 'DCD', 'DON_AGE', 'donage_slope_ge18', 'dcadcodanox', 'dcadcodcva', 'dcadcodcnst', 'dcadcodoth', 'don_cmv_negative',
              'don_htn_0c', 'ln_don_wgt_kg_0c', 'ln_don_wgt_kg_0c_s55', 'don_ecd', 'age_ecd', 'yearslice', 'REC_AGE_AT_TX',
              'rec_age_spline_35', 'rec_age_spline_50', 'rec_age_spline_65', 'diab_noted', 'age_diab', 'dm_can_age_spline_50',
              'can_dgn_htn_ndm', 'can_dgn_pk_ndm', 'can_dgn_gd_ndm', 'rec_prev_ki_tx', 'rec_prev_ki_tx_dm', 'rbmi_0c', 'rbmi_miss',
              'rbmi_gt_20', 'rbmi_DM', 'rbmi_gt_20_DM', 'ln_c_hd_m', 'ln_c_hd_0c', 'ln_c_hd_m_ptx', 'PKPRA_MS', 'PKPRA_1080',
              'PKPRA_GE80', 'hispanic', 'CAN_RACE_BLACK', 'CAN_RACE_asian', 'CAN_RACE_WHITE', 'Agmm0']

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
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= 0 else 1)

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
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= 0 else 1)

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
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'None')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0)

    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='Duration')  
    parser.add_argument('--i', dest='iterations', help='iterations', type=int, default=100)
    parser.add_argument('--ps', dest='pop_size', help='population size', type=int, default=50)
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
    
    loci_list = options.loci_list.split(',')
    if options.cov_list == 'None':
        cov_list = None
    else:
        cov_list = options.cov_list.split(',')
    rare_filter = options.rare_filter


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
    
    #Create Final Covariate List
    if cov_list != None:
        for covariate in cov_list:
            cov_sub_list = cov_typ_dict[covariate]
            for each in cov_sub_list:
                covariates.append(each) #add selected Ag covariate to primary covariates
    print(covariates) #temporary

    # Get Dataset Name
    filename = os.path.basename(datapath)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(datapath)

    #Identify MM features to include as independent features
    MM_feature_list = []
    for locus in loci_list: #each specified locus to evaluate as independent features
        for j in range(locus_range_dict[locus][0],locus_range_dict[locus][1]+1):
            MM_feature_list.append('MM_'+str(locus)+'_'+str(j))

    features = MM_feature_list + [outcome_label] + [censor_label]
    print(features) #temporary
    data = data[features]

    #Missing data values check
    missing_sum = data.isna().sum().sum()
    if missing_sum > 0:
        print("Sum of data missing values:", missing_sum)

    #Data Cleaning
    if rare_filter > 0.0: #filter out rare features and invariant features
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        print(percentages)
        columns_to_remove = percentages[percentages < rare_filter].index.tolist()
        data = data.drop(columns=columns_to_remove)
    else: #filter out invariant features only
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        print(percentages)
        columns_to_remove = percentages[percentages == 0.0].index.tolist()
        data = data.drop(columns=columns_to_remove)

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
