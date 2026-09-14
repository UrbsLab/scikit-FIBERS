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
if os.path.isdir('/project/kamoun_shared/code_shared/scikit-FIBERS-protective') and '/project/kamoun_shared/code_shared/scikit-FIBERS-protective' not in sys.path:
    sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS-protective')
# sys.path.append('/project/kamoun_shared/amy_fibers_project/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN


def save_summary(summary, csv_path, label):
    if summary is not None:
        summary.to_csv(csv_path, index=True)
        return True

    failure_path = os.path.splitext(csv_path)[0] + "_failed.txt"
    with open(failure_path, "w") as handle:
        handle.write(f"{label} could not be estimated by CoxPHFitter.\n")
    print(f"Warning: {label} was not estimable; wrote {failure_path}")
    return False


def save_kaplan_meier(fibers, data, bin_index, outputpath, data_name):
    try:
        fibers.get_kaplan_meir(
            data,
            bin_index,
            save=True,
            show=False,
            output_folder=outputpath,
            data_name=data_name,
        )
    except Exception as error:
        failure_path = os.path.join(outputpath, f"{data_name}_kaplan_meier_failed.txt")
        with open(failure_path, "w") as handle:
            handle.write(f"Kaplan-Meier plot failed: {error}\n")
        print(f"Warning: Kaplan-Meier plot failed; wrote {failure_path}")

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='folder containing CV train/test CSV files', type=str, required=True)
    parser.add_argument('--o', dest='outputpath', help='directory path for run outputs', type=str, required=True)
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'None')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0.1)
    parser.add_argument('--cv', dest='cv', help='current cv split', type=int, required=True)

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
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.1)
    parser.add_argument('--p', dest='penalty', help='group strata min penalty', type=float, default=0.5)
    parser.add_argument('--t', dest='group_thresh', help='group threshold', type=str, default=0)
    parser.add_argument('--it', dest='min_thresh', help='minimum threshold', type=int, default=0)
    parser.add_argument('--at', dest='max_thresh', help='maximum threshold', type=int, default=5)
    #int_thresh
    parser.add_argument('--te', dest='thresh_evolve_prob', help='threshold evolution probability', type=float, default=0.5)
    parser.add_argument('--de', dest='desired_bin_effect', help='desired bin effect mode', type=str, default='default')
    parser.add_argument('--cl', dest='pop_clean', help='clean population', type=str, default='None')
    parser.add_argument('--r', dest='random_seed', help='random seed', type=int, default=None)

    options=parser.parse_args(argv[1:])

    datafolder= options.datafolder
    outputpath = options.outputpath
    if not os.path.isdir(datafolder):
        raise NotADirectoryError(datafolder)
    os.makedirs(outputpath, exist_ok=True)
    if options.manual_bin_init == 'None':
        manual_bin_init = None
    else:
        manual_bin_init = pd.read_csv(options.manual_bin_init,low_memory=False)
    loci_list = [item.strip() for item in options.loci_list.split(',') if item.strip()]
    if options.cov_list == 'None':
        cov_list = None
    else:
        cov_list = [item.strip() for item in options.cov_list.split(',') if item.strip()]
    rare_filter = options.rare_filter
    cv = options.cv
    if not 0.0 <= rare_filter <= 1.0:
        raise ValueError("--ra must be between 0 and 1")
    if cv < 1:
        raise ValueError("--cv must be at least 1")

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

    if desired_bin_effect not in ["default", "protective", "high_risk"]:
        raise Exception("'desired_bin_effect' must be one of: 'default', 'protective', 'high_risk'")

    valid_loci = {'A', 'B', 'C', 'DRB1', 'DRB345', 'DQA1', 'DQB1', 'DPA1', 'DPB1'}
    invalid_loci = sorted(set(loci_list) - valid_loci)
    invalid_cov_loci = sorted(set(cov_list or []) - valid_loci)
    if invalid_loci or invalid_cov_loci:
        raise ValueError(
            "Unknown loci: " + ", ".join(invalid_loci + invalid_cov_loci)
        )

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

    train_suffix = f"_{cv}_Train.csv"
    train_files = sorted(
        filename
        for filename in os.listdir(datafolder)
        if filename.endswith(train_suffix)
        and os.path.isfile(os.path.join(datafolder, filename))
    )
    if len(train_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{train_suffix}' file in {datafolder}; found {len(train_files)}"
        )
    train_filename = train_files[0]
    test_filename = train_filename[:-len(train_suffix)] + f"_{cv}_Test.csv"
    data_full_train = os.path.join(datafolder, train_filename)
    data_full_test = os.path.join(datafolder, test_filename)
    if not os.path.isfile(data_full_test):
        raise FileNotFoundError(data_full_test)

    #Load/Process Dataset
    original_train_data = pd.read_csv(data_full_train)

    #Identify MM features to include as independent features
    MM_feature_list = []
    for locus in loci_list: #each specified locus to evaluate as independent features
        for j in range(locus_range_dict[locus][0],locus_range_dict[locus][1]+1):
            MM_feature_list.append('MM_'+str(locus)+'_'+str(j))

    MM_feature_list = [feature for feature in MM_feature_list if feature in original_train_data.columns]
    if len(MM_feature_list) == 0:
        raise ValueError("No requested MM features were found in the training fold")
    required_train_columns = final_covariates + [outcome_label] + [censor_label]
    missing_train_columns = [
        column for column in required_train_columns if column not in original_train_data.columns
    ]
    if missing_train_columns:
        raise ValueError(
            "Training fold is missing required columns: " + ", ".join(missing_train_columns)
        )
    features = MM_feature_list + required_train_columns
    train_data = original_train_data.loc[:, features].copy()

    #Missing data values check
    missing_sum = train_data.isna().sum().sum()
    if missing_sum > 0:
        print("Sum of data missing values:", missing_sum)

    #Data Cleaning
    # Calculate the percentage of occurrences greater than 0 for each MM feature.
    percentages = train_data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
    print(percentages)

    if rare_filter > 0.0: #filter out rare features and invariant features
        columns_to_remove = percentages[percentages < rare_filter].index.tolist()
        filter_rule = f"frequency < {rare_filter}"
        filter_threshold = rare_filter
    else: #filter out invariant features only
        columns_to_remove = percentages[percentages == 0.0].index.tolist()
        filter_rule = "frequency == 0.0"
        filter_threshold = 0.0

    train_data = train_data.drop(columns=columns_to_remove)
    MM_feature_list = [
        feature for feature in MM_feature_list if feature not in columns_to_remove
    ]
    if len(MM_feature_list) < min_bin_size:
        raise ValueError(
            f"Rare filtering left {len(MM_feature_list)} MM features, fewer than min_bin_size={min_bin_size}"
        )
    max_bin_init_size = min(max_bin_init_size, len(MM_feature_list))
    if max_bin_size is not None:
        max_bin_size = min(max_bin_size, len(MM_feature_list))

    percentages_df = (
        percentages.rename("nonzero_fraction")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(by="feature")
    )
    percentages_df["nonzero_percent"] = percentages_df["nonzero_fraction"] * 100.0
    percentages_df.to_csv(outputpath+'/'+str(cv)+'_rare_filter_percentages.csv', index=False)

    frequency_df = (
        percentages.rename("nonzero_fraction")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    frequency_df["nonzero_percent"] = frequency_df["nonzero_fraction"] * 100.0
    frequency_df["filter_threshold"] = filter_threshold
    frequency_df["filter_rule"] = filter_rule
    frequency_df["removed_by_filter"] = frequency_df["feature"].isin(columns_to_remove)
    frequency_df["kept_for_training"] = ~frequency_df["removed_by_filter"]
    frequency_df = frequency_df.sort_values(
        by=["removed_by_filter", "nonzero_fraction", "feature"],
        ascending=[False, True, True],
    )
    frequency_df.to_csv(outputpath+'/'+str(cv)+'_rare_filter_feature_frequencies.csv', index=False)

    #Report filtering
    count_list = []
    total_count = 0
    for locus in loci_list:
        count = sum(['MM_'+str(locus) in col for col in train_data.columns])
        total_count += count
        count_list.append(str(locus)+":"+str(count))

    with open(outputpath+'/'+str(cv)+'_post_filter_counts.txt', 'w') as file:
        file.write('RareFilterThreshold:'+str(filter_threshold)+'\n')
        file.write('RareFilterRule:'+filter_rule+'\n')
        file.write('RemovedFeatures:'+str(len(columns_to_remove))+'\n')
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
    save_summary(
        summary,
        os.path.join(outputpath, f"{cv}_coxph_unadj_bin_train_{bin_index}.csv"),
        "Training-fold unadjusted Cox model",
    )

    #Kaplan Meir Plot
    save_kaplan_meier(fibers, train_data, bin_index, outputpath, f"{cv}_train")

    if final_covariates != None:
        summary = fibers.get_cox_prop_hazard_adjusted(train_data, y, bin_index, use_bin_sums, show_progress)
        save_summary(
            summary,
            os.path.join(outputpath, f"{cv}_coxph_adj_bin_train_{bin_index}.csv"),
            "Training-fold adjusted Cox model",
        )
        if final_covariates != covariates:
            train_data_no_ag = train_data.drop(columns=Ag_covariates)
            summary = fibers.get_cox_prop_hazard_adjusted(
                train_data_no_ag,
                y,
                bin_index,
                use_bin_sums,
                show_progress,
                covariates,
            )
            save_summary(
                summary,
                os.path.join(outputpath, f"{cv}_coxph_adj_bin_train_{bin_index}_NoAg.csv"),
                "Training-fold adjusted Cox model without antigen covariates",
            )

    #Load/Process Dataset
    original_test_data = pd.read_csv(data_full_test)
    required_test_columns = MM_feature_list + [outcome_label] + [censor_label]
    missing_test_columns = [
        column for column in required_test_columns if column not in original_test_data.columns
    ]
    if missing_test_columns:
        raise ValueError(
            "Testing fold is missing required columns: " + ", ".join(missing_test_columns)
        )
    available_test_covariates = [
        column for column in final_covariates if column in original_test_data.columns
    ]
    missing_test_covariates = sorted(set(final_covariates) - set(available_test_covariates))
    if missing_test_covariates:
        print(
            "Warning: testing fold is missing covariates that will be omitted from adjusted analysis: "
            + ", ".join(missing_test_covariates)
        )
    features = required_test_columns + available_test_covariates
    test_data = original_test_data.loc[:, list(dict.fromkeys(features))].copy()

    summary = fibers.get_cox_prop_hazard_unadjust(test_data, y, bin_index, use_bin_sums, show_progress)
    save_summary(
        summary,
        os.path.join(outputpath, f"{cv}_coxph_unadj_bin_test_{bin_index}.csv"),
        "Testing-fold unadjusted Cox model",
    )

    #Kaplan Meir Plot
    save_kaplan_meier(fibers, test_data, bin_index, outputpath, f"{cv}_test")

    if available_test_covariates:
        summary = fibers.get_cox_prop_hazard_adjusted(
            test_data,
            y,
            bin_index,
            use_bin_sums,
            show_progress,
            available_test_covariates,
        )
        save_summary(
            summary,
            os.path.join(outputpath, f"{cv}_coxph_adj_bin_test_{bin_index}.csv"),
            "Testing-fold adjusted Cox model",
        )

        if final_covariates != covariates:
            available_base_covariates = [
                column for column in covariates if column in test_data.columns
            ]
            test_data_no_ag = test_data.drop(
                columns=[column for column in Ag_covariates if column in test_data.columns]
            )
            summary = fibers.get_cox_prop_hazard_adjusted(
                test_data_no_ag,
                y,
                bin_index,
                use_bin_sums,
                show_progress,
                available_base_covariates,
            )
            save_summary(
                summary,
                os.path.join(outputpath, f"{cv}_coxph_adj_bin_test_{bin_index}_NoAg.csv"),
                "Testing-fold adjusted Cox model without antigen covariates",
            )

if __name__=="__main__":
    sys.exit(main(sys.argv))
