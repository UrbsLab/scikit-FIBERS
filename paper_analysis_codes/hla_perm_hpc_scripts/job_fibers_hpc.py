import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import re
from pathlib import Path
# Use FIBERS from the checkout containing this job.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

VALID_EFFECTS = ("default", "protective", "high_risk")
IGNORE_COLUMNS = (
    "PX_ID", "PERS_ID", "TX_ID", "DONOR_ID", "REC_TX_DT",
    "TFL_GRAFT_DT", "TFL_DEATH_DT", "TFL_LAFUDATE", "DON_TY",
    "REC_A_MM_EQUIV_CUR", "REC_B_MM_EQUIV_CUR", "REC_C_MM_EQUIV_CUR",
    "REC_DR_MM_EQUIV_CUR", "REC_DQA1_MM_EQUIV_CUR", "REC_DQ_MM_EQUIV_CUR",
    "REC_DPA1_MM_EQUIV_CUR", "REC_DPB1_MM_EQUIV_CUR",
)

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datapath', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'A,B,C,DRB1,DQA1,DQB1')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0)

    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='graftyrs')
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
    parser.add_argument('--c', dest='censor_label', help='censor column label', type=str, default='grf_fail')
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.2)
    parser.add_argument('--p', dest='penalty', help='group strata min penalty', type=float, default=0.5)
    parser.add_argument('--t', dest='group_thresh', help='group threshold', type=str, default=0)
    parser.add_argument('--it', dest='min_thresh', help='minimum threshold', type=int, default=0)
    parser.add_argument('--at', dest='max_thresh', help='maximum threshold', type=int, default=5)
    #int_thresh
    parser.add_argument('--te', dest='thresh_evolve_prob', help='threshold evolution probability', type=float, default=0.5)
    parser.add_argument('--de', dest='desired_bin_effect', help='desired bin effect', type=str, choices=VALID_EFFECTS, default='default')
    parser.add_argument('--cl', dest='pop_clean', help='clean population', type=str, default='None')
    parser.add_argument('--r', dest='random_seed', help='random seed', type=int, default=None)

    parser.add_argument('--all-positions', action='store_true', help='Compatibility option: all observed AA positions are always used')

    options=parser.parse_args(argv[1:])

    datapath= options.datapath
    outputpath = options.outputpath
    os.makedirs(outputpath, exist_ok=True)
    if options.manual_bin_init == 'None':
        manual_bin_init = None
    else:
        manual_bin_init = pd.read_csv(options.manual_bin_init,low_memory=False)
    loci_list = list(dict.fromkeys(item.strip() for item in options.loci_list.split(',') if item.strip()))
    if options.cov_list.lower() == 'none':
        cov_list = None
    else:
        cov_list = list(dict.fromkeys(item.strip() for item in options.cov_list.split(',') if item.strip()))
    rare_filter = options.rare_filter
    if not 0 <= rare_filter <= 1:
        raise ValueError("--ra must be between 0 and 1")

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
    covariates = None #Manually included in script
    if options.pop_clean == 'None':
        pop_clean = None
    else:
        pop_clean = str(options.pop_clean)
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

    # Disabled: raw antigen counts are not used for training.
    # raw_antigen_columns = {
    #     'A': 'REC_A_MM_EQUIV_CUR', 'B': 'REC_B_MM_EQUIV_CUR',
    #     'C': 'REC_C_MM_EQUIV_CUR', 'DRB1': 'REC_DR_MM_EQUIV_CUR',
    #     'DRB345': 'REC_DR_MM_EQUIV_CUR', 'DQA1': 'REC_DQA1_MM_EQUIV_CUR',
    #     'DQB1': 'REC_DQ_MM_EQUIV_CUR', 'DPA1': 'REC_DPA1_MM_EQUIV_CUR',
    #     'DPB1': 'REC_DPB1_MM_EQUIV_CUR',
    # }

    # Historical position limits retained for reference; filtering is disabled.
    # locus_range_dict = {
    #     'A': [1,182],
    #     'B': [1,182],
    #     'C': [1,182],
    #     'DRB1': [6,94],
    #     'DRB345': [6,94],
    #     'DQA1':  [6,94],
    #     'DQB1':  [6,95],
    #     'DPA1':  [6,94],
    #     'DPB1':  [6,94]}

    base_covariates = covariates[:]
    if not loci_list or set(loci_list) - set(cov_typ_dict) or set(cov_list or []) - set(cov_typ_dict):
        raise ValueError("Unknown or empty locus list")

    #Create Final Covariate List
    if cov_list != None:
        for covariate in cov_list:
            cov_sub_list = cov_typ_dict[covariate]
            for each in cov_sub_list:
                covariates.append(each) #add selected Ag covariate to primary covariates

    covariates = list(dict.fromkeys(covariates))

    # Get Dataset Name
    filename = os.path.basename(datapath)
    dataset_name,ext = os.path.splitext(filename)

    #Load/Process Dataset
    data = pd.read_csv(datapath, usecols=lambda column: column not in IGNORE_COLUMNS)

    # Use only source antigen indicators present in the dataset; base covariates stay required.
    covariates = [column for column in covariates if column in base_covariates or column in data]
    print(covariates)

    # Disabled: retain source Agmm values without reconstruction.
    # for locus in cov_list or []:
    #     missing_indicators = [column for column in cov_typ_dict[locus] if column not in data]
    #     if missing_indicators:
    #         source_column = raw_antigen_columns[locus]
    #         if source_column not in data:
    #             raise ValueError('Missing required antigen indicators or raw count: '+source_column)
    #         raw_counts = pd.to_numeric(data[source_column], errors='raise')
    #         if not raw_counts.isin([0, 1, 2]).all():
    #             raise ValueError(source_column+' must contain only 0, 1, or 2')
    #         for count, column in enumerate(cov_typ_dict[locus]):
    #             if column in missing_indicators:
    #                 data[column] = (raw_counts == count).astype(int)

    # Donor-to-recipient permutations only. rF_dY means donor Y -> recipient F.
    permutation_pattern = re.compile(r"MM_(A|B|C|DRB1|DRB345|DQA1|DQB1|DPA1|DPB1)_([1-9][0-9]*)_r[A-Z]_d[A-Z]")
    MM_feature_list = []
    for feature in data.columns:
        match = permutation_pattern.fullmatch(feature)
        if match and match[1] in loci_list:
            # lower, upper = locus_range_dict[match[1]]
            # if options.all_positions or lower <= int(match[2]) <= upper:
            MM_feature_list.append(feature)
    if not MM_feature_list:
        raise ValueError("No requested permutation features (e.g. MM_A_9_rF_dY) were found")
    features = MM_feature_list + covariates + [outcome_label, censor_label]
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = data.loc[:, features].apply(pd.to_numeric, errors='raise')
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Model data must contain rows and finite numeric values")
    counts = data.loc[:, MM_feature_list].to_numpy(dtype=float)
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Permutation mismatches must be nonnegative integer counts")
    if (data[outcome_label] < 0).any() or not data[censor_label].isin([0, 1]).all():
        raise ValueError("Invalid survival times or event indicators")

    # Filter on the training data only; the test fold reuses the retained columns.
    percentages = (data.loc[:, MM_feature_list] > 0).mean()
    removed = percentages < rare_filter if rare_filter > 0 else percentages == 0
    columns_to_remove = percentages.index[removed].tolist()
    data = data.drop(columns=columns_to_remove)
    MM_feature_list = [feature for feature in MM_feature_list if feature not in columns_to_remove]
    if len(MM_feature_list) < min_bin_size:
        raise ValueError("Too few permutation features remain after rare filtering")
    max_bin_init_size = min(max_bin_init_size, len(MM_feature_list))
    if max_bin_size is not None:
        max_bin_size = min(max_bin_size, len(MM_feature_list))
    run_prefix = dataset_name+'_'+str(random_seed)
    pd.DataFrame({'feature': percentages.index, 'nonzero_fraction': percentages.values,
                  'kept_for_training': (~removed).values}).to_csv(
        os.path.join(outputpath, run_prefix+'_rare_filter_feature_frequencies.csv'), index=False)

    #Report filtering
    count_list = []
    total_count = 0
    for locus in loci_list:
        count = sum(col.startswith('MM_'+str(locus)+'_') for col in MM_feature_list)
        total_count += count
        count_list.append(str(locus)+":"+str(count))

    with open(outputpath+'/'+dataset_name+'_'+str(random_seed)+'_post_filter_counts.txt', 'w') as file:
        for item in count_list:
            file.write(f"{item}\n")
        file.write('Total:'+str(total_count))

    #Job Definition
    fibers = FIBERS(outcome_label=outcome_label, outcome_type=outcome_type, iterations=iterations, pop_size=pop_size, tournament_prop=tournament_prop,
                    crossover_prob=crossover_prob, min_mutation_prob=min_mutation_prob, max_mutation_prob=max_mutation_prob, merge_prob=merge_prob,
                    new_gen=new_gen, elitism=elitism, diversity_pressure=diversity_pressure, min_bin_size=min_bin_size, max_bin_size=max_bin_size,
                    max_bin_init_size=max_bin_init_size, fitness_metric=fitness_metric, log_rank_weighting=log_rank_weighting, censor_label=censor_label,
                    group_strata_min=group_strata_min, penalty=penalty, group_thresh=group_thresh, min_thresh=min_thresh, max_thresh=max_thresh,
                    int_thresh=True, thresh_evolve_prob=thresh_evolve_prob, desired_bin_effect=desired_bin_effect,
                    manual_bin_init=manual_bin_init, covariates=covariates, pop_clean=pop_clean,
                    report=None, random_seed=random_seed, verbose=False)

    fibers = fibers.fit(data)
    output = Path(outputpath)
    save_model(fibers, output, run_prefix)
    if not fibers.set.bin_pop:
        (output / (run_prefix+'_no_valid_bins.txt')).write_text('No bins remained after training/cleanup.\n')
        return
    evaluate_model(fibers, data, output, run_prefix, "", covariates, base_covariates)
    save_model(fibers, output, run_prefix)


def save_model(model, output, prefix):
    model.get_pop().to_csv(output / f"{prefix}_pop.csv", index=False)
    with (output / f"{prefix}_fibers.pickle").open("wb") as handle:
        pickle.dump(model, handle)
    model.save_run_params(str(output / f"{prefix}_run_parameters.txt"))

def evaluate_model(model, data, output, prefix, split, covariates, base_covariates):
    """Keep fitted artifacts even when a secondary Cox model cannot be estimated."""
    suffix = f"_{split}" if split else ""
    analyses = [
        (f"coxph_unadj_bin{suffix}_0", lambda: model.get_cox_prop_hazard_unadjust(data)),
        (f"coxph_adj_bin{suffix}_0", lambda: model.get_cox_prop_hazard_adjusted(data)),
    ]
    antigen_covariates = [column for column in covariates if column not in base_covariates]
    if antigen_covariates and split:
        analyses.insert(0, (
            f"coxph_adj_bin{suffix}_0_NoAg",
            lambda: model.get_cox_prop_hazard_adjusted(
                data.drop(columns=antigen_covariates), new_covariates=list(base_covariates)
            ),
        ))
    original_features = model.feature_names[:]
    for name, calculate in analyses:
        try:
            summary = calculate()
            if summary is None:
                raise ValueError("CoxPHFitter could not estimate this model.")
            summary.to_csv(output / f"{prefix}_{name}.csv", index=True)
        except Exception as error:
            failure = output / f"{prefix}_{name}_failed.txt"
            failure.write_text(str(error) + "\n")
            print(f"Warning: {name} was not estimable; details: {failure}")
        finally:
            model.feature_names = original_features[:]
    try:
        model.get_kaplan_meir(data, 0, save=True, show=False,
                             output_folder=str(output), data_name=f"{prefix}{suffix}")
    except Exception as error:
        failure = output / f"{prefix}{suffix}_kaplan_meier_failed.txt"
        failure.write_text(str(error) + "\n")
        print(f"Warning: Kaplan-Meier plot failed; details: {failure}")

if __name__ == '__main__':
    sys.exit(main(sys.argv))
