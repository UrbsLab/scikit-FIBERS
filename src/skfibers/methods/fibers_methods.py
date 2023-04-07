import numpy as np
from tqdm import tqdm
from .common_methods import remove_empty_variables
from .common_methods import random_feature_grouping
from .common_methods import grouped_feature_matrix
from .common_methods import crossover_and_mutation_multiprocess as crossover_and_mutation
from .common_methods import create_next_generation
from .common_methods import regroup_feature_matrix
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test


# Functions for FIBERS


def log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name, duration_name,
                                     informative_cutoff):
    bin_scores = {}
    for Bin_name in amino_acid_bins.keys():
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[Bin_name] == 0]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[Bin_name] > 0]

        durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        durations_mm = df_1[duration_name].to_list()
        event_observed_mm = df_1[label_name].to_list()

        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                                   event_observed_B=event_observed_mm)
            bin_scores[Bin_name] = results.test_statistic

        else:
            bin_scores[Bin_name] = 0

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0

    return bin_scores


def log_rank_test_feature_importance_new(bin_feature_matrix,
                                         amino_acid_bins, label_name, duration_name, informative_cutoff):
    bin_scores = {}

    if duration_name is None:
        raise Exception("No Survival/Duration column name given")

    for Bin_name in amino_acid_bins.keys():

        df_0 = bin_feature_matrix.loc[bin_feature_matrix[Bin_name] == 0]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[Bin_name] == 1]
        df_1 = df_1.append(bin_feature_matrix.loc[bin_feature_matrix[Bin_name] == 1], ignore_index=True)
        df_2 = bin_feature_matrix.loc[bin_feature_matrix[Bin_name] > 2]

        if duration_name:
            durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        group_no = list([0] * len(event_observed_no))

        if duration_name:
            durations_mm1 = df_1[duration_name].to_list()
        event_observed_mm1 = df_1[label_name].to_list()
        group_mm1 = list([1] * len(event_observed_mm1))

        if duration_name:
            durations_mm2 = df_2[duration_name].to_list()
        event_observed_mm2 = df_2[label_name].to_list()
        group_mm2 = list([2] * len(event_observed_mm2))

        total_len = len(event_observed_no) + len(event_observed_mm1) + len(event_observed_mm2)

        if duration_name:
            all_durations = durations_no + durations_mm1 + durations_mm2
        else:
            all_durations = 0
        all_events = event_observed_no + event_observed_mm1 + event_observed_mm2
        groups = group_no + group_mm1 + group_mm2

        if len(event_observed_no) > informative_cutoff * total_len and len(
                event_observed_mm1) > informative_cutoff * total_len \
                and len(event_observed_mm2) > informative_cutoff * total_len:
            results = multivariate_logrank_test(all_durations, groups, all_events)
            bin_scores[Bin_name] = results.test_statistic

        else:
            bin_scores[Bin_name] = 0

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0

    return bin_scores


# FIBERS Algorithm
def fibers_algorithm(given_starting_point,
                     amino_acid_start_point,
                     amino_acid_bins_start_point,
                     iterations,
                     original_feature_matrix,
                     label_name, duration_name,
                     set_number_of_bins,
                     min_features_per_group,
                     max_number_of_groups_with_feature,
                     informative_cutoff,
                     crossover_probability,
                     mutation_probability,
                     elitism_parameter, random_seed,
                     bin_size_variability_constraint,
                     max_features_per_bin):
    # Step 0: Deleting Empty Features (MAF = 0)
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list = remove_empty_variables(
        original_feature_matrix,
        label_name, duration_name)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups

    # If there is a starting point, use that for the amino acid list and the amino acid bins list
    if given_starting_point:
        # Keep only MAF != 0 features from starting points in amino_acids and amino_acid_bins
        amino_acids = list(set(amino_acid_start_point).intersection(nonempty_feature_list))

        amino_acid_bins = amino_acid_bins_start_point.copy()
        bin_names = amino_acid_bins.keys()
        features_to_remove = [item for item in amino_acid_start_point if item not in nonempty_feature_list]
        for bin_name in bin_names:
            # Remove duplicate features
            amino_acid_bins[bin_name] = list(set(amino_acid_bins[bin_name]))
            for feature in features_to_remove:
                if feature in amino_acid_bins[bin_name]:
                    amino_acid_bins[bin_name].remove(feature)

    # Otherwise randomly initialize the bins
    elif not given_starting_point:
        amino_acids, amino_acid_bins = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                               duration_name,
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature, random_seed,
                                                               max_features_per_bin)

    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, duration_name,
                                                amino_acid_bins)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    np.random.seed(random_seed)
    upper_bound = (len(maf_0_features) + len(nonempty_feature_list)) * (
            len(maf_0_features) + len(nonempty_feature_list))
    random_seeds = np.random.randint(upper_bound, size=iterations * 2)
    for i in tqdm(range(0, iterations)):
        # print("iteration:" + str(i))

        # Step 2a: Feature Importance Scoring and Bin Deletion
        amino_acid_bin_scores = log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name,
                                                                 duration_name, informative_cutoff)

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = crossover_and_mutation(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins,
                                                amino_acid_bin_scores,
                                                crossover_probability, mutation_probability, random_seeds[i],
                                                bin_size_variability_constraint,
                                                max_features_per_bin)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = create_next_generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        bin_feature_matrix, amino_acid_bins = regroup_feature_matrix(amino_acids, original_feature_matrix, label_name,
                                                                     duration_name, feature_bin_list,
                                                                     random_seeds[iterations + i])

    # Creating the final amino acid bin scores
    amino_acid_bin_scores = log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name,
                                                             duration_name, informative_cutoff)

    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, maf_0_features


def top_bin_summary_fibers(original_feature_matrix, label_name, duration_name, bin_feature_matrix, bins, bin_scores):
    # Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())

    topbin = sorted_bin_list[0]

    df_0 = bin_feature_matrix.loc[bin_feature_matrix[topbin] == 0]
    df_1 = bin_feature_matrix.loc[bin_feature_matrix[topbin] == 1]
    df_1 = df_1.append(bin_feature_matrix.loc[bin_feature_matrix[topbin] == 2], ignore_index=True)
    df_2 = bin_feature_matrix.loc[bin_feature_matrix[topbin] > 2]

    durations_no = df_0[duration_name].to_list()
    event_observed_no = df_0[label_name].to_list()
    group_no = list([0] * len(event_observed_no))

    durations_mm1 = df_1[duration_name].to_list()
    event_observed_mm1 = df_1[label_name].to_list()
    group_mm1 = list([1] * len(event_observed_no))

    durations_mm2 = df_2[duration_name].to_list()
    event_observed_mm2 = df_2[label_name].to_list()
    group_mm2 = list([2] * len(event_observed_no))

    all_durations = durations_no + durations_mm1 + durations_mm2
    all_events = event_observed_no + event_observed_mm1 + event_observed_mm2
    groups = group_no + group_mm1 + group_mm2

    results = multivariate_logrank_test(all_durations, groups, all_events)

    print("Bin of Amino Acid Positions:")
    print(bins[topbin])
    print("---")
    print("Number of Instances with No Mismatches in Bin:")
    print(len(event_observed_no))
    print("Number of Instances with 1-2 Mismatches in Bin:")
    print(len(event_observed_mm1))
    print("Number of Instances with >2 Mismatches in Bin:")
    print(len(event_observed_mm2))
    print("---")
    print("p-value from Log Rank Test:")
    print(results.p_value)
    results.print_summary()

    kmf1 = KaplanMeierFitter()

    # fit the model for 1st cohort
    kmf1.fit(durations_no, event_observed_no, label='No Mismatches in Bin')

    a1 = kmf1.plot_survival_function()
    a1.set_ylabel('Survival Probability')

    # fit the model for 2nd cohort
    kmf1.fit(durations_mm1, event_observed_mm1, label='1-2 Mismatch(es) in Bin')
    kmf1.plot_survival_function(ax=a1)
    a1.set_xlabel('Years After Transplant')

    # fit the model for 3rd cohort
    kmf1.fit(durations_mm2, event_observed_mm2, label='>2 Mismatch(es) in Bin')
    kmf1.plot_survival_function(ax=a1)
    a1.set_xlabel('Years After Transplant')
