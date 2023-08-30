import math
import random
import statistics
from random import randrange
import time
from warnings import simplefilter
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from tqdm import tqdm
from .bin import BIN

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=DeprecationWarning)


# Defining a function to delete variables with MAF = 0
def remove_empty_variables(original_feature_matrix, label_name, duration_name):
    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name, duration_name])

    # Calculating the MAF of each feature
    maf = list(feature_df.sum() / (2 * len(feature_df.index)))

    # Creating a df of features and their MAFs
    feature_maf_df = pd.DataFrame(feature_df.columns, columns=['feature'])
    feature_maf_df['maf'] = maf

    maf_0_df = feature_maf_df.loc[feature_maf_df['maf'] == 0]
    maf_not0_df = feature_maf_df.loc[feature_maf_df['maf'] != 0]

    # Creating a list of features with MAF = 0
    maf_0_features = list(maf_0_df['feature'])

    # Saving the feature list of nonempty features
    nonempty_feature_list = list(maf_not0_df['feature'])

    # Creating feature matrix with only features where MAF != 0
    feature_matrix_no_empty_variables = feature_df[nonempty_feature_list]

    # Adding the class label to the feature matrix
    feature_matrix_no_empty_variables[label_name] = original_feature_matrix[label_name]
    feature_matrix_no_empty_variables[duration_name] = original_feature_matrix[duration_name]

    return feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list


# Defining a function to group features randomly, each feature can be in a number of groups up to a set max
def random_feature_grouping(feature_matrix, label_name, duration_name, number_of_groups, min_features_per_group,
                            max_number_of_groups_with_feature, random_seed, threshold, max_features_per_bin=None):
    # Removing the label column to create a list of features
    feature_df = feature_matrix.drop(columns=[label_name, duration_name])

    # Creating a list of features
    feature_list = list(feature_df.columns)

    # Adding a random number of repeats of the features so that features can be in more than one group
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=len(feature_list))
    for w in range(0, len(feature_list)):
        random.seed(random_seeds[w])
        repeats = randrange(max_number_of_groups_with_feature)
        feature_list.extend([feature_list[w]] * repeats)

    # Shuffling the feature list to enable random groups
    random.seed(random_seed)
    random.shuffle(feature_list)

    # Creating a dictionary of the groups
    feature_groups = {}

    # Assigns the minimum number of features to all the groups
    for x in range(0, min_features_per_group * number_of_groups, min_features_per_group):
        feature_groups[x / min_features_per_group] = feature_list[x:x + min_features_per_group]

    # Randomly distributes the remaining features across the set number of groups
    try:
        np.random.seed(random_seed)
        random_seeds = np.random.randint(len(feature_list) * len(feature_list),
                                         size=(len(feature_list) - min_features_per_group * number_of_groups))
        for y in range(min_features_per_group * number_of_groups, len(feature_list)):
            random.seed(random_seeds[y - min_features_per_group * number_of_groups])
            feature_groups[random.choice(list(feature_groups.keys()))].append(feature_list[y])
    except Exception:
        pass

    # # Removing duplicates of features in the same bin
    # for z in range(0, len(feature_groups)):
    #     feature_groups[z] = list(set(feature_groups[z]))
    #
    #     # Randomly removing features until the number of features is equal to or less than the max_features_per_bin
    #     # param
    #     if not (max_features_per_bin is None):
    #         if len(feature_groups[z]) > max_features_per_bin:
    #             random.seed(random_seeds[z])
    #             feature_groups[z] = list(random.sample(feature_groups[z], max_features_per_bin))

    # Removing duplicates of features in the same bin
    for z in range(0, len(feature_groups)):
        unique = []
        for a in range(0, len(feature_groups[z])):
            if feature_groups[z][a] not in unique:
                unique.append(feature_groups[z][a])
        feature_groups[z] = unique

    # Creating a dictionary with bin labels, and instances of BIN class sphia
    binned_feature_groups = {}
    for index in range(0, len(feature_groups)):
        # SPHIA
        binned_feature_groups["Bin " + str(index + 1)] = BIN(feature_groups[index], threshold,
                                                             "Bin " + str(index + 1))

    return feature_list, binned_feature_groups


# Defining a function to create a feature matrix where each feature is a bin of features from the original feature
# matrix

def grouped_feature_matrix(feature_matrix, label_name, duration_name, binned_feature_groups):
    # Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()

    # Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = [0] * len(feature_matrix.index)

    # Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    # The list and dummy data frame will be used for adding later

    # For each feature group/bin, the values of the amino acid in the bin will be summed to create a value for the bin
    for key in binned_feature_groups:
        sum_column = dummy_df['Zeros']
        for j in range(0, len(binned_feature_groups[key])):
            sum_column = sum_column + feature_matrix[binned_feature_groups[key][j]]
        bins_df[key] = sum_column

    # Adding the class label to the data frame
    bins_df[label_name] = feature_matrix[label_name]
    bins_df[duration_name] = feature_matrix[duration_name]
    return bins_df


def log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name, duration_name,
                                     informative_cutoff):  # SPHIA
    bin_scores = {}
    for bin_name in amino_acid_bins.keys():

        # To not repeat calculations, if this bin has been seen (meaning evaluated before by the log rank test), it will not
        # calculate the score again
        if (not amino_acid_bins[bin_name].was_seen()):
            df_0 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] <=
                                          amino_acid_bins[bin_name].get_threshold()]  # SPHIA
            df_1 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] >
                                          amino_acid_bins[bin_name].get_threshold()]

            durations_no = df_0[duration_name].to_list()
            event_observed_no = df_0[label_name].to_list()
            durations_mm = df_1[duration_name].to_list()
            event_observed_mm = df_1[label_name].to_list()

            if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                    event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
                results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                                       event_observed_B=event_observed_mm)
                bin_scores[bin_name] = results.test_statistic
                amino_acid_bins[bin_name].set_score(results.test_statistic)
            else:
                bin_scores[bin_name] = 0
                amino_acid_bins[bin_name].set_score(0)

            amino_acid_bins[bin_name].set_seen()

        else:
            bin_scores[bin_name] = amino_acid_bins[bin_name].get_score()

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0
            amino_acid_bins[bin_name].set_score(0)
            amino_acid_bins[bin_name].set_seen()

    return bin_scores


# Defining a function to probabilistically select 2 parent bins based on their feature importance rank
# Tournament Selection works in this case by choosing a random sample of the bins and choosing the best two scores
def tournament_selection_parent_bins(binned_feature_groups, random_seed):
    random.seed(random_seed)

    # Choosing a random sample of 5% of the bin population or if that would be too small, choosing a sample of 50%
    if round(0.05 * len(binned_feature_groups)) < 2:
        samples = random.sample(list(binned_feature_groups.values()), round(0.5 * len(binned_feature_groups)))
    else:
        samples = random.sample(list(binned_feature_groups.values()), round(0.05 * len(binned_feature_groups)))

    # Sorting the bins from best score to worst score
    sorted_bins = sorted(samples, reverse=True)

    # Choosing the parent bins and adding them to a list of parent bins
    parent_bins = [sorted_bins[0], sorted_bins[1]]

    return parent_bins


def create_next_generation(binned_feature_groups, max_population_of_bins, elitism_parameter,
                           offspring_list):
    # Sorting the bins from best score to worst score
    # sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    # sorted_bin_list = list(sorted_bin_scores.keys())

    sorted_bins = sorted(list(binned_feature_groups.values()), reverse=True)
    # Determining the number of elite bins
    number_of_elite_bins = round(max_population_of_bins * elitism_parameter)
    elites = []
    # Adding the elites to a list of elite feature bins
    for bin in range(0, number_of_elite_bins):
        elites.append(sorted_bins[bin])

    # Creating a list of feature bins (without labels because those will be changed as things get deleted and added)
    feature_bin_list = elites.copy()

    # Adding the offspring to the feature bin list
    feature_bin_list.extend(offspring_list)

    return feature_bin_list


# Defining a function to recreate the feature matrix (add up values of amino acids from original dataset)
def regroup_feature_matrix(feature_list, feature_matrix, label_name, duration_name, feature_bin_list, random_seed,
                           threshold):
    # First deleting any bins that are empty  
    bins_deleted = [x for x in feature_bin_list if len(x) == 0]
    feature_bin_list = [x for x in feature_bin_list if len(x) != 0]

    # Checking each pair of bins, if the bins are duplicates then one of the copies will be deleted
    no_duplicates = []
    num_duplicates = 0
    for Bin in feature_bin_list:
        if Bin not in no_duplicates:
            no_duplicates.append(Bin)
        else:
            num_duplicates += 1

    feature_bin_list = no_duplicates

    # Calculate average length of nonempty bins in the population
    bin_lengths = [len(x) for x in feature_bin_list if len(x) != 0]
    replacement_length = round(statistics.mean(bin_lengths))

    # ---------- DIFFERENT FROM ORIGINAL HERE

    # Replacing each deleted bin with a bin with random features
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list),
                                     size=((len(bins_deleted) + num_duplicates) * 2))

    for i in range(0, len(bins_deleted) + num_duplicates):
        random.seed(random_seeds[i])
        replacement = random.sample(feature_list, replacement_length)

        random.seed(random_seeds[len(bins_deleted) + num_duplicates + i])
        random_seeds_replacement = np.random.randint(len(feature_list) * len(feature_list),
                                                     size=2 * len(feature_bin_list))
        counter = 0
        replacement = BIN(replacement, threshold=threshold)
        while replacement in feature_bin_list:
            random.seed(random_seeds_replacement[counter])
            replacement = BIN(random.sample(feature_list, replacement_length), threshold=threshold)
            counter = counter + 1
        feature_bin_list.append(replacement)

    random_seeds_replacement = np.random.randint(len(feature_list) * len(feature_list),
                                                 size=2 * len(feature_bin_list))
    counter = 0

    # Deleting duplicate features in the same bin and replacing them with random features
    for Bin in range(0, len(feature_bin_list)):
        unique = []
        for a in range(0, len(feature_bin_list[Bin])):
            if feature_bin_list[Bin][a] not in unique:
                unique.append(feature_bin_list[Bin][a])

        replace_number = len(feature_bin_list[Bin]) - len(unique)

        features_not_in_offspring = [item for item in feature_list if item not in feature_bin_list[Bin]]

        bin_replacement = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            random.seed(random_seeds_replacement[counter])
            replacements = random.sample(features_not_in_offspring, replace_number)
            counter += 1
        else:
            replacements = features_not_in_offspring.copy()
        bin_replacement.extend(replacements)

        # only new feature lists will change the BIN not seen attribute to False
        if (feature_bin_list[Bin].get_feature_list() != bin_replacement.copy()):
            feature_bin_list[Bin].set_feature_list(bin_replacement.copy())
            feature_bin_list[Bin].set_not_seen()

    # Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()

    # Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = [0] * len(feature_matrix.index)

    # Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    # The list and dummy data frame will be used for adding later

    # For each feature group/bin, the values of the features in the bin will be summed to create a value for the bin
    # This will be used to create a feature matrix for the bins and a dictionary of binned feature groups

    count = 0
    binned_feature_groups = {}

    for i in range(0, len(feature_bin_list)):
        sum_column = dummy_df['Zeros']
        for j in range(0, len(feature_bin_list[i])):
            sum_column = sum_column + feature_matrix[feature_bin_list[i][j]]
        count = count + 1
        bins_df["Bin " + str(count)] = sum_column
        feature_bin_list[i].set_name("Bin " + str(count))
        binned_feature_groups["Bin " + str(count)] = feature_bin_list[i]

    # Adding the class label to the data frame
    bins_df[label_name] = feature_matrix[label_name]
    bins_df[duration_name] = feature_matrix[duration_name]

    return bins_df, binned_feature_groups


def crossover_and_mutation_old(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups,
                               crossover_probability, mutation_probability, random_seed, threshold,
                               threshold_is_evolving, min_threshold, max_threshold):
    # Creating a list for offspring
    offspring_list = []

    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 8)
    # Creating a number of offspring equal to the number needed to replace the non-elites
    # Each pair of parents will produce two offspring
    for i in range(0, num_replacement_sets):
        # Choosing the two parents and getting the list of features in each parent bin
        parent_bins = tournament_selection_parent_bins(binned_feature_groups, random_seeds[i])
        parent1_features = parent_bins[0].get_feature_list()  # SPHIA
        parent2_features = parent_bins[1].get_feature_list()  # SPHIA

        # Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []

        # Creating two thresholds for the offspring bins
        threshold1 = threshold
        threshold2 = threshold

        # CROSSOVER
        # Each feature in the parent bin will cross over based on the given probability (uniform crossover)
        for j in range(0, len(parent1_features)):
            if crossover_probability > random.random():
                offspring2.append(parent1_features[j])
            else:
                offspring1.append(parent1_features[j])

        for k in range(0, len(parent2_features)):
            if crossover_probability > random.random():
                offspring1.append(parent2_features[k])
            else:
                offspring2.append(parent2_features[k])

        # Ensuring that each of the offspring is no more than twice the size of the other offspring
        while len(offspring1) > len(offspring2):
            switch = random.choice(offspring1)
            offspring1.remove(switch)
            offspring2.append(switch)

        while len(offspring2) > len(offspring1):
            switch = random.choice(offspring2)
            offspring2.remove(switch)
            offspring1.append(switch)
        # check this rule sphia, remove this and turn off merge_probability

        # Crossover the thresholds if threshold is evolving
        if (threshold_is_evolving):
            # The threshold of the parent bin is crossed over to offspring based on the given probability (uniform crossover)
            if crossover_probability > random.random():
                threshold1 = parent_bins[0].get_threshold()
                threshold2 = parent_bins[1].get_threshold()
            else:
                threshold2 = parent_bins[0].get_threshold()
                threshold1 = parent_bins[1].get_threshold()

        # MUTATION

        # Mutation only occurs with a certain probability on each feature in the original feature space

        # Applying the mutation operation to the first offspring

        # Creating a probability for adding a feature that accounts for the ratio between the feature list and the
        # size of the bin
        mutation_addition_prob = None
        if len(offspring1) > 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob = mutation_probability * (len(offspring1)) / \
                                     (len(feature_list) - len(offspring1))
        elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob = mutation_probability
        elif len(offspring1) == len(feature_list):
            mutation_addition_prob = 0

        deleted_list = []
        # Deletion form of mutation
        for idx in range(0, len(offspring1)):
            # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
            if mutation_probability > random.random():
                deleted_list.append(offspring1[idx])

        for idx in range(0, len(deleted_list)):
            offspring1.remove(deleted_list[idx])

        # Creating a list of features outside the offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring1]

        # Addition form of mutation
        for idx in range(0, len(features_not_in_offspring)):
            # Mutation (addition) occurs on this feature with probability proportional to the mutation parameter
            # The probability accounts for the ratio between the feature list and the size of the bin
            if mutation_addition_prob > random.random():
                offspring1.append(features_not_in_offspring[idx])

        # Applying the mutation operation to the second offspring

        # Creating a probability for adding a feature that accounts for the ratio between the feature list and the
        # size of the bin
        if len(offspring2) > 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob = mutation_probability * (len(offspring2)) / \
                                     (len(feature_list) - len(offspring2))
        elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob = mutation_probability
        elif len(offspring2) == len(feature_list):
            mutation_addition_prob = 0

        deleted_list = []

        # update the mutation section with one probability
        # Deletion form of mutation
        for idx in range(0, len(offspring2)):
            # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
            if mutation_probability > random.random():
                deleted_list.append(offspring2[idx])

        for idx in range(0, len(deleted_list)):
            offspring2.remove(deleted_list[idx])

        # Creating a list of features outside the offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring2]

        # Addition form of mutation
        for idx in range(0, len(features_not_in_offspring)):
            # Mutation (addition) occurs on this feature with probability proportional to the mutation parameter
            # The probability accounts for the ratio between the feature list and the size of the bin
            if mutation_addition_prob > random.random():
                offspring2.append(features_not_in_offspring[idx])

        if (threshold_is_evolving):
            # Mutating the threshold for Offspring 1 based on the mutation_probability
            if mutation_probability < random.random():
                threshold1 = np.random.randint(min_threshold, max_threshold + 1)

            # Mutating the threshold for Offspring 2 based on the mutation_probability
            if mutation_probability < random.random():
                threshold2 = np.random.randint(min_threshold, max_threshold + 1)

        # CLEANUP
        # Deleting any repeats of an amino acid in a bin
        # Removing duplicates of features in the same bin that may arise due to crossover
        unique = []
        for a in range(0, len(offspring1)):
            if offspring1[a] not in unique:
                unique.append(offspring1[a])

        # Comment this out, anything that forces a bin size sphia
        # Adding random features from outside the bin to replace the deleted features in the bin
        replace_number = len(offspring1) - len(unique)
        features_not_in_offspring = [item for item in feature_list if item not in offspring1]
        offspring1 = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        offspring1.extend(replacements)

        unique = []
        for a in range(0, len(offspring2)):
            if offspring2[a] not in unique:
                unique.append(offspring2[a])

        # Adding random features from outside the bin to replace the deleted features in the bin
        replace_number = len(offspring2) - len(unique)
        features_not_in_offspring = [item for item in feature_list if item not in offspring2]
        offspring2 = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        offspring2.extend(replacements)

        # Adding the new offspring to the list of feature bins SPHIA
        temp_off_spring_bin1 = BIN(offspring1, threshold1)
        temp_off_spring_bin2 = BIN(offspring2, threshold2)

        offspring_list.append(temp_off_spring_bin1)
        offspring_list.append(temp_off_spring_bin2)

    return offspring_list


def crossover_and_mutation_new(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups,
                               crossover_probability, mutation_probability, random_seed, threshold,
                               threshold_is_evolving, min_threshold, max_threshold):
    # Creating a list for offspring
    offspring_list = []

    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 8)
    # Creating a number of offspring equal to the number needed to replace the non-elites
    # Each pair of parents will produce two offspring
    for i in range(0, num_replacement_sets):
        # Choosing the two parents and getting the list of features in each parent bin
        parent_bins = tournament_selection_parent_bins(binned_feature_groups, random_seeds[i])
        parent1_features = parent_bins[0].get_feature_list()  # SPHIA
        parent2_features = parent_bins[1].get_feature_list()  # SPHIA

        # Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []

        # Creating two thresholds for the offspring bins
        threshold1 = threshold
        threshold2 = threshold

        # CROSSOVER
        # Each feature in the parent bin will cross over based on the given probability (uniform crossover)
        for j in range(0, len(parent1_features)):
            if crossover_probability > random.random():
                offspring2.append(parent1_features[j])
            else:
                offspring1.append(parent1_features[j])

        for k in range(0, len(parent2_features)):
            if crossover_probability > random.random():
                offspring1.append(parent2_features[k])
            else:
                offspring2.append(parent2_features[k])

        # CLEANUP
        # Deleting any repeats of an amino acid in a bin
        # Removing duplicates of features in the same bin that may arise due to crossover

        unique = []
        for a in range(0, len(offspring1)):
            if offspring1[a] not in unique:
                unique.append(offspring1[a])
        offspring1 = unique

        unique = []
        for a in range(0, len(offspring2)):
            if offspring2[a] not in unique:
                unique.append(offspring2[a])
        offspring2 = unique

        # Crossover the thresholds if threshold is evolving
        if (threshold_is_evolving):
            # The threshold of the parent bin is crossed over to offspring based on the given probability (uniform crossover)
            if crossover_probability > random.random():
                threshold1 = parent_bins[0].get_threshold()
                threshold2 = parent_bins[1].get_threshold()
            else:
                threshold2 = parent_bins[0].get_threshold()
                threshold1 = parent_bins[1].get_threshold()

        print("Bin before mutation: " + str(len(offspring1)))
        print(offspring1)
        count_removed = 0
        count_added = 0
        # MUTATION
        # Mutation only occurs with a certain probability on each feature in the original feature space
        # Applying the mutation operation to the first offspring
        for feature in feature_list:
            if (mutation_probability > random.random()):
                # equal chance of either removing the feature or adding a feature
                if (not feature in offspring1):
                    offspring1.append(feature)
                    count_added += 1
                else:
                    offspring1.remove(feature)
                    count_removed += 1

        print("Bin after mutation: " + str(len(offspring1)))
        print(offspring1)
        print("Features added: " + str(count_added))
        print("Features removed: " + str(count_removed))

        # Applying mutation to the second offspring        
        for feature in feature_list:
            if (mutation_probability > random.random()):
                # equal chance of either removing the feature or adding a feature
                if (not feature in offspring2):
                    offspring2.append(feature)
                else:
                    offspring2.remove(feature)

        # EVOLVING THRESHOLD
        if (threshold_is_evolving):
            # Mutating the threshold for Offspring 1 based on the mutation_probability
            if mutation_probability < random.random():
                threshold1 = np.random.randint(min_threshold, max_threshold + 1)

            # Mutating the threshold for Offspring 2 based on the mutation_probability
            if mutation_probability < random.random():
                threshold2 = np.random.randint(min_threshold, max_threshold + 1)

        # Adding the new offspring to the list of feature bins SPHIA
        temp_off_spring_bin1 = BIN(offspring1, threshold1)
        temp_off_spring_bin2 = BIN(offspring2, threshold2)

        offspring_list.append(temp_off_spring_bin1)
        offspring_list.append(temp_off_spring_bin2)

    return offspring_list


def crossover_and_mutation_new_previous(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups,
                                        crossover_probability, mutation_probability, random_seed, threshold,
                                        threshold_is_evolving, min_threshold, max_threshold):
    # Creating a list for offspring
    offspring_list = []

    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 8)
    # Creating a number of offspring equal to the number needed to replace the non-elites
    # Each pair of parents will produce two offspring
    for i in range(0, num_replacement_sets):
        # Choosing the two parents and getting the list of features in each parent bin
        parent_bins = tournament_selection_parent_bins(binned_feature_groups, random_seeds[i])
        parent1_features = parent_bins[0].get_feature_list()
        parent2_features = parent_bins[1].get_feature_list()

        # Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []

        # Creating two thresholds for the offspring bins
        threshold1 = threshold
        threshold2 = threshold

        # CROSSOVER
        # Each feature in the parent bin will cross over based on the given probability (uniform crossover)
        for j in range(0, len(parent1_features)):
            if crossover_probability > random.random():
                offspring2.append(parent1_features[j])
            else:
                offspring1.append(parent1_features[j])

        for k in range(0, len(parent2_features)):
            if crossover_probability > random.random():
                offspring1.append(parent2_features[k])
            else:
                offspring2.append(parent2_features[k])

        # CLEANUP
        # Deleting any repeats of an amino acid in a bin
        # Removing duplicates of features in the same bin that may arise due to crossover

        unique = []
        for a in range(0, len(offspring1)):
            if offspring1[a] not in unique:
                unique.append(offspring1[a])
        offspring1 = unique

        features_not_in_offspring1 = [item for item in feature_list if item not in offspring1]
        random.shuffle(features_not_in_offspring1)
        idx1 = 0

        unique = []
        for a in range(0, len(offspring2)):
            if offspring2[a] not in unique:
                unique.append(offspring2[a])
        offspring2 = unique

        features_not_in_offspring2 = [item for item in feature_list if item not in offspring2]
        random.shuffle(features_not_in_offspring2)
        idx2 = 0

        # Crossover the thresholds if threshold is evolving
        if (threshold_is_evolving):
            # The threshold of the parent bin is crossed over to offspring based on the given probability (uniform crossover)
            if crossover_probability > random.random():
                threshold1 = parent_bins[0].get_threshold()
                threshold2 = parent_bins[1].get_threshold()
            else:
                threshold2 = parent_bins[0].get_threshold()
                threshold1 = parent_bins[1].get_threshold()

        # MUTATION
        # Mutation only occurs with a certain probability on each feature in the original feature space
        # Applying the mutation operation to the first offspring
        new_offspring1 = []
        for feature in offspring1:
            if (mutation_probability > random.random()):
                # equal chance of either removing the feature or adding a feature
                if (0.5 > random.random() and len(offspring1) < len(feature_list)):
                    new_offspring1.append(features_not_in_offspring1[idx1])
                    new_offspring1.append(feature)
                    idx1 += 1
            else:
                new_offspring1.append(feature)

        offspring1 = new_offspring1

        # Applying mutation to the second offspring        
        new_offspring2 = []
        for feature in offspring2:
            if (mutation_probability > random.random()):
                # Equal chance of either removing the feature or adding a feature
                if (0.5 > random.random() and len(offspring2) < len(feature_list)):
                    new_offspring2.append(features_not_in_offspring2[idx2])
                    new_offspring2.append(feature)
                    idx2 += 1
            else:
                new_offspring2.append(feature)

        offspring2 = new_offspring2

        # EVOLVING THRESHOLD
        if (threshold_is_evolving):
            # Mutating the threshold for Offspring 1 based on the mutation_probability
            if mutation_probability < random.random():
                threshold1 = np.random.randint(min_threshold, max_threshold + 1)

            # Mutating the threshold for Offspring 2 based on the mutation_probability
            if mutation_probability < random.random():
                threshold2 = np.random.randint(min_threshold, max_threshold + 1)

        # Adding the new offspring to the list of feature bins SPHIA
        temp_off_spring_bin1 = BIN(offspring1, threshold1)
        temp_off_spring_bin2 = BIN(offspring2, threshold2)

        offspring_list.append(temp_off_spring_bin1)
        offspring_list.append(temp_off_spring_bin2)

    return offspring_list


def fibers_algorithm(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
                     original_feature_matrix, label_name, duration_name,
                     set_number_of_bins, min_features_per_group, max_number_of_groups_with_feature,
                     informative_cutoff,
                     crossover_probability, mutation_probability, elitism_parameter, random_seed,
                     set_threshold, evolving_probability, max_threshold, min_threshold,
                     merge_probability, adaptable_threshold):  # SPHIA

    # Step 0: Deleting Empty Features (MAF = 0)
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list = remove_empty_variables(
        original_feature_matrix,
        label_name, duration_name)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups

    amino_acids, amino_acid_bins = None, None
    # If there is a starting point, use that for the amino acid list and the amino acid bins list
    # SPHIA FIX
    if given_starting_point:
        # Keep only MAF != 0 features from starting points in amino_acids and amino_acid_bins
        # amino_acids = list(set(amino_acid_start_point).intersection(nonempty_feature_list))

        # Original
        amino_acids = amino_acid_start_point.copy()
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
                                                               max_features_per_bin=None,
                                                               threshold=set_threshold)

    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, duration_name,
                                                amino_acid_bins)

    # Step 1b: Initializing all thresholds
    if (adaptable_threshold):
        # To initialize, tries all thresholds to find the best one
        for bin in amino_acid_bins.values():
            bin.try_all_thresholds(
                min_threshold, max_threshold, bin_feature_matrix,
                label_name, duration_name, informative_cutoff)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    np.random.seed(random_seed)
    upper_bound = (len(maf_0_features) + len(nonempty_feature_list)) * (
            len(maf_0_features) + len(nonempty_feature_list))
    random_seeds = np.random.randint(upper_bound, size=iterations * 2)

    # current
    previous_bin = amino_acid_bins['Bin 1']

    # starting time to calculate time for each iteration
    start_time = time.time()

    time_list = []
    score_list = []

    threshold_is_evolving = False

    stop_time = 0

    random.seed(random_seed)
    for i in tqdm(range(0, iterations)):

        # variable to keep track of whether the iteration evolves or tries all thresholds
        evolve = random.random()

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # If the try_all_thresholds is set to True, then their log_rank_scores have already been evaluated
        log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name,
                                         duration_name, informative_cutoff)

        # random.seed(random_seed)
        # #merging the top two bins based on the merge_probability
        # if merge_probability > random.random():
        #     sorted_bins = sorted(list(amino_acid_bins.values()), reverse = True)
        #     merged_feature_list = list(set(sorted_bins[0].get_feature_list() + sorted_bins[1].get_feature_list()))
        #     merged_bin = sorted_bins[set_number_of_bins - 1]
        #     merged_bin.set_feature_list(merged_feature_list)
        #     merged_bin.set_not_seen()
        #     amino_acid_bins['Bin 50'] = merged_bin
        #     if(previous_merged_bin != merged_bin):
        #         previous_merged_bin = merged_bin
        #         print(merged_bin)

        # Given a evolving probability, there is a chance
        # It doesn't try all thresholds but instead evolves the threshold
        if evolving_probability > evolve and adaptable_threshold:
            threshold_is_evolving = True
        else:
            threshold_is_evolving = False

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = crossover_and_mutation_new_previous(set_number_of_bins, elitism_parameter, amino_acids,
                                                             amino_acid_bins,
                                                             crossover_probability, mutation_probability,
                                                             random_seeds[i], set_threshold,
                                                             threshold_is_evolving, min_threshold,
                                                             max_threshold)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = create_next_generation(amino_acid_bins, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        bin_feature_matrix, amino_acid_bins = regroup_feature_matrix(amino_acids, original_feature_matrix, label_name,
                                                                     duration_name, feature_bin_list,
                                                                     random_seeds[iterations + i], set_threshold)

        # If the adaptable_threshold boolean is set true by the user and threshold did not evolve,
        # then the offspring BIN objects will have their
        # threshold changed based on highest log rank score
        if not threshold_is_evolving and adaptable_threshold:
            for bin in amino_acid_bins.values():
                bin.try_all_thresholds(min_threshold, max_threshold, bin_feature_matrix,
                                       label_name, duration_name, informative_cutoff)

        stop_time = time.time()

        current_bin = amino_acid_bins['Bin 1']

        time_list.append(stop_time)
        score_list.append(current_bin.get_score())

        if (current_bin != previous_bin):
            print("Time took: " + str((stop_time - start_time) // 60) + " Minutes and " + str(
                (stop_time - start_time) % 60) +
                  " Seconds")
            print("Bin Change Iteration: " + str(i))
            print(current_bin)
            print("Score: " + str(current_bin.get_score()))
            previous_bin = current_bin

        if ((stop_time - start_time) // 60 >= 60):
            break

    plt.scatter(time_list, score_list)
    plt.show()

    # calculating the last thresholds
    # Creating the final amino acid bin scores
    amino_acid_bin_scores = log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name,
                                                             duration_name, informative_cutoff)

    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, maf_0_features
