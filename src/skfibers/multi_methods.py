import os
import random
import numpy as np
import pandas as pd
import statistics
from random import randrange
from skrebate import MultiSURF
from sklearn.feature_selection import chi2
from multiprocessing import Pool


# Step 1: Initialize Population of Candidate Bins

# Random initialization of candidate bins, which are groupings of multiple features
# The value of each bin/feature is the sum of values for each feature in the bin
# Adding a function that can be an option to automatically separate rare and common variables

# Defining a function to delete variables with MAF = 0
def remove_empty_variables(original_feature_matrix, label_name):
    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name])

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
    feature_matrix_no_empty_variables['Class'] = original_feature_matrix[label_name]

    return feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list


# Defining a function to group features randomly, each feature can be in a number of groups up to a set max

def random_feature_grouping(feature_matrix, label_name, number_of_groups, min_features_per_group,
                            max_number_of_groups_with_feature, random_seed, max_features_per_bin):
    # Removing the label column to create a list of features
    feature_df = feature_matrix.drop(columns=[label_name])

    # Creating a list of features
    feature_list = list(feature_df.columns)

    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=len(feature_list))
    # Adding a random number of repeats of the features so that features can be in more than one group
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
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list),
                                     size=(len(feature_list) - min_features_per_group * number_of_groups))
    for y in range(min_features_per_group * number_of_groups, len(feature_list)):
        random.seed(random_seeds[y - min_features_per_group * number_of_groups])
        feature_groups[random.choice(list(feature_groups.keys()))].append(feature_list[y])

    # Removing duplicates of features in the same bin
    for z in range(0, len(feature_groups)):
        feature_groups[z] = list(set(feature_groups[z]))

        # Randomly removing features until the number of features is equal to or less than the max_features_per_bin 
        # param
        if not (max_features_per_bin is None):
            if len(feature_groups[z]) > max_features_per_bin:
                random.seed(random_seeds[z])
                feature_groups[z] = list(random.sample(feature_groups[z], max_features_per_bin))

    # Creating a dictionary with bin labels
    binned_feature_groups = {}
    for index in range(0, len(feature_groups)):
        binned_feature_groups["Bin " + str(index + 1)] = feature_groups[index]

    return feature_list, binned_feature_groups


# Defining a function to create a feature matrix where each feature is a bin of features from the original feature
# matrix

def grouped_feature_matrix(feature_matrix, label_name, binned_feature_groups):
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
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df


# Defining a function to separate rare and common variables based on a rare variant minor allele frequency (MAF) cutoff
def rare_and_common_variable_separation(original_feature_matrix, label_name, rare_variant_maf_cutoff):
    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name])

    # Calculating the MAF of each feature
    maf = list(feature_df.sum() / (2 * len(feature_df.index)))

    # Creating a df of features and their MAFs
    feature_maf_df = pd.DataFrame(feature_df.columns, columns=['feature'])
    feature_maf_df['maf'] = maf

    # If the MAF of the feature is less than the cutoff, it will be designated as a rare variant
    # If the MAF of the feature is greater than or equal to the cutoff, it will be considered as a common feature
    rare_df = feature_maf_df.loc[(feature_maf_df['maf'] < rare_variant_maf_cutoff) & (feature_maf_df['maf'] > 0)]
    common_df = feature_maf_df.loc[feature_maf_df['maf'] > rare_variant_maf_cutoff]
    maf_0_df = feature_maf_df.loc[feature_maf_df['maf'] == 0]

    # Creating lists of rare and common features
    rare_feature_list = list(rare_df['feature'])
    common_feature_list = list(common_df['feature'])
    maf_0_features = list(maf_0_df['feature'])

    # Creating dictionaries of rare and common features, as the MAF of the features will be useful later
    rare_feature_maf_dict = dict(zip(rare_df['feature'], rare_df['maf']))
    common_feature_maf_dict = dict(zip(common_df['feature'], common_df['maf']))

    # Creating data frames for feature matrices of rare features and common features
    rare_feature_df = feature_df[rare_feature_list]
    common_feature_df = feature_df[common_feature_list]

    # Adding the class label to each data frame
    rare_feature_df['Class'] = original_feature_matrix[label_name]
    common_feature_df['Class'] = original_feature_matrix[label_name]
    return rare_feature_list, rare_feature_maf_dict, rare_feature_df, \
        common_feature_list, common_feature_maf_dict, common_feature_df, maf_0_features


# Step 2: Genetic Algorithm with Relief-based Feature Scoring (repeated for a given number of iterations)


# Step 2a: Relief-based Feature Importance Scoring and Bin Deletion

# Use MultiSURF to calculate the feature importance of each candidate bin
# If the population size > the set max population size then bins will be probabilistically deleted based on fitness


# Defining a function to calculate the feature importance of each bin using MultiSURF
def multi_surf_feature_importance(bin_feature_matrix, label_name):
    # Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = bin_feature_matrix.astype(float)

    # Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                           fs.feature_importances_):
        feature_scores[feature_name] = feature_score

    return feature_scores


# Defining a function to calculate MultiSURF feature importance based on a sample of instances
def multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name, sample_size):
    # Taking a random sample of the instances based on the sample size paramter to calculate MultiSURF
    bin_feature_matrix_sample = bin_feature_matrix.sample(sample_size)

    # Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = bin_feature_matrix_sample.astype(float)

    # Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                           fs.feature_importances_):
        feature_scores[feature_name] = feature_score

    return feature_scores


# Defining a function to calculate the feature importance of each bin using MultiSURF for rare variants in context
# with common variables
def multi_surf_feature_importance_rare_variants(bin_feature_matrix, common_feature_list, common_feature_matrix,
                                                label_name):
    # Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()

    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]

    # Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = common_features_and_bins_matrix.astype(float)

    # Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                           fs.feature_importances_):
        feature_scores[feature_name] = feature_score

    for i in range(0, len(common_feature_list)):
        del feature_scores[common_feature_list[i]]

    return feature_scores


# Defining a function to calculate MultiSURF feature importance for rare variants in context with common variables
# based on a sample of instances
def multi_surf_feature_importance_rare_variants_instance_sample(bin_feature_matrix, common_feature_list,
                                                                common_feature_matrix,
                                                                label_name, sample_size):
    # Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()

    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]

    # Taking a random sample of the instances based on the sample size paramter to calculate MultiSURF
    common_features_and_bins_matrix_sample = common_features_and_bins_matrix.sample(sample_size)

    # Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = common_features_and_bins_matrix_sample.astype(float)

    # Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                           fs.feature_importances_):
        feature_scores[feature_name] = feature_score

    for i in range(0, len(common_feature_list)):
        del feature_scores[common_feature_list[i]]

    return feature_scores


# Defining a function to calculate MultiSURF feature importance only considering the bin and common feature(s)
def multi_surf_feature_importance_bin_and_common_features(bin_feature_matrix, amino_acid_bins, common_feature_list,
                                                          common_feature_matrix, label_name):
    # Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()

    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]

    bin_scores = {}
    for i in amino_acid_bins.keys():
        # Only taking the bin and the common features for the feature importance calculation
        bin_and_common_features = list()
        bin_and_common_features.append(i)
        bin_and_common_features.extend(common_feature_list)
        bin_and_common_features.append(label_name)
        bin_and_cf_df = common_features_and_bins_matrix[bin_and_common_features]
        float_feature_matrix = bin_and_cf_df.astype(float)
        features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
        fs = MultiSURF()
        fs.fit(features, labels)
        feature_scores = {}
        for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                               fs.feature_importances_):
            feature_scores[feature_name] = feature_score
        bin_scores[i] = feature_scores[i]

    return bin_scores


# Defining a function to calculate MultiSURF feature importance only considering the bin and common feature(s)
def multi_surf_feature_importance_bin_and_common_features_instance_sample(bin_feature_matrix, amino_acid_bins,
                                                                          common_feature_list, common_feature_matrix,
                                                                          label_name, sample_size):
    # Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()

    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]

    bin_scores = {}
    for i in amino_acid_bins.keys():
        # Only taking the bin and the common features for the feature importance calculation
        bin_and_common_features = list()
        bin_and_common_features.append(i)
        bin_and_common_features.extend(common_feature_list)
        bin_and_common_features.append(label_name)
        bin_and_cf_df = common_features_and_bins_matrix[bin_and_common_features]

        # Taking a sample to run MultiSURF on
        bin_and_cf_df_sample = bin_and_cf_df.sample(sample_size)
        float_feature_matrix = bin_and_cf_df_sample.astype(float)
        features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
        fs = MultiSURF()
        fs.fit(features, labels)
        feature_scores = {}
        for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                               fs.feature_importances_):
            feature_scores[feature_name] = feature_score
        bin_scores[i] = feature_scores[i]

    return bin_scores


# Defining a function to score bins based on chi squared value
def chi_square_feature_importance(bin_feature_matrix, label_name, amino_acid_bins):
    # Calculating the chisquare and p values of each of the bin features in the bin feature matrix
    x = bin_feature_matrix.drop(label_name, axis=1)
    y = bin_feature_matrix[label_name]
    chi_scores, p_values = chi2(x, y)

    # Creating a dictionary with each bin and the chi-square value and p-value
    bin_names_list = list(amino_acid_bins.keys())
    bin_scores = dict(zip(bin_names_list, chi_scores))

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0

    return bin_scores


# Step 2b: Genetic Algorithm

# Parent bins are probabilistically selected based on fitness (score calculated in Step 2a)
# New offspring bins will be created through cross over and mutation and are added to the next generation's population
# Based on the value of the elitism parameter, a number of high scoring parent bins will be preserved for the next gen


# Defining a function to probabilistically select 2 parent bins based on their feature importance rank
# Tournament Selection works in this case by choosing a random sample of the bins and choosing the best two scores
def tournament_selection_parent_bins(bin_scores, random_seed):
    random.seed(random_seed)

    # Choosing a random sample of 5% of the bin population or if that would be too small, choosing a sample of 50%
    if round(0.05 * len(bin_scores)) < 2:
        samplekeys = random.sample(bin_scores.keys(), round(0.5 * len(bin_scores)))
    else:
        samplekeys = random.sample(bin_scores.keys(), round(0.05 * len(bin_scores)))

    sample = {}
    for key in samplekeys:
        sample[key] = bin_scores[key]

    # Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(sample.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())

    # Choosing the parent bins and adding them to a list of parent bins
    parent_bins = [sorted_bin_list[0], sorted_bin_list[1]]

    return parent_bins


# Defining a function to perform crossover and mutation using multiple cores at once
def crossover_and_mutation_multiprocess(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups,
                                        bin_scores,
                                        crossover_probability, mutation_probability, random_seed,
                                        bin_size_variability_constraint,
                                        max_features_per_bin):
    # Determining the number of offspring created
    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 8)
    arg_of_func = [(feature_list, binned_feature_groups, bin_scores,
                    crossover_probability, mutation_probability,
                    bin_size_variability_constraint, list([]), max_features_per_bin)] * num_replacement_sets
    arg_of_func = list(arg_of_func)

    # Create sets of random seeds for generating offspring
    for i in range(num_replacement_sets):
        random_seeds_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for seed in range(10):
            random_seeds_array[seed] = random_seeds[seed * num_replacement_sets + i]
        arg_list = list(arg_of_func[i])
        arg_list[6] = list(random_seeds_array)
        arg_of_func[i] = tuple(arg_list)

    p = Pool(processes=(os.cpu_count()))
    offspring_list = p.starmap(crossover_and_mutation_for_pool, arg_of_func)
    p.close()

    # formatting offspring list 
    reformatted = []
    for i in range(len(offspring_list)):
        reformatted.extend(offspring_list[i])

    return reformatted


# Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents
# n is the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next
# time)
def crossover_and_mutation_for_pool(feature_list, binned_feature_groups, bin_scores,
                                    crossover_probability, mutation_probability,
                                    bin_size_variability_constraint, random_seeds,
                                    max_features_per_bin):
    feature_list = list(set(feature_list))

    # Creating a list for offspring
    offspring_list = []

    # Creating a number of offspring equal to the number needed to replace the non-elites
    # Each pair of parents will produce two offspring
    # Choosing the two parents and getting the list of features in each parent bin
    parent_bins = tournament_selection_parent_bins(bin_scores, random_seeds[0])
    parent1_features = binned_feature_groups[parent_bins[0]].copy()
    parent2_features = binned_feature_groups[parent_bins[1]].copy()

    # Creating two lists for the offspring bins
    offspring1 = []
    offspring2 = []

    # CROSSOVER
    # Each feature in the parent bin will cross over based on the given probability (uniform crossover)

    # Creating two df for parent features and probability of crossover
    np.random.seed(random_seeds[1])
    randnums1 = list(np.random.randint(0, 101, len(parent1_features)))
    crossover_threshold1 = list([crossover_probability * 100] * len(parent1_features))
    parent1_df = pd.DataFrame(parent1_features, columns=['Features'])
    parent1_df['Threshold'] = crossover_threshold1
    parent1_df['Rand_prob'] = randnums1

    np.random.seed(random_seeds[2])
    randnums2 = list(np.random.randint(0, 101, len(parent2_features)))
    crossover_threshold2 = list([crossover_probability * 100] * len(parent2_features))
    parent2_df = pd.DataFrame(parent2_features, columns=['Features'])
    parent2_df['Threshold'] = crossover_threshold2
    parent2_df['Rand_prob'] = randnums2

    # Features with random probability less than the crossover probability will go to offspring 1.
    # The rest will go to offspring 2.
    offspring1.extend(list(parent1_df.loc[parent1_df['Threshold'] > parent1_df['Rand_prob']]['Features']))
    offspring2.extend(list(parent1_df.loc[parent1_df['Threshold'] <= parent1_df['Rand_prob']]['Features']))
    offspring2.extend(list(parent2_df.loc[parent2_df['Threshold'] > parent2_df['Rand_prob']]['Features']))
    offspring1.extend(list(parent2_df.loc[parent2_df['Threshold'] <= parent2_df['Rand_prob']]['Features']))

    # Remove repeated features within each offspring
    offspring1 = list(set(offspring1))
    offspring2 = list(set(offspring2))

    # MUTATION

    # Mutation (deletion and addition) only occurs with a certain probability on each feature in the original feature
    # space

    # Creating a probability for mutation (addition) that accounts for the ratio between the feature list and the
    # size of the bin
    if len(offspring1) > 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob1 = mutation_probability * (len(offspring1)) / (len(feature_list) - len(offspring1))
    elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob1 = mutation_probability
    elif len(offspring1) == len(feature_list):
        mutation_addition_prob1 = 0

    if len(offspring2) > 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob2 = mutation_probability * (len(offspring2)) / (len(feature_list) - len(offspring2))
    elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob2 = mutation_probability
    elif len(offspring2) == len(feature_list):
        mutation_addition_prob2 = 0

    # Mutation: Deletion occurs on features with probability equal to the mutation parameter
    offspring1_df = pd.DataFrame(offspring1, columns=['Features'])
    mutation_threshold1 = list([mutation_probability * 100] * len(offspring1))
    np.random.seed(random_seeds[3])
    rand1 = list(np.random.randint(0, 101, len(offspring1)))
    offspring1_df['Threshold'] = mutation_threshold1
    offspring1_df['Rand_prob'] = rand1

    offspring2_df = pd.DataFrame(offspring2, columns=['Features'])
    mutation_threshold2 = list([mutation_probability * 100] * len(offspring2))
    np.random.seed(random_seeds[4])
    rand2 = list(np.random.randint(0, 101, len(offspring2)))
    offspring2_df['Threshold'] = mutation_threshold2
    offspring2_df['Rand_prob'] = rand2

    offspring1_df = offspring1_df.loc[offspring1_df['Threshold'] < offspring1_df['Rand_prob']]
    offspring1 = list(offspring1_df['Features'])

    offspring2_df = offspring2_df.loc[offspring2_df['Threshold'] < offspring2_df['Rand_prob']]
    offspring2 = list(offspring2_df['Features'])

    # Mutation: Addition occurs on this feature with probability proportional to the mutation parameter
    # The probability accounts for the ratio between the feature list and the size of the bin

    features_not_in_offspring1 = [item for item in feature_list if item not in offspring1]
    features_not_in_offspring2 = [item for item in feature_list if item not in offspring2]

    features_not_in_offspring1_df = pd.DataFrame(features_not_in_offspring1, columns=['Features'])
    mutation_addition_threshold1 = list([mutation_addition_prob1 * 100] * len(features_not_in_offspring1_df))
    np.random.seed(random_seeds[5])
    rand1 = list(np.random.randint(0, 101, len(features_not_in_offspring1)))
    features_not_in_offspring1_df['Threshold'] = mutation_addition_threshold1
    features_not_in_offspring1_df['Rand_prob'] = rand1

    features_not_in_offspring2_df = pd.DataFrame(features_not_in_offspring2, columns=['Features'])
    mutation_addition_threshold2 = list([mutation_addition_prob2 * 100] * len(features_not_in_offspring2_df))
    np.random.seed(random_seeds[6])
    rand2 = list(np.random.randint(0, 101, len(features_not_in_offspring2)))
    features_not_in_offspring2_df['Threshold'] = mutation_addition_threshold2
    features_not_in_offspring2_df['Rand_prob'] = rand2

    features_to_add1 = list(features_not_in_offspring1_df.loc[
                                features_not_in_offspring1_df['Threshold'] >= features_not_in_offspring1_df[
                                    'Rand_prob']]['Features'])
    features_to_add2 = list(features_not_in_offspring2_df.loc[
                                features_not_in_offspring2_df['Threshold'] >= features_not_in_offspring2_df[
                                    'Rand_prob']]['Features'])

    offspring1.extend(features_to_add1)
    offspring2.extend(features_to_add2)

    # Ensuring that each of the offspring is no more than c times the size of the other offspring
    if not (bin_size_variability_constraint is None):
        c_constraint = bin_size_variability_constraint
        np.random.seed(random_seeds[7])
        random_seeds_loop = np.random.randint(len(feature_list) * len(feature_list), size=2 * len(feature_list))
        counter = 0
        while counter < 2 * len(feature_list) and len(offspring1) > c_constraint * len(offspring2) or len(
                offspring2) > c_constraint * len(offspring1):
            np.random.seed(random_seeds_loop[counter])
            random.seed(random_seeds_loop[counter])

            if len(offspring1) > c_constraint * len(offspring2):
                min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                min_to_move = min_features - len(offspring2)
                max_to_move = len(offspring1) - min_features
                num_to_move = np.random.randint(min_to_move, max_to_move + 1)
                features_to_move = list(random.sample(offspring1, num_to_move))
                offspring1 = [x for x in offspring1 if x not in features_to_move]
                offspring2.extend(features_to_move)
            elif len(offspring2) > c_constraint * len(offspring1):
                min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                min_to_move = min_features - len(offspring1)
                max_to_move = len(offspring2) - min_features
                num_to_move = np.random.randint(min_to_move, max_to_move + 1)
                features_to_move = random.sample(offspring2, num_to_move)
                offspring2 = [x for x in offspring2 if x not in features_to_move]
                offspring1.extend(features_to_move)
            offspring1 = list(set(offspring1))
            offspring2 = list(set(offspring2))
            counter = counter + 1

    # Ensuring the size of the offspring is not greater than the max_features_per_bin allowed
    if not (max_features_per_bin is None):
        if len(offspring1) > max_features_per_bin:
            random.seed(random_seeds[8])
            offspring1 = list(random.sample(offspring1, max_features_per_bin))
        if len(offspring2) > max_features_per_bin:
            random.seed(random_seeds[9])
            offspring2 = list(random.sample(offspring2, max_features_per_bin))

    # Adding the new offspring to the list of feature bins
    offspring_list.append(offspring1)
    offspring_list.append(offspring2)

    return offspring_list


def create_next_generation(binned_feature_groups, bin_scores, max_population_of_bins, elitism_parameter,
                           offspring_list):
    # Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())

    # Determining the number of elite bins
    number_of_elite_bins = round(max_population_of_bins * elitism_parameter)
    elite_bin_list = sorted_bin_list[0:number_of_elite_bins]

    # Adding the elites to a list of elite feature bins
    elite_dict = {k: v for k, v in binned_feature_groups.items() if k in elite_bin_list}

    # Creating a list of feature bins (without labels because those will be changed as things get deleted and added)
    feature_bin_list = list(elite_dict.values())

    # Adding the offspring to the feature bin list
    feature_bin_list.extend(offspring_list)
    return feature_bin_list


# Defining a function to recreate the feature matrix (add up values of amino a cids from original dataset)
def regroup_feature_matrix(feature_list, feature_matrix, label_name, feature_bin_list, random_seed):
    # First deleting any bins that are empty
    bins_deleted = [x for x in feature_bin_list if x == []]
    feature_bin_list = [x for x in feature_bin_list if x != []]

    # Checking each pair of bins, if the bins are duplicates then one of the copies will be deleted
    no_duplicates = []
    num_duplicates = 0
    for bin_var in feature_bin_list:
        if bin_var not in no_duplicates:
            no_duplicates.append(bin_var)
        else:
            num_duplicates += 1

    feature_bin_list = no_duplicates

    # Calculate average length of nonempty bins in the population
    bin_lengths = [len(x) for x in feature_bin_list if len(x) != 0]
    replacement_length = round(statistics.mean(bin_lengths))

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
        while replacement in feature_bin_list:
            random.seed(random_seeds_replacement[counter])
            replacement = random.sample(feature_list, replacement_length)
            counter = counter + 1

        feature_bin_list.append(replacement)

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
        binned_feature_groups["Bin " + str(count)] = feature_bin_list[i]

    # Adding the class label to the data frame
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df, binned_feature_groups


# Defining a function for the RAFE algorithm (Relief-based Association Feature-bin Evolver)
# Same as RARE, but it bins all features not just rare features
def rafe_algorithm(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
                   original_feature_matrix,
                   label_name, set_number_of_bins, min_features_per_group, max_number_of_groups_with_feature,
                   scoring_method, score_based_on_sample, instance_sample_size,
                   crossover_probability, mutation_probability, elitism_parameter, random_seed,
                   bin_size_variability_constraint,
                   max_features_per_bin):
    # Step 0: Deleting Empty Features (MAF = 0)
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list = remove_empty_variables(
        original_feature_matrix, label_name)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups

    # If there is a starting point, use that for the amino acid list and the amino acid bins list
    if given_starting_point is True:
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
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature, random_seed,
                                                               max_features_per_bin)

    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, amino_acid_bins)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    np.random.seed(random_seed)
    upper_bound = (len(maf_0_features) + len(nonempty_feature_list)) * (
            len(maf_0_features) + len(nonempty_feature_list))
    random_seeds = np.random.randint(upper_bound, size=iterations * 2)
    for i in range(0, iterations):

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # Feature scoring can be done with Relief or a univariate chi squared test
        if scoring_method == 'Relief':
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                      instance_sample_size)
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, 'Class', amino_acid_bins)

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = crossover_and_mutation_multiprocess(set_number_of_bins, elitism_parameter, amino_acids,
                                                             amino_acid_bins, amino_acid_bin_scores,
                                                             crossover_probability, mutation_probability,
                                                             random_seeds[i], bin_size_variability_constraint,
                                                             max_features_per_bin)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = create_next_generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        bin_feature_matrix, amino_acid_bins = regroup_feature_matrix(amino_acids, original_feature_matrix, label_name,
                                                                     feature_bin_list, random_seeds[iterations + i])

    # Creating the final bin scores
    # Feature scoring can be done with Relief or a univariate chi squared test
    if scoring_method == 'Relief':
        # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if not score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
        elif score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                  instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, label_name, amino_acid_bins)

    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, maf_0_features


# Defining a function to present the top bins
def top_bins_summary(original_feature_matrix, label_name, bin_feature_matrix, bins, bin_scores, number_of_top_bins):
    # Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    # Calculating the chi square and p values of each of the features in the original feature matrix
    df = original_feature_matrix
    x = df.drop('Class', axis=1)
    y = df['Class']
    chi_scores, p_values = chi2(x, y)

    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name])

    # Creating a list of features
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))

    # Creating a dictionary with each feature and the chi-square value and p-value
    univariate_feature_stats = {}
    for i in range(0, len(feature_list)):
        list_of_stats = list()
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        univariate_feature_stats[feature_list[i]] = list_of_stats
        # There will be features with nan for their chi-square value and p-value because the whole column is zeroes

    # Calculating the chi-square and p values of each of the features in the bin feature matrix
    x = bin_feature_matrix.drop('Class', axis=1)
    y = bin_feature_matrix['Class']
    chi_scores, p_values = chi2(x, y)

    # Creating a dictionary with each bin and the chi-square value and p-value
    bin_stats = {}
    bin_names_list = list(bins.keys())
    for i in range(0, len(bin_names_list)):
        list_of_stats = list()
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        bin_stats[bin_names_list[i]] = list_of_stats

    for i in range(0, number_of_top_bins):
        # Printing the bin Name
        print("Bin Rank " + str(i + 1) + ": " + sorted_bin_list[i])
        # Printing the bin's MultiSURF/Univariate score, chi-square value, and p-value
        print("MultiSURF or Univariate Score: " + str(
            sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(
            bin_stats[sorted_bin_list[i]][0]) + "; p-value: " + str(bin_stats[sorted_bin_list[i]][1]))
        # Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range(0, len(bins[sorted_bin_list[i]])):
            print("Feature Name: " + bins[sorted_bin_list[i]][j] + "; chi-square value: " + str(
                univariate_feature_stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(
                univariate_feature_stats[bins[sorted_bin_list[i]][j]][1]))
        print('---------------------------')


# Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
def rare_algorithm_v2(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
                      original_feature_matrix,
                      label_name, rare_variant_maf_cutoff, set_number_of_bins,
                      min_features_per_group, max_number_of_groups_with_feature,
                      scoring_method, score_based_on_sample, score_with_common_variables,
                      instance_sample_size, crossover_probability, mutation_probability, elitism_parameter, random_seed,
                      bin_size_variability_constraint, max_features_per_bin):
    # Step 0: Separate Rare Variants and Common Features
    rare_feature_list, rare_feature_maf_dict, \
        rare_feature_df, common_feature_list, \
        common_feature_maf_dict, common_feature_df, \
        maf_0_features = rare_and_common_variable_separation(original_feature_matrix,
                                                             label_name, rare_variant_maf_cutoff)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups
    if given_starting_point:
        # Keep only rare features from starting points in amino_acids and amino_acid_bins
        amino_acids = list(set(amino_acid_start_point).intersection(rare_feature_list))

        amino_acid_bins = amino_acid_bins_start_point.copy()
        bin_names = amino_acid_bins.keys()
        features_to_remove = [item for item in amino_acid_start_point if item not in rare_feature_list]
        for bin_name in bin_names:
            # Remove duplicate features
            amino_acid_bins[bin_name] = list(set(amino_acid_bins[bin_name]))
            for feature in features_to_remove:
                if feature in amino_acid_bins[bin_name]:
                    # Remove features not in rare_feature_list
                    amino_acid_bins[bin_name].remove(feature)

    # Otherwise randomly initialize the bins
    elif not given_starting_point:
        amino_acids, amino_acid_bins = random_feature_grouping(rare_feature_df, label_name,
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature, random_seed,
                                                               max_features_per_bin)
    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(rare_feature_df, label_name, amino_acid_bins)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    np.random.seed(random_seed)
    upper_bound = (len(rare_feature_list) + len(common_feature_list) + len(maf_0_features)) * (
            len(rare_feature_list) + len(common_feature_list) + len(maf_0_features))
    random_seeds = np.random.randint(upper_bound, size=iterations * 2)
    for i in range(0, iterations):
        print("iteration:" + str(i))

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # Feature importance can be scored with Relief or a univariate metric (chi squared value)
        if scoring_method == 'Relief':
            # Feature importance is calculating either with common variables
            if score_with_common_variables:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if not score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_rare_variants(bin_feature_matrix,
                                                                                        common_feature_list,
                                                                                        common_feature_df, label_name)
                elif score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_rare_variants_instance_sample(
                        bin_feature_matrix,
                        common_feature_list,
                        common_feature_df,
                        label_name,
                        instance_sample_size)

            # Or feature importance is calculated only based on rare variant bins
            elif not score_with_common_variables:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if not score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
                elif score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix,
                                                                                          label_name,
                                                                                          instance_sample_size)
        elif scoring_method == 'Relief only on bin and common features':
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features_instance_sample(
                    bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                    instance_sample_size)
            elif not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features(bin_feature_matrix,
                                                                                              amino_acid_bins,
                                                                                              common_feature_list,
                                                                                              common_feature_df,
                                                                                              label_name)

        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, 'Class', amino_acid_bins)

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = crossover_and_mutation_multiprocess(set_number_of_bins, elitism_parameter, amino_acids,
                                                             amino_acid_bins, amino_acid_bin_scores,
                                                             crossover_probability, mutation_probability,
                                                             random_seeds[i], bin_size_variability_constraint,
                                                             max_features_per_bin)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = create_next_generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        # Updating the binned feature matrix
        bin_feature_matrix, amino_acid_bins = regroup_feature_matrix(amino_acids, rare_feature_df, label_name,
                                                                     feature_bin_list, random_seeds[iterations + i])

    # Creating the final bin scores
    if scoring_method == 'Relief':
        # Feature importance is calculating either with common variables
        if score_with_common_variables:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_rare_variants(bin_feature_matrix,
                                                                                    common_feature_list,
                                                                                    common_feature_df, label_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_rare_variants_instance_sample(
                    bin_feature_matrix,
                    common_feature_list,
                    common_feature_df,
                    label_name,
                    instance_sample_size)

        # Or feature importance is calculated only based on rare variant bins
        elif not score_with_common_variables:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                      instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, 'Class', amino_acid_bins)

    elif scoring_method == 'Relief only on bin and common features':
        # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features_instance_sample(
                bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                instance_sample_size)
        elif not score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features(bin_feature_matrix,
                                                                                          amino_acid_bins,
                                                                                          common_feature_list,
                                                                                          common_feature_df, label_name)

    # Creating a final feature matrix with both rare variant bins and common features
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_df[common_feature_list[i]]

    bin_feature_matrix['Class'] = original_feature_matrix[label_name]
    common_features_and_bins_matrix['Class'] = original_feature_matrix[label_name]

    return bin_feature_matrix, common_features_and_bins_matrix, \
        amino_acid_bins, amino_acid_bin_scores, rare_feature_maf_dict, \
        common_feature_maf_dict, rare_feature_df, common_feature_df, maf_0_features


# Defining a function to present the top bins
def top_rare_variant_bins_summary(rare_feature_matrix, label_name, bins, bin_scores,
                                  rare_feature_maf_dict, number_of_top_bins, bin_feature_matrix):
    # Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    # Calculating the chi square and p values of each of the features in the rare feature matrix
    df = rare_feature_matrix
    x = df.drop('Class', axis=1)
    y = df['Class']
    chi_scores, p_values = chi2(x, y)

    # Removing the label column to create a list of features
    feature_df = rare_feature_matrix.drop(columns=[label_name])

    # Creating a list of features
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))

    # Creating a dictionary with each feature and the chi-square value and p-value
    univariate_feature_stats = {}
    for i in range(0, len(feature_list)):
        list_of_stats = list()
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        univariate_feature_stats[feature_list[i]] = list_of_stats
        # There will be features with nan for their chi-square value and p-value because the whole column is zeroes

    # Calculating the chisquare and p values of each of the features in the bin feature matrix
    x = bin_feature_matrix.drop('Class', axis=1)
    y = bin_feature_matrix['Class']
    chi_scores, p_values = chi2(x, y)

    # Creating a dictionary with each bin and the chi-square value and p-value
    bin_stats = {}
    bin_names_list = list(bins.keys())
    for i in range(0, len(bin_names_list)):
        list_of_stats = list()
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        bin_stats[bin_names_list[i]] = list_of_stats
    string_df = []
    for i in range(0, number_of_top_bins):
        # Printing the bin Name
        string_df.append("Bin Rank " + str(i + 1) + ": " + str(sorted_bin_list[i]))
        # Printing the bin's MultiSURF/Univariate score, chi-square value, and p-value
        string_df.append("MultiSURF or Univariate Score: " + str(sorted_bin_feature_importance_values[i]) +
                         "; chi-square value: " + str(bin_stats[sorted_bin_list[i]][0]) +
                         "; p-value: " + str(bin_stats[sorted_bin_list[i]][1]))
        # Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range(0, len(bins[sorted_bin_list[i]])):
            string_df.append("Feature Name: " + str(bins[sorted_bin_list[i]][j]) +
                             "; minor allele frequency: " + str(rare_feature_maf_dict[bins[sorted_bin_list[i]][j]]) +
                             "; chi-square value: " + str(univariate_feature_stats[bins[sorted_bin_list[i]][j]][0]) +
                             "; p-value: " + str(univariate_feature_stats[bins[sorted_bin_list[i]][j]][1]))
        string_df.append('---------------------------')
    summary = pd.DataFrame(string_df, columns=['bin_rankings'])
    return summary


# ADDING FUNCTIONS TO TEST RARE


# Defining a function to create an artificial dataset with parameters, there will be one ideal/strong bin
# Note: MAF (minor allele frequency) cutoff refers to the threshold separating rare variant features from common
# features
def rvds_one_bin(number_of_instances, number_of_features, number_of_features_in_bin,
                 rare_variant_maf_cutoff, endpoint_cutoff_parameter, endpoint_variation_probability):
    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    # Adding one to number of features to give space for the class column
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_features + 1)))

    # Creating a list of features
    feature_list = []

    # Creating a list of predictive features in the strong bin
    predictive_features = []
    for a in range(0, number_of_features_in_bin):
        predictive_features.append("P_" + str(a + 1))

    for b in range(0, len(predictive_features)):
        feature_list.append(predictive_features[b])

    # Creating a list of randomly created features
    random_features = []
    for c in range(0, number_of_features - number_of_features_in_bin):
        random_features.append("R_" + str(c + 1))

    for d in range(0, len(random_features)):
        feature_list.append(random_features[d])

    # Adding the features and the class/endpoint
    features_and_class = feature_list.copy()
    features_and_class.append('Class')
    df.columns = features_and_class

    # Creating a list of numbers with the amount of numbers equals to the number of instances
    # This will be used when assigning values to the values of features that are in the bin
    instance_list = []
    for number in range(0, number_of_instances):
        instance_list.append(number)

    # ASSIGNING VALUES TO PREDICTIVE FEATURES
    # Randomly assigning instances in each of the predictive features the value of 1 or 2
    # Ensuring that the MAF (minor allele frequency) of each feature is a random value between 0 and the cutoff
    # Multiplying by 2 because there are two alleles for each instance
    for e in range(0, number_of_features_in_bin):
        # Calculating the sum of instances with minor alleles
        ma_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_ma2_instances = round(0.5 * (random.uniform(0, ma_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_ma1_instances = ma_sum - 2 * number_of_ma2_instances
        ma1_instances = random.sample(instance_list, number_of_ma1_instances)
        for f in ma1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_ma1 = list(set(instance_list) - set(ma1_instances))
        ma2_instances = random.sample(instances_wo_ma1, number_of_ma2_instances)
        for f in ma2_instances:
            df.at[f, predictive_features[e]] = 2

    # ASSIGNING ENDPOINT (CLASS) VALUES
    # Creating a list of bin values for the sum of values across predictive features in the bin
    bin_values = []
    for g in range(0, number_of_instances):
        sum_of_values = 0
        for h in predictive_features:
            sum_of_values += df.iloc[g][h]
        bin_values.append(sum_of_values)

    # User input for the cutoff between 0s and 1s is either mean or median.
    if endpoint_cutoff_parameter == 'mean':
        # Finding the mean to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.mean(bin_values)

        # If the sum of feature values in an instance is greater than or equal to the mean, then the endpoint will be 1
        for i in range(0, number_of_instances):
            if bin_values[i] > endpoint_cutoff:
                df.at[i, 'Class'] = 1

            elif bin_values[i] == endpoint_cutoff:
                df.at[i, 'Class'] = 1

            elif bin_values[1] < endpoint_cutoff:
                df.at[i, 'Class'] = 0

    elif endpoint_cutoff_parameter == 'median':
        # Finding the median to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.median(bin_values)

        # If the sum of feature values in an instance is greater than or equal to the mean, then the endpoint will be 1
        for i in range(0, number_of_instances):
            if bin_values[i] > endpoint_cutoff:
                df.at[i, 'Class'] = 1

            elif bin_values[i] == endpoint_cutoff:
                df.at[i, 'Class'] = 1

            elif bin_values[1] < endpoint_cutoff:
                df.at[i, 'Class'] = 0

    # Applying the "noise" parameter to introduce endpoint variability
    for instance in range(0, number_of_instances):
        if endpoint_variation_probability > random.uniform(0, 1):
            if df.loc[instance]['Class'] == 0:
                df.at[instance, 'Class'] = 1

            elif df.loc[instance]['Class'] == 1:
                df.at[instance, 'Class'] = 0

    # ASSIGNING VALUES TO RANDOM FEATURES
    # Randomly assigning instances in each of the random features the value of 1
    # Ensuring that the MAF of each feature is a random value between 0 and the cutoff (probably 0.05)
    # Multiplying by 2 because there are two alleles for each instance
    for e in range(0, len(random_features)):
        # Calculating the sum of instances with minor alleles
        ma_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_ma2_instances = round(0.5 * (random.uniform(0, ma_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_ma1_instances = ma_sum - 2 * number_of_ma2_instances
        ma1_instances = random.sample(instance_list, number_of_ma1_instances)
        for f in ma1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_ma1 = list(set(instance_list) - set(ma1_instances))
        ma2_instances = random.sample(instances_wo_ma1, number_of_ma2_instances)
        for f in ma2_instances:
            df.at[f, random_features[e]] = 2

    return df, endpoint_cutoff


# Defining a function to create an artificial dataset with parameters
# There will be an epi-static relationship between a bin and a common feature

# Note: MAF (minor allele frequency) cutoff refers to the threshold separating rare variant features from common
# features Common feature genotype frequencies list should be of the form [0.25, 0.5, 0.25] (numbers should add up to
# 1) List of predictive MLGs (multi-locus genotypes) should contain any of the 9 possibilites and be of the form
# below: [AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB, aaBb, aabb] Genotype Cutoff Metric can be mean or median
def rvds_bin_epistatic_interaction_with_common_feature(number_of_instances, number_of_rare_features,
                                                       number_of_features_in_bin,
                                                       rare_variant_maf_cutoff,
                                                       common_feature_genotype_frequencies_list,
                                                       genotype_cutoff_metric,
                                                       endpoint_variation_probability,
                                                       list_of_ml_gs_predicting_disease, print_summary):
    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    # Adding two to number of rare features to give space for the class column and give space for the common feature
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_rare_features + 2)))

    # Creating a list of features
    feature_list = []

    # Creating a list of features in the bin that interacts epi-statically with the common feature
    predictive_features = []
    for a in range(0, number_of_features_in_bin):
        predictive_features.append("P_" + str(a + 1))

    for b in range(0, len(predictive_features)):
        feature_list.append(predictive_features[b])

    # Creating a list of randomly created features
    random_features = []
    for c in range(0, number_of_rare_features - number_of_features_in_bin):
        random_features.append("R_" + str(c + 1))

    for d in range(0, len(random_features)):
        feature_list.append(random_features[d])

    # Adding the common feature to the feature list
    feature_list.append('Common_Feature')

    # Adding the features and the class/endpoint
    features_and_class = feature_list
    features_and_class.append('Class')
    df.columns = features_and_class

    # Creating a list of numbers with the amount of numbers equals to the number of instances
    # This will be used when assigning values to the values of features that are in the bin
    instance_list = []
    for number in range(0, number_of_instances):
        instance_list.append(number)

    # RANDOMLY ASSIGNING VALUES TO THE FEATURES IN THE BIN
    # Randomly assigning instances in each of the predictive features the value of 1 or 2
    # Ensuring that the MAF (minor allele frequency) of each feature is a random value between 0 and the cutoff
    for e in range(0, number_of_features_in_bin):
        # Calculating the sum of instances with minor alleles
        ma_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_ma2_instances = round(0.5 * (random.uniform(0, ma_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_ma1_instances = ma_sum - 2 * number_of_ma2_instances
        ma1_instances = random.sample(instance_list, number_of_ma1_instances)
        for f in ma1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_ma1 = list(set(instance_list) - set(ma1_instances))
        ma2_instances = random.sample(instances_wo_ma1, number_of_ma2_instances)
        for f in ma2_instances:
            df.at[f, predictive_features[e]] = 2

    # DETERMINE GENOTYPE CUTOFF FOR THE BIN (AA vs. Aa vs. aa)
    # Creating a list of bin values for the sum of values across predictive features in the bin
    bin_values = []
    for g in range(0, number_of_instances):
        sum_of_values = 0
        for h in predictive_features:
            sum_of_values += df.iloc[g][h]
        bin_values.append(sum_of_values)

    # Bin values that are 0 will be AA

    nonzero_bin_values = []
    # Creating a list of nonzero bin values, the mean of median of this will determine the cutoff between Aa and aa
    for i in range(0, len(bin_values)):
        if bin_values[i] != 0:
            nonzero_bin_values.append(bin_values[i])

    Aa_aa_cutoff = 0.5

    if genotype_cutoff_metric == 'mean':
        Aa_aa_cutoff = statistics.mean(nonzero_bin_values)

    elif genotype_cutoff_metric == 'median':
        Aa_aa_cutoff = statistics.median(nonzero_bin_values)

    # Creating a list for each of the bin genotypes
    bin_AA_genotype_list = []
    bin_Aa_genotype_list = []
    bin_aa_genotype_list = []

    for g in range(0, number_of_instances):
        if bin_values[g] == 0:
            bin_AA_genotype_list.append(g)

        elif (bin_values[g] > 0) and (bin_values[g] < Aa_aa_cutoff):
            bin_Aa_genotype_list.append(g)

        elif bin_values[g] == Aa_aa_cutoff or bin_values[g] > Aa_aa_cutoff:
            bin_aa_genotype_list.append(g)

    # ASSIGNING VALUES FOR THE COMMON FEATURE
    # Based on the given allele frequencies the user inputs, randomly choosing instances for each genotype (BB, Bb, bb)
    # of the common feature

    instances_left = instance_list.copy()
    common_feature_BB_genotype_list = random.sample(instance_list, (
        round(number_of_instances * float(common_feature_genotype_frequencies_list[0]))))
    for instance in common_feature_BB_genotype_list:
        df.at[instance, 'Common_Feature'] = 0
        instances_left.remove(instance)

    common_feature_Bb_genotype_list = random.sample(instances_left, (
        round(number_of_instances * float(common_feature_genotype_frequencies_list[1]))))
    for instance in common_feature_Bb_genotype_list:
        df.at[instance, 'Common_Feature'] = 1
        instances_left.remove(instance)

    common_feature_bb_genotype_list = random.sample(instances_left, (
        round(number_of_instances * float(common_feature_genotype_frequencies_list[2]))))
    for instance in common_feature_bb_genotype_list:
        df.at[instance, 'Common_Feature'] = 2

    # ASSIGNING CLASS/ENDPOINT VALUES

    # First assigning a value of 0 for all the classes (this will be overwritten later)
    for instance in instance_list:
        df.at[instance, 'Class'] = 0

    # Assigning a value of 1 for each instance that matches the MLG (multi-locus genotype) that the user specifies
    # should result in a diseased state
    disease_instances = []
    for instance in instance_list:
        if 'AABB' in list_of_ml_gs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AABb' in list_of_ml_gs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AAbb' in list_of_ml_gs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AaBB' in list_of_ml_gs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AaBb' in list_of_ml_gs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'Aabb' in list_of_ml_gs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aaBB' in list_of_ml_gs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aaBb' in list_of_ml_gs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aabb' in list_of_ml_gs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

    # Applying the "noise" parameter to introduce endpoint variability
    for instance in range(0, number_of_instances):
        if endpoint_variation_probability > random.uniform(0, 1):
            if df.loc[instance]['Class'] == 0:
                df.at[instance, 'Class'] = 1

            elif df.loc[instance]['Class'] == 1:
                df.at[instance, 'Class'] = 0

    # ASSIGNING VALUES TO RANDOM FEATURES
    # Randomly assigning instances in each of the random features the value of 1
    # Ensuring that the MAF of each feature is a random value between 0 and the cutoff (probably 0.05)
    for e in range(0, len(random_features)):
        # Calculating the sum of instances with minor alleles
        ma_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_ma2_instances = round(0.5 * (random.uniform(0, ma_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_ma1_instances = ma_sum - 2 * number_of_ma2_instances
        ma1_instances = random.sample(instance_list, number_of_ma1_instances)
        for f in ma1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_ma1 = list(set(instance_list) - set(ma1_instances))
        ma2_instances = random.sample(instances_wo_ma1, number_of_ma2_instances)
        for f in ma2_instances:
            df.at[f, random_features[e]] = 2

    if print_summary:
        # PRINTING INFORMATION ABOUT THE DATASET THAT CAN BE INPUTTED IN A PENETRANCE TABLE
        print("Probability of Disease (Class = 1) for Multi-Locus Genotypes:")
        MLG_list = ['AABB', 'AABb', 'AAbb', 'AaBB', 'AaBb', 'Aabb', 'aaBB', 'aaBb', 'aabb']
        MLG_penetrance_dict = {}
        for MLG in MLG_list:
            if MLG in list_of_ml_gs_predicting_disease:
                print(MLG + ': 1')
                MLG_penetrance_dict[MLG] = 1
            elif MLG not in list_of_ml_gs_predicting_disease:
                print(MLG + ': 0')
                MLG_penetrance_dict[MLG] = 0
        print("---")
        print('Marginal Penetrance for Genotypes:')
        AA_penetrance = ((len(common_feature_BB_genotype_list) * MLG_penetrance_dict['AABB'] / number_of_instances) +
                         (len(common_feature_Bb_genotype_list) * MLG_penetrance_dict['AABb'] / number_of_instances) +
                         (len(common_feature_bb_genotype_list) * MLG_penetrance_dict['AAbb'] / number_of_instances))
        print("AA: " + str(AA_penetrance))

        Aa_penetrance = ((len(common_feature_BB_genotype_list) * MLG_penetrance_dict['AaBB'] / number_of_instances) +
                         (len(common_feature_Bb_genotype_list) * MLG_penetrance_dict['AaBb'] / number_of_instances) +
                         (len(common_feature_bb_genotype_list) * MLG_penetrance_dict['Aabb'] / number_of_instances))
        print("Aa: " + str(Aa_penetrance))

        aa_penetrance = ((len(common_feature_BB_genotype_list) * MLG_penetrance_dict['aaBB'] / number_of_instances) +
                         (len(common_feature_Bb_genotype_list) * MLG_penetrance_dict['aaBb'] / number_of_instances) +
                         (len(common_feature_bb_genotype_list) * MLG_penetrance_dict['aabb'] / number_of_instances))
        print("aa: " + str(aa_penetrance))

        BB_penetrance = ((len(bin_AA_genotype_list) * MLG_penetrance_dict['AABB'] / number_of_instances) +
                         (len(bin_Aa_genotype_list) * MLG_penetrance_dict['AaBB'] / number_of_instances) +
                         (len(bin_aa_genotype_list) * MLG_penetrance_dict['aaBB'] / number_of_instances))
        print("BB: " + str(BB_penetrance))

        Bb_penetrance = ((len(bin_AA_genotype_list) * MLG_penetrance_dict['AABb'] / number_of_instances) +
                         (len(bin_Aa_genotype_list) * MLG_penetrance_dict['AaBb'] / number_of_instances) +
                         (len(bin_aa_genotype_list) * MLG_penetrance_dict['aaBb'] / number_of_instances))
        print("Bb: " + str(Bb_penetrance))

        bb_penetrance = ((len(bin_AA_genotype_list) * MLG_penetrance_dict['AAbb'] / number_of_instances) +
                         (len(bin_Aa_genotype_list) * MLG_penetrance_dict['Aabb'] / number_of_instances) +
                         (len(bin_aa_genotype_list) * MLG_penetrance_dict['aabb'] / number_of_instances))
        print("bb: " + str(bb_penetrance))
        print('---')
        print("Population Prevalence of Disease (K): " + str(len(disease_instances) / number_of_instances))

    return df, Aa_aa_cutoff


# Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
# This version of the function will tell if the algorithm has reached the 80% solution
# For testing purposes
# def rare_check_for_80_pct(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
#                           original_feature_matrix,
#                           label_name, rare_variant_maf_cutoff, set_number_of_bins,
#                           min_features_per_group, max_number_of_groups_with_feature,
#                           scoring_method, score_based_on_sample, score_with_common_variables,
#                           instance_sample_size, crossover_probability, mutation_probability, elitism_parameter):
def rare_check_for_80_pct(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
                          original_feature_matrix,
                          label_name, rare_variant_maf_cutoff, set_number_of_bins,
                          min_features_per_group, max_number_of_groups_with_feature,
                          scoring_method, score_based_on_sample, score_with_common_variables,
                          instance_sample_size, crossover_probability, mutation_probability, elitism_parameter,
                          random_seed, bin_size_variability_constraint, max_features_per_bin):
    # Step 0: Separate Rare Variants and Common Features
    rare_feature_list, rare_feature_maf_dict, rare_feature_df, common_feature_list, common_feature_maf_dict, \
        common_feature_df, maf_0_features = rare_and_common_variable_separation(original_feature_matrix,
                                                                                label_name, rare_variant_maf_cutoff)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups
    if given_starting_point:
        amino_acid_bins = amino_acid_bins_start_point.copy()
        amino_acids = amino_acid_start_point.copy()

        features_to_remove = [item for item in amino_acids if item not in rare_feature_list]
        for feature in features_to_remove:
            amino_acids.remove(feature)

        bin_names = amino_acid_bins.keys()
        for bin_name in bin_names:
            for feature in features_to_remove:
                if feature in amino_acid_bins[bin_name]:
                    amino_acid_bins[bin_name].remove(feature)

    # Otherwise randomly initialize the bins
    elif not given_starting_point:
        amino_acids, amino_acid_bins = random_feature_grouping(rare_feature_df, label_name,
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature,
                                                               random_seed, max_features_per_bin)
    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(rare_feature_df, label_name, amino_acid_bins)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    for iteration in range(0, iterations):

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # Feature importance can be scored with Relief or a univariate metric (chi squared value)
        if scoring_method == 'Relief':
            # Feature importance is calculating either with common variables
            if score_with_common_variables:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if not score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_rare_variants(bin_feature_matrix,
                                                                                        common_feature_list,
                                                                                        common_feature_df, label_name)
                elif score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_rare_variants_instance_sample(
                        bin_feature_matrix,
                        common_feature_list,
                        common_feature_df,
                        label_name,
                        instance_sample_size)

            # Or feature importance is calculated only based on rare variant bins
            elif not score_with_common_variables:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if not score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
                elif score_based_on_sample:
                    amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix,
                                                                                          label_name,
                                                                                          instance_sample_size)
        elif scoring_method == 'Relief only on bin and common features':
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features_instance_sample(
                    bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                    instance_sample_size)
            elif not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features(bin_feature_matrix,
                                                                                              amino_acid_bins,
                                                                                              common_feature_list,
                                                                                              common_feature_df,
                                                                                              label_name)

        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, 'Class', amino_acid_bins)

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = crossover_and_mutation_multiprocess(set_number_of_bins, elitism_parameter, amino_acids,
                                                             amino_acid_bins, amino_acid_bin_scores,
                                                             crossover_probability, mutation_probability, random_seed,
                                                             bin_size_variability_constraint,
                                                             max_features_per_bin)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = create_next_generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        # Updating the binned feature matrix
        bin_feature_matrix, amino_acid_bins = regroup_feature_matrix(amino_acids, rare_feature_df, label_name,
                                                                     feature_bin_list, random_seed)

        # Adding the stopping criteria:
        stop = False
        for bin_of_features in amino_acid_bins.values():
            predictive_features = ["P_1", "P_2", "P_3", "P_4", "P_5", "P_6", "P_7", "P_8", "P_9", "P_10"]
            random_features = ["R_1", "R_2", "R_3", "R_4", "R_5", "R_6", "R_7", "R_8", "R_9", "R_10",
                               "R_11", "R_12", "R_13", "R_14", "R_15", "R_16", "R_17", "R_18", "R_19", "R_20",
                               "R_21", "R_22", "R_23", "R_24", "R_25", "R_26", "R_27", "R_28", "R_29", "R_30",
                               "R_31", "R_32", "R_33", "R_34", "R_35", "R_36", "R_37", "R_38", "R_39", "R_40"]
            predictive_intersection = [value for value in bin_of_features if value in predictive_features]
            random_intersection = [value for value in bin_of_features if value in random_features]
            nonempty_predictive_features = [value for value in predictive_features if value in rare_feature_list]

            if len(predictive_intersection) > 0.7949 * len(nonempty_predictive_features) and len(
                    random_intersection) < 3:
                stop = True

        if stop is True:
            print("Reached 80% at " + str(iteration + 1))

        elif not stop:
            print("Didn't reach 80% at " + str(iteration + 1))

    # Creating the final amino acid bin scores
    if scoring_method == 'Relief':
        # Feature importance is calculating either with common variables
        if score_with_common_variables:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_rare_variants(bin_feature_matrix,
                                                                                    common_feature_list,
                                                                                    common_feature_df, label_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_rare_variants_instance_sample(
                    bin_feature_matrix,
                    common_feature_list,
                    common_feature_df,
                    label_name,
                    instance_sample_size)

        # Or feature importance is calculated only based on rare variant bins
        elif not score_with_common_variables:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                      instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, 'Class', amino_acid_bins)

    elif scoring_method == 'Relief only on bin and common features':
        # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features_instance_sample(
                bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                instance_sample_size)
        elif not score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_bin_and_common_features(bin_feature_matrix,
                                                                                          amino_acid_bins,
                                                                                          common_feature_list,
                                                                                          common_feature_df, label_name)

    # Creating a final feature matrix with both rare variant bins and common features
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_df[common_feature_list[i]]

    bin_feature_matrix['Class'] = original_feature_matrix[label_name]
    common_features_and_bins_matrix['Class'] = original_feature_matrix[label_name]

    return bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, \
        amino_acid_bin_scores, rare_feature_maf_dict, common_feature_maf_dict, \
        rare_feature_df, common_feature_df, maf_0_features
