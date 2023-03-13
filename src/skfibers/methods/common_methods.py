import os
import random
import statistics
import numpy as np
import pandas as pd
from random import randrange
from multiprocessing import Pool


# Common Functions

# Defining a function to delete variables with MAF = 0
def remove_empty_variables(original_feature_matrix, label_name, duration_name):
    # Removing the label column to create a list of features
    if duration_name:
        feature_df = original_feature_matrix.drop(columns=[label_name, duration_name])
    else:
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
    feature_matrix_no_empty_variables[label_name] = original_feature_matrix[label_name]

    if duration_name:
        feature_matrix_no_empty_variables[duration_name] = original_feature_matrix[duration_name]

    return feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list


# Defining a function to group features randomly, each feature can be in a number of groups up to a set max
def random_feature_grouping(feature_matrix, label_name, duration_name, number_of_groups, min_features_per_group,
                            max_number_of_groups_with_feature, random_seed, max_features_per_bin):
    # Removing the label column to create a list of features

    if duration_name:
        feature_df = feature_matrix.drop(columns=[label_name, duration_name])
    else:
        feature_df = feature_matrix.drop(columns=[label_name])

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
    if duration_name:
        bins_df[duration_name] = feature_matrix[duration_name]
    return bins_df


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


# Crossover and mutation - Single Core

# Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents n is
# the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next time)
def crossover_and_mutation(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups, bin_scores,
                           crossover_probability, mutation_probability, random_seed, bin_size_variability_constraint,
                           max_features_per_bin):
    # Creating a list for offspring
    offspring_list = []

    # Creating a number of offspring equal to the number needed to replace the non-elites
    # Each pair of parents will produce two offspring
    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 10)
    for i in range(0, num_replacement_sets):
        # Choosing the two parents and getting the list of features in each parent bin
        parent_bins = tournament_selection_parent_bins(bin_scores, random_seeds[i])
        parent1_features = binned_feature_groups[parent_bins[0]].copy()
        parent2_features = binned_feature_groups[parent_bins[1]].copy()

        # Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []

        # CROSSOVER
        # Each feature in the parent bin will cross over based on the given probability (uniform crossover)

        # Creating two df for parent features and probability of crossover
        np.random.seed(random_seeds[num_replacement_sets + i])
        randnums1 = list(np.random.randint(0, 101, len(parent1_features)))
        crossover_threshold1 = list([crossover_probability * 100] * len(parent1_features))
        parent1_df = pd.DataFrame(parent1_features, columns=['Features'])
        parent1_df['Threshold'] = crossover_threshold1
        parent1_df['Rand_prob'] = randnums1

        np.random.seed(random_seeds[num_replacement_sets * 2 + i])
        randnums2 = list(np.random.randint(0, 101, len(parent2_features)))
        crossover_threshold2 = list([crossover_probability * 100] * len(parent2_features))
        parent2_df = pd.DataFrame(parent2_features, columns=['Features'])
        parent2_df['Threshold'] = crossover_threshold2
        parent2_df['Rand_prob'] = randnums2

        # Features with random probability less than the crossover probability will go to offspring 1.
        # The rest will go to offspring 2.
        offspring1.extend(list(parent1_df.loc[parent1_df['Threshold'] > parent1_df['Rand_prob']]['Features']))
        offspring2.extend(list(parent1_df.loc[parent1_df['Threshold'] <= parent1_df['Rand_prob']]['Features']))
        offspring1.extend(list(parent2_df.loc[parent2_df['Threshold'] > parent2_df['Rand_prob']]['Features']))
        offspring2.extend(list(parent2_df.loc[parent2_df['Threshold'] <= parent2_df['Rand_prob']]['Features']))

        # Remove repeated features within each offspring
        offspring1 = list(set(offspring1))
        offspring2 = list(set(offspring2))

        # MUTATION
        # (deletion and addition) only occurs with a certain probability on each feature in the
        # original feature space

        # Creating a probability for mutation (addition) that accounts for the ratio between the feature list and the
        # size of the bin
        if len(offspring1) > 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob1 = mutation_probability * (len(offspring1)) / (
                (len(feature_list) - len(offspring1)))
        elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob1 = mutation_probability
        elif len(offspring1) == len(feature_list):
            mutation_addition_prob1 = 0

        if len(offspring2) > 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob2 = mutation_probability * (len(offspring2)) / (
                (len(feature_list) - len(offspring2)))
        elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob2 = mutation_probability
        elif len(offspring2) == len(feature_list):
            mutation_addition_prob2 = 0

        # Mutation: Deletion occurs on features with probability equal to the mutation parameter
        offspring1_df = pd.DataFrame(offspring1, columns=['Features'])
        mutation_threshold1 = list([mutation_probability * 100] * len(offspring1))
        np.random.seed(random_seeds[num_replacement_sets * 3 + i])
        rand1 = list(np.random.randint(0, 101, len(offspring1)))
        offspring1_df['Threshold'] = mutation_threshold1
        offspring1_df['Rand_prob'] = rand1

        offspring2_df = pd.DataFrame(offspring2, columns=['Features'])
        mutation_threshold2 = list([mutation_probability * 100] * len(offspring2))
        np.random.seed(random_seeds[num_replacement_sets * 4 + i])
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
        np.random.seed(random_seeds[num_replacement_sets * 5 + i])
        rand1 = list(np.random.randint(0, 101, len(features_not_in_offspring1)))
        features_not_in_offspring1_df['Threshold'] = mutation_addition_threshold1
        features_not_in_offspring1_df['Rand_prob'] = rand1

        features_not_in_offspring2_df = pd.DataFrame(features_not_in_offspring2, columns=['Features'])
        mutation_addition_threshold2 = list([mutation_addition_prob2 * 100] * len(features_not_in_offspring2_df))
        np.random.seed(random_seeds[num_replacement_sets * 6 + i])
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
            np.random.seed(random_seeds[num_replacement_sets * 7 + i])
            random_seeds_loop = np.random.randint(len(feature_list) * len(feature_list), size=2 * len(feature_list))
            counter = 0
            while counter < 2 * len(feature_list) and (
                    len(offspring1) > c_constraint * len(offspring2)
                    or len(offspring2) > c_constraint * len(offspring1)):
                np.random.seed(random_seeds_loop[counter])
                random.seed(random_seeds_loop[counter])

                if len(offspring1) > c_constraint * len(offspring2):
                    min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                    min_to_move = min_features - len(offspring2)
                    max_to_move = len(offspring1) - min_features
                    np.random.seed(random_seed)
                    num_to_move = np.random.randint(min_to_move, max_to_move + 1)
                    random.seed(random_seed)
                    features_to_move = list(random.sample(offspring1, num_to_move))
                    offspring1 = [x for x in offspring1 if x not in features_to_move]
                    offspring2.extend(features_to_move)
                elif len(offspring2) > c_constraint * len(offspring1):
                    min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                    min_to_move = min_features - len(offspring1)
                    max_to_move = len(offspring2) - min_features
                    np.random.seed(random_seed)
                    num_to_move = np.random.randint(min_to_move, max_to_move + 1)
                    random.seed(random_seed)
                    features_to_move = random.sample(offspring2, num_to_move)
                    offspring2 = [x for x in offspring2 if x not in features_to_move]
                    offspring1.extend(features_to_move)
                offspring1 = list(set(offspring1))
                offspring2 = list(set(offspring2))
                counter = counter + 1

        # Ensuring the size of the offspring is not greater than the max_features_per_bin allowed
        if not (max_features_per_bin is None):
            if len(offspring1) > max_features_per_bin:
                random.seed(random_seeds[num_replacement_sets * 10 + i])
                offspring1 = list(random.sample(offspring1, max_features_per_bin))
            if len(offspring2) > max_features_per_bin:
                random.seed(random_seeds[num_replacement_sets * 10 + i])
                offspring2 = list(random.sample(offspring2, max_features_per_bin))

        # Adding the new offspring to the list of feature bins
        offspring_list.append(offspring1)
        offspring_list.append(offspring2)

    return offspring_list


# Crossover and mutation - Multiprocessing
# Defining a function to perform crossover and mutation using multiple cores at once
def crossover_and_mutation_multiprocess(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups,
                                        bin_scores,
                                        crossover_probability, mutation_probability, random_seed,
                                        bin_size_variability_constraint,
                                        max_features_per_bin):
    # Determining the number of offspring created
    num_replacement_sets = int((max_population_of_bins - (elitism_parameter * max_population_of_bins)) / 2)
    np.random.seed(random_seed)

    # TODO: next parameter depends on number of cores
    random_seeds = np.random.randint(len(feature_list) * len(feature_list), size=num_replacement_sets * 10)
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
def regroup_feature_matrix(feature_list, feature_matrix, label_name, duration_name, feature_bin_list, random_seed):
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
    bins_df[label_name] = feature_matrix[label_name]
    if duration_name:
        bins_df[duration_name] = feature_matrix[duration_name]
    return bins_df, binned_feature_groups
