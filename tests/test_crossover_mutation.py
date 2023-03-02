import timeit
import random
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def crossover_and_mutation2(parent1_features, parent2_features, feature_list, crossover_probability,
                            mutation_probability,
                            bin_size_variability_constraint, max_features_per_bin):
    offspring_list = []

    # Creating two lists for the offspring bins
    offspring1 = []
    offspring2 = []

    # CROSSOVER
    # Each feature in the parent bin will cross over based on the given probability (uniform crossover)

    # Creating two df for parent features and probability of crossover
    randnums1 = list(np.random.randint(0, 101, len(parent1_features)))
    crossover_threshold1 = list([crossover_probability * 100] * len(parent1_features))
    parent1_df = pd.DataFrame(parent1_features, columns=['Features'])
    parent1_df['Threshold'] = crossover_threshold1
    parent1_df['Rand_prob'] = randnums1

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
    #  (deletion and addition) only occurs with a certain probability on each feature in the
    # original feature space

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
    rand1 = list(np.random.randint(0, 101, len(offspring1)))
    offspring1_df['Threshold'] = mutation_threshold1
    offspring1_df['Rand_prob'] = rand1

    offspring2_df = pd.DataFrame(offspring2, columns=['Features'])
    mutation_threshold2 = list([mutation_probability * 100] * len(offspring2))
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
    rand1 = list(np.random.randint(0, 101, len(features_not_in_offspring1)))
    features_not_in_offspring1_df['Threshold'] = mutation_addition_threshold1
    features_not_in_offspring1_df['Rand_prob'] = rand1

    features_not_in_offspring2_df = pd.DataFrame(features_not_in_offspring2, columns=['Features'])
    mutation_addition_threshold2 = list([mutation_addition_prob2 * 100] * len(features_not_in_offspring2_df))
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
        counter = 0
        while counter < 2 * len(feature_list) and (
                len(offspring1) > c_constraint * len(offspring2) or len(offspring2) > c_constraint * len(offspring1)):
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
            offspring1 = list(random.sample(offspring1, max_features_per_bin))
        if len(offspring2) > max_features_per_bin:
            offspring2 = list(random.sample(offspring2, max_features_per_bin))

    offspring_list.extend([offspring1, offspring2])

    return offspring_list


# Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents n is
# the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next time)
def crossover_and_mutation(parent1_features, parent2_features, feature_list, crossover_probability,
                           mutation_probability):
    offspring_list = []
    # Creating two lists for the offspring bins
    offspring1 = []
    offspring2 = []

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
    while len(offspring1) > 2 * len(offspring2):
        switch = random.choice(offspring1)
        offspring1.remove(switch)
        offspring2.append(switch)

    while len(offspring2) > 2 * len(offspring1):
        switch = random.choice(offspring2)
        offspring2.remove(switch)
        offspring1.append(switch)

    # MUTATION
    #  only occurs with a certain probability on each feature in the original feature space

    # Applying the mutation operation to the first offspring Creating a probability for adding a feature that
    # accounts for the ratio between the feature list and the size of the bin
    if len(offspring1) > 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob = mutation_probability * (len(offspring1)) / (len(feature_list) - len(offspring1))
    elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob = mutation_probability
    elif len(offspring1) == len(feature_list):
        mutation_addition_prob = 0

    deleted_list = []
    # Deletion form of mutation
    for var in range(0, len(offspring1)):
        # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
        if mutation_probability > random.random():
            deleted_list.append(offspring1[var])

    for var in range(0, len(deleted_list)):
        offspring1.remove(deleted_list[var])

    # Creating a list of features outside the offspring
    features_not_in_offspring = [item for item in feature_list if item not in offspring1]

    # Addition form of mutation
    for var in range(0, len(features_not_in_offspring)):
        # Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
        # The probability accounts for the ratio between the feature list and the size of the bin
        if mutation_addition_prob > random.random():
            offspring1.append(features_not_in_offspring[var])

    # Applying the mutation operation to the second offspring Creating a probability for adding a feature that
    # accounts for the ratio between the feature list and the size of the bin
    if len(offspring2) > 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob = mutation_probability * (len(offspring2)) / (len(feature_list) - len(offspring2))
    elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob = mutation_probability
    elif len(offspring2) == len(feature_list):
        mutation_addition_prob = 0

    deleted_list = []
    # Deletion form of mutation
    for var in range(0, len(offspring2)):
        # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
        if mutation_probability > random.random():
            deleted_list.append(offspring2[var])

    for var in range(0, len(deleted_list)):
        offspring2.remove(deleted_list[var])

    # Creating a list of features outside the offspring
    features_not_in_offspring = [item for item in feature_list if item not in offspring2]

    # Addition form of mutation
    for var in range(0, len(features_not_in_offspring)):
        # Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
        # The probability accounts for the ratio between the feature list and the size of the bin
        if mutation_addition_prob > random.random():
            offspring2.append(features_not_in_offspring[var])

    # CLEANUP
    # Deleting any repeats of an amino acid in a bin
    # Removing duplicates of features in the same bin that may arise due to crossover
    unique = []
    for a in range(0, len(offspring1)):
        if offspring1[a] not in unique:
            unique.append(offspring1[a])

    # Adding random features from outside the bin to replace the deleted features in the bin
    replace_number = len(offspring1) - len(unique)
    features_not_in_offspring = []
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
    features_not_in_offspring = []
    features_not_in_offspring = [item for item in feature_list if item not in offspring2]
    offspring2 = unique.copy()
    if len(features_not_in_offspring) > replace_number:
        replacements = random.sample(features_not_in_offspring, replace_number)
    else:
        replacements = features_not_in_offspring.copy()
    offspring2.extend(replacements)

    offspring_list.append(offspring1)
    offspring_list.append(offspring2)

    return offspring_list


class TestCrossOver(unittest.TestCase):
    def testFn(self):
        times = pd.DataFrame(columns=["Orig_time", "V2_time", 'Orig_L1', 'Orig_L2', 'V2_L1', 'V2_L2'])
        for i in range(0, 1000):
            print("iteration: " + str(i))

            num_features = 100
            bin_size = 15

            parent1_features = list(np.random.randint(0, num_features, bin_size))
            parent2_features = list(np.random.randint(0, num_features, bin_size))

            feature_list = []
            for x in range(0, num_features):
                feature_list.append(x)

            crossover_probability = 0.8
            mutation_probability = 0.5
            bin_size_variability_constraint = 2
            max_features_per_bin = None

            startsep2 = timeit.default_timer()
            # Running RARE and checking if 80% of predictive features are reached at each iteration
            # This will be used for Trial 2 to see how fast RARE bins when given/not given partial expert knowledge
            offspring_list2 = crossover_and_mutation2(parent1_features, parent2_features, feature_list, crossover_probability,
                                                      mutation_probability, bin_size_variability_constraint,
                                                      max_features_per_bin)
            print(offspring_list2)
            stopsep2 = timeit.default_timer()

            startsep1 = timeit.default_timer()
            offspring_list1 = crossover_and_mutation(parent1_features, parent2_features, feature_list, crossover_probability,
                                                     mutation_probability)
            print(offspring_list1)
            stopsep1 = timeit.default_timer()

            times.loc[i] = [stopsep1 - startsep1, stopsep2 - startsep2, len(offspring_list1[0]), len(offspring_list1[1]),
                            len(offspring_list2[0]), len(offspring_list2[1])]
            print(i)

        offspring1 = list(times['Orig_L1']) + list(times['Orig_L2'])
        offspring2 = list(times['V2_L1']) + list(times['V2_L2'])
        print(times['Orig_time'].mean())
        print(times['V2_time'].mean())
        print((times['V2_time'].mean() - times['Orig_time'].mean()) / times['Orig_time'].mean())

        fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the *bins* keyword argument.
        axs[0, 0].hist(times['Orig_L1'])
        axs[0, 1].hist(times['Orig_L2'])
        axs[1, 0].hist(times['V2_L1'])
        axs[1, 1].hist(times['V2_L2'])


if __name__ == '__main__':
    unittest.main()
