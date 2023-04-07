import random
import numpy as np
import pandas as pd


def generate_features(row, number_of_features, number_of_features_in_bin, mm_frequency):
    if row['Class'] == 1:
        idxs = random.sample(list(range(1, number_of_features_in_bin + 1)),
                             int(mm_frequency/2 * number_of_features_in_bin))
        for idx in idxs:
            row['P_' + str(idx)] = 1
        idxs = random.sample(list(range(1, number_of_features - number_of_features_in_bin + 1)),
                             int(mm_frequency/2 * (number_of_features - number_of_features_in_bin)))
        for idx in idxs:
            row['R_' + str(idx)] = 1
    else:
        idxs = random.sample(list(range(1, number_of_features - number_of_features_in_bin + 1)),
                             int(mm_frequency * (number_of_features - number_of_features_in_bin)))
        for idx in idxs:
            row['R_' + str(idx)] = 1
    return row


def create_data_simulation_bin(number_of_instances, number_of_features, number_of_features_in_bin,
                               no_fail_proportion, mm_frequency_range, noise_frequency,
                               class0_time_to_event_range, class1_time_to_event_range):
    """
    Defining a function to create an artificial dataset with parameters, there will be one ideal/strong bin
    Note: MAF (minor allele frequency) cutoff refers to the threshold
    separating rare variant features from common features

    :param number_of_instances: dataset size
    :param number_of_features: total number of features in dataset
    :param number_of_features_in_bin: total number of predictive features in the ideal bin
    :param no_fail_proportion: the proportion of instances to be labled as (no fail class)
    :param mm_frequency_range: the max and min MM frequency for a given column/feature in data. (e.g. 0.1 to 0.5)
    :param noise_frequency: Value from 0 to 0.5 representing the proportion of class 0/class 1 instance pairs that \
                            have their outcome switched from 0 to 1
    :param class0_time_to_event_range: (min, max) time to event as a tuple (should be larger (e.g. 100 to 200))
    :param class1_time_to_event_range: (min, max) time to event as a tuple (should be smaller but a bit overlapping \
                                        with above range (e.g. 20 to 150))

    :return: pandas dataframe of generated data
    """

    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    # Adding one to number of features to give space for the class and Duration column
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_features + 2)))

    # Creating a list of predictive features in the strong bin
    predictive_features = ["P_" + str(i + 1) for i in range(number_of_features_in_bin)]

    # Creating a list of randomly created features
    random_features = ["R_" + str(i + 1) for i in range(number_of_features - number_of_features_in_bin)]

    # Adding the features and the class/endpoint
    df.columns = predictive_features + random_features + ['Class', 'Duration']

    # Assigning class according to no_fail_proportion parameter
    fail_count = int(number_of_instances * (1 - no_fail_proportion))
    no_fail_count = number_of_instances - fail_count
    class_list = [1] * fail_count + [0] * no_fail_count
    df['Class'] = class_list

    # Generating predictive and random features columns
    mm_frequency = np.random.uniform(mm_frequency_range[0], mm_frequency_range[1])
    df = df.apply(generate_features,
                  args=(number_of_features, number_of_features_in_bin, mm_frequency), axis=1).astype(int)

    # Assigning Gaussian according to class
    df_0 = df[df['Class'] == 0].sample(frac=1).reset_index(drop=True)
    df_1 = df[df['Class'] == 1].sample(frac=1).reset_index(drop=True)
    df_0['Duration'] = np.random.uniform(class0_time_to_event_range[0],
                                         class0_time_to_event_range[1], size=len(df_0))
    df_1['Duration'] = np.random.uniform(class1_time_to_event_range[0],
                                         class1_time_to_event_range[1], size=len(df_1))
    swap_count = min(no_fail_count, fail_count) * noise_frequency

    idxs = random.sample(list(min(no_fail_count, fail_count)), swap_count)



    df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)

    return df
