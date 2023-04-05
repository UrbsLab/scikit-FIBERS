import random
import numpy as np
import pandas as pd
import statistics


def create_data_simulation_bin(number_of_instances, number_of_features, number_of_features_in_bin,
                               rare_variant_maf_cutoff, endpoint_cutoff_parameter,
                               endpoint_variation_probability):
    """
    Defining a function to create an artificial dataset with parameters, there will be one ideal/strong bin
    Note: MAF (minor allele frequency) cutoff refers to the threshold
    separating rare variant features from common features

    :param number_of_instances:
    :param number_of_features:
    :param number_of_features_in_bin:
    :param rare_variant_maf_cutoff:
    :param endpoint_cutoff_parameter:
    :param endpoint_variation_probability:
    :return:
    """
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

    endpoint_cutoff = None

    # User input for the cutoff between 0s and 1s is either mean or median.
    if endpoint_cutoff_parameter == 'mean':
        # Finding the mean to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.mean(bin_values)

        # If the sum of feature values in an instance
        # is greater than or equal to the mean, then the endpoint will be 1
        for i in range(0, number_of_instances):
            if bin_values[i] >= endpoint_cutoff:
                df.at[i, 'Class'] = 1
            else:
                df.at[i, 'Class'] = 0

    elif endpoint_cutoff_parameter == 'median':
        # Finding the median to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.median(bin_values)

        # If the sum of feature values in an instance is greater than or equal to the mean, then the endpoint will be 1
        for i in range(0, number_of_instances):
            if bin_values[i] >= endpoint_cutoff:
                df.at[i, 'Class'] = 1
            else:
                df.at[i, 'Class'] = 0

    # Applying the "noise" parameter to introduce endpoint variability
    for instance in range(0, number_of_instances):
        if endpoint_variation_probability > random.uniform(0, 1):
            if df.loc[instance]['Class'] == 0:
                df.at[instance, 'Class'] = 1
            elif df.loc[instance]['Class'] == 1:
                df.at[instance, 'Class'] = 0

    # ASSIGNING GAUSSIAN DURATION TO CLASS
    df_0 = df[df['Class'] == 0].sample(frac=1).reset_index(drop=True)
    df_1 = df[df['Class'] == 1].sample(frac=1).reset_index(drop=True)
    df_0['Duration'] = np.random.normal(6.86, 4.86, size=len(df_0))
    df_1['Duration'] = np.random.normal(4.47, 3.77, size=len(df_1))

    df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)

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
