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
    features_and_class.append('Duration')
    df.columns = features_and_class

    # Creating a list of numbers with the amount of numbers equal to the number of instances
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
        MA_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5 * (random.uniform(0, MA_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2 * number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
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
        MA_sum = round((random.uniform(0, 2 * rare_variant_maf_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5 * (random.uniform(0, MA_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2 * number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, random_features[e]] = 2

    return df, endpoint_cutoff


# In[ ]:


# Defining a function to create an artificial dataset with parameters
# There will be an epistatic relationship between a bin and a common feature
# Note: MAF (minor allele frequency) cutoff refers to the threshold separating rare variant features from common features
# Common feature genotype frequencies list should be of the form [0.25, 0.5, 0.25] (numbers should add up to 1)
# List of predictive MLGs (multi-locus genotypes) should contain any of the 9 possibilites and be of the form below:
# [AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB, aaBb, aabb]
# Genotype Cutoff Metric can be mean or median
def RVDS_Bin_Epistatic_Interaction_with_Common_Feature(number_of_instances, number_of_rare_features,
                                                       number_of_features_in_bin,
                                                       rare_variant_MAF_cutoff,
                                                       common_feature_genotype_frequencies_list,
                                                       genotype_cutoff_metric,
                                                       endpoint_variation_probability,
                                                       list_of_MLGs_predicting_disease, print_summary):
    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    # Adding two to number of rare features to give space for the class column and give space for the common feature
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_rare_features + 2)))

    # Creating a list of features
    feature_list = []

    # Creating a list of features in the bin that interacts epistatically with the common feature
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

    # Creating a list of numbers with the amount of numbers equal to the number of instances
    # This will be used when assigning values to the values of features that are in the bin
    instance_list = []
    for number in range(0, number_of_instances):
        instance_list.append(number)

    # RANDOMLY ASSIGNING VALUES TO THE FEATURES IN THE BIN
    # Randomly assigning instances in each of the predictive features the value of 1 or 2
    # Ensuring that the MAF (minor allele frequency) of each feature is a random value between 0 and the cutoff
    for e in range(0, number_of_features_in_bin):
        # Calculating the sum of instances with minor alleles
        MA_sum = round((random.uniform(0, 2 * rare_variant_MAF_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5 * (random.uniform(0, MA_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2 * number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, predictive_features[e]] = 2

    # DETERMING GENOTYPE CUTOFF FOR THE BIN (AA vs. Aa vs. aa)
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

        elif bin_values[g] > 0 and bin_values[g] < Aa_aa_cutoff:
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

    # First assigning a value of 0 for all the classes (this will be overwrote later)
    for instance in instance_list:
        df.at[instance, 'Class'] = 0

    # Assigning a value of 1 for each instance that matches the MLG (multi-locus genotype) that the user specifies should
    # result in a diseased state
    disease_instances = []
    for instance in instance_list:
        if 'AABB' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AABb' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AAbb' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AaBB' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'AaBb' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'Aabb' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aaBB' in list_of_MLGs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aaBb' in list_of_MLGs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aabb' in list_of_MLGs_predicting_disease:
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
        MA_sum = round((random.uniform(0, 2 * rare_variant_MAF_cutoff)) * number_of_instances)
        # Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5 * (random.uniform(0, MA_sum * 0.5)))
        # The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2 * number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, random_features[e]] = 2

    if print_summary == True:
        # PRINTING INFORMATION ABOUT THE DATASET THAT CAN BE INPUTTED IN A PENETRANCE TABLE
        print("Probability of Disease (Class = 1) for Multi-Locus Genotypes:")
        MLG_list = ['AABB', 'AABb', 'AAbb', 'AaBB', 'AaBb', 'Aabb', 'aaBB', 'aaBb', 'aabb']
        MLG_penetrance_dict = {}
        for MLG in MLG_list:
            if MLG in list_of_MLGs_predicting_disease:
                print(MLG + ': 1')
                MLG_penetrance_dict[MLG] = 1
            elif MLG not in list_of_MLGs_predicting_disease:
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


# In[ ]:


# Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
# This version of the function will tell if the algorithm has reached the 80% solution
# For testing purposes
def RARE_check_for_80_pct(given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations,
                          original_feature_matrix,
                          label_name, rare_variant_MAF_cutoff, set_number_of_bins,
                          min_features_per_group, max_number_of_groups_with_feature,
                          scoring_method, score_based_on_sample, score_with_common_variables,
                          instance_sample_size, crossover_probability, mutation_probability, elitism_parameter):
    # Step 0: Separate Rare Variants and Common Features
    rare_feature_list, rare_feature_MAF_dict, rare_feature_df, common_feature_list, common_feature_MAF_dict, common_feature_df, MAF_0_features = Rare_and_Common_Variable_Separation(
        original_feature_matrix, label_name, rare_variant_MAF_cutoff)

    # Step 1: Initialize Population of Candidate Bins
    # Initialize Feature Groups
    if given_starting_point == True:
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
    elif given_starting_point == False:
        amino_acids, amino_acid_bins = Random_Feature_Grouping(rare_feature_df, label_name,
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature)
    # Create Initial Binned Feature Matrix
    bin_feature_matrix = Grouped_Feature_Matrix(rare_feature_df, label_name, amino_acid_bins)

    # Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    for iteration in range(0, iterations):

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # Feature importance can be scored with Relief or a univariate metric (chi squared value)
        if scoring_method == 'Relief':
            # Feature importance is calculating either with common variables
            if score_with_common_variables == True:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants(bin_feature_matrix,
                                                                                       common_feature_list,
                                                                                       common_feature_df, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants_Instance_Sample(
                        bin_feature_matrix,
                        common_feature_list,
                        common_feature_df,
                        label_name,
                        instance_sample_size)

            # Or feauture importance is calculated only based on rare variant bins
            elif score_with_common_variables == False:
                # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name,
                                                                                         instance_sample_size)
        elif scoring_method == 'Relief only on bin and common features':
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample == True:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features_Instance_Sample(
                    bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                    instance_sample_size)
            elif score_based_on_sample == False:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features(bin_feature_matrix,
                                                                                             amino_acid_bins,
                                                                                             common_feature_list,
                                                                                             common_feature_df,
                                                                                             label_name)

        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)

        # Step 2b: Genetic Algorithm
        # Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation_Multiprocess(set_number_of_bins, elitism_parameter, amino_acids,
                                                             amino_acid_bins, amino_acid_bin_scores,
                                                             crossover_probability, mutation_probability)

        # Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = Create_Next_Generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins,
                                                  elitism_parameter, offspring_bins)

        # Updating the binned feature matrix
        bin_feature_matrix, amino_acid_bins = Regroup_Feature_Matrix(amino_acids, rare_feature_df, label_name,
                                                                     feature_bin_list)

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

        if stop == True:
            print("Reached 80% at " + str(iteration + 1))

        elif stop == False:
            print("Didn't reach 80% at " + str(iteration + 1))

    # Creating the final amino acid bin scores
    if scoring_method == 'Relief':
        # Feature importance is calculating either with common variables
        if score_with_common_variables == True:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample == False:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants(bin_feature_matrix,
                                                                                   common_feature_list,
                                                                                   common_feature_df, label_name)
            elif score_based_on_sample == True:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants_Instance_Sample(bin_feature_matrix,
                                                                                                   common_feature_list,
                                                                                                   common_feature_df,
                                                                                                   label_name,
                                                                                                   instance_sample_size)

        # Or feature importance is calculated only based on rare variant bins
        elif score_with_common_variables == False:
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample == False:
                amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
            elif score_based_on_sample == True:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name,
                                                                                     instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)

    elif scoring_method == 'Relief only on bin and common features':
        # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if score_based_on_sample == True:
            amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features_Instance_Sample(
                bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name,
                instance_sample_size)
        elif score_based_on_sample == False:
            amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features(bin_feature_matrix,
                                                                                         amino_acid_bins,
                                                                                         common_feature_list,
                                                                                         common_feature_df, label_name)

    # Creating a final feature matrix with both rare variant bins and common features
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    for i in range(0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_df[common_feature_list[i]]

    bin_feature_matrix['Class'] = original_feature_matrix[label_name]
    common_features_and_bins_matrix['Class'] = original_feature_matrix[label_name]

    return bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features
