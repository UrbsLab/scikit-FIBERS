# RAFE Algorithm
# Defining a function for the RAFE algorithm (Relief-based Association Feature-bin Evolver)
# Same as RARE, but it bins all features not just rare features
def rafe_algorithm(given_starting_point,
                   amino_acid_start_point,
                   amino_acid_bins_start_point,
                   iterations,
                   original_feature_matrix,
                   label_name, duration_name,
                   set_number_of_bins,
                   min_features_per_group,
                   max_number_of_groups_with_feature,
                   scoring_method,
                   score_based_on_sample,
                   instance_sample_size,
                   crossover_probability,
                   mutation_probability,
                   elitism_parameter, random_seed,
                   bin_size_variability_constraint,
                   max_features_per_bin):
    # Step 0: Deleting Empty Features (MAF = 0)
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list = remove_empty_variables(
        original_feature_matrix, label_name, duration_name)

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
    for i in range(0, iterations):

        # Step 2a: Feature Importance Scoring and Bin Deletion

        # Feature scoring can be done with Relief or a univariate chi squared test
        if scoring_method == 'Relief':
            # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if not score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name, duration_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                      instance_sample_size)
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, label_name, duration_name,
                                                                  amino_acid_bins)

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
                                                                     duration_name,
                                                                     feature_bin_list, random_seeds[iterations + i])

    # Creating the final bin scores
    # Feature scoring can be done with Relief or a univariate chi squared test
    if scoring_method == 'Relief':
        # Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if not score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name, duration_name)
        elif score_based_on_sample:
            amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                  instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, label_name, duration_name,
                                                              amino_acid_bins)

    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, maf_0_features


# RARE Algorithm
# Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
def rare_algorithm_v2(given_starting_point,
                      amino_acid_start_point,
                      amino_acid_bins_start_point,
                      iterations,
                      original_feature_matrix,
                      label_name, duration_name,
                      rare_variant_maf_cutoff,
                      set_number_of_bins,
                      min_features_per_group,
                      max_number_of_groups_with_feature,
                      scoring_method,
                      score_based_on_sample,
                      score_with_common_variables,
                      instance_sample_size,
                      crossover_probability,
                      mutation_probability,
                      elitism_parameter, random_seed,
                      bin_size_variability_constraint,
                      max_features_per_bin):
    # Step 0: Separate Rare Variants and Common Features
    rare_feature_list, rare_feature_maf_dict, \
        rare_feature_df, common_feature_list, \
        common_feature_maf_dict, common_feature_df, \
        maf_0_features = rare_and_common_variable_separation(original_feature_matrix,
                                                             label_name, duration_name, rare_variant_maf_cutoff)

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
        amino_acids, amino_acid_bins = random_feature_grouping(rare_feature_df, label_name, duration_name,
                                                               set_number_of_bins, min_features_per_group,
                                                               max_number_of_groups_with_feature, random_seed,
                                                               max_features_per_bin)
    # Create Initial Binned Feature Matrix
    bin_feature_matrix = grouped_feature_matrix(rare_feature_df, label_name, duration_name, amino_acid_bins)

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
                    amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name, duration_name)
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
            amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, label_name, duration_name,
                                                                  amino_acid_bins)

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
                                                                     duration_name,
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
                amino_acid_bin_scores = multi_surf_feature_importance(bin_feature_matrix, label_name, duration_name)
            elif score_based_on_sample:
                amino_acid_bin_scores = multi_surf_feature_importance_instance_sample(bin_feature_matrix, label_name,
                                                                                      instance_sample_size)
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = chi_square_feature_importance(bin_feature_matrix, label_name, duration_name,
                                                              amino_acid_bins)

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

    bin_feature_matrix[label_name] = original_feature_matrix[label_name]
    common_features_and_bins_matrix[label_name] = original_feature_matrix[label_name]

    return bin_feature_matrix, common_features_and_bins_matrix, \
        amino_acid_bins, amino_acid_bin_scores, rare_feature_maf_dict, \
        common_feature_maf_dict, rare_feature_df, common_feature_df, maf_0_features


# Defining a function to present the top bins
def top_bins_summary_rare(original_feature_matrix, label_name, duration_name,
                          bin_feature_matrix, bins, bin_scores, number_of_top_bins,):
    # Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    # Calculating the chi square and p values of each of the features in the original feature matrix
    df = original_feature_matrix
    x = df.drop(label_name, axis=1)
    if duration_name:
        x = x.drop(duration_name, axis=1)
    y = df[label_name]
    chi_scores, p_values = chi2(x, y)

    # Removing the label column to create a list of features
    if duration_name:
        feature_df = original_feature_matrix.drop(columns=[label_name, duration_name])
    else:
        feature_df = original_feature_matrix.drop(columns=[label_name, duration_name])

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
    x = bin_feature_matrix.drop(label_name, axis=1)
    y = bin_feature_matrix[label_name]
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


# Defining a function to present the top bins
def top_rare_variant_bins_summary(rare_feature_matrix, label_name, duration_name, bins, bin_scores,
                                  rare_feature_maf_dict, number_of_top_bins, bin_feature_matrix):
    # Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    # Calculating the chi square and p values of each of the features in the rare feature matrix
    df = rare_feature_matrix
    x = df.drop(label_name, axis=1)
    if duration_name:
        x = x.drop(duration_name, axis=1)
    y = df[label_name]
    chi_scores, p_values = chi2(x, y)

    # Removing the label column to create a list of features
    if duration_name:
        feature_df = rare_feature_matrix.drop(columns=[label_name, duration_name])
    else:
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
    x = bin_feature_matrix.drop(label_name, axis=1)
    if duration_name:
        x = x.drop(duration_name, axis=1)
    y = bin_feature_matrix[label_name]
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
