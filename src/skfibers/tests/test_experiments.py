import pytest
from skfibers.fibers import FIBERS
from skfibers.experiments.datagen import create_data_simulation_bin
from skfibers.methods.algorithms import remove_empty_variables, random_feature_grouping, grouped_feature_matrix, \
    log_rank_test_feature_importance, tournament_selection_parent_bins, crossover_and_mutation, \
    create_next_generation, regroup_feature_matrix


def test_create_data_simulation_bin():
    data1 = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data1 = data1.drop('TrueRiskGroup', axis=1)
    data2 = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data2 = data2.drop('TrueRiskGroup', axis=1)
    assert (data1.equals(data2))


def test_random_feature_grouping():
    data1 = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data1 = data1.drop('TrueRiskGroup', axis=1)
    label_name = "Censoring"
    duration_name = "Duration"
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list \
        = remove_empty_variables(data1, label_name, duration_name)
    feature_list_1, binned_feature_groups_1 = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                                      duration_name, number_of_groups=50,
                                                                      min_features_per_group=2,
                                                                      max_number_of_groups_with_feature=10,
                                                                      random_seed=42, max_features_per_bin=None)
    feature_list_2, binned_feature_groups_2 = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                                      duration_name, number_of_groups=50,
                                                                      min_features_per_group=2,
                                                                      max_number_of_groups_with_feature=10,
                                                                      random_seed=42, max_features_per_bin=None)

    assert (feature_list_2 == feature_list_1)
    assert (binned_feature_groups_2 == binned_feature_groups_1)


def test_tournament_selection_parent_bins():
    data = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data = data.drop('TrueRiskGroup', axis=1)
    label_name = "Censoring"
    duration_name = "Duration"
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list \
        = remove_empty_variables(data, label_name, duration_name)
    feature_list, binned_feature_groups = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                                  duration_name, number_of_groups=50,
                                                                  min_features_per_group=2,
                                                                  max_number_of_groups_with_feature=10,
                                                                  random_seed=42, max_features_per_bin=None)

    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, duration_name,
                                                binned_feature_groups)

    bin_scores = log_rank_test_feature_importance(bin_feature_matrix, binned_feature_groups, label_name, duration_name,
                                                  informative_cutoff=0.2)
    parent_bins_1 = tournament_selection_parent_bins(bin_scores, random_seed=42)
    parent_bins_2 = tournament_selection_parent_bins(bin_scores, random_seed=42)
    assert (parent_bins_2 == parent_bins_1)


def test_crossover_and_mutation():
    data = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data = data.drop('TrueRiskGroup', axis=1)
    label_name = "Censoring"
    duration_name = "Duration"
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list \
        = remove_empty_variables(data, label_name, duration_name)
    feature_list, binned_feature_groups = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                                  duration_name, number_of_groups=50,
                                                                  min_features_per_group=2,
                                                                  max_number_of_groups_with_feature=10,
                                                                  random_seed=42, max_features_per_bin=None)

    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, duration_name,
                                                binned_feature_groups)

    bin_scores = log_rank_test_feature_importance(bin_feature_matrix, binned_feature_groups, label_name, duration_name,
                                                  informative_cutoff=0.2)

    offspring_bins_1 = crossover_and_mutation(max_population_of_bins=50, elitism_parameter=0.8,
                                              feature_list=feature_list, binned_feature_groups=binned_feature_groups,
                                              bin_scores=bin_scores,
                                              crossover_probability=0.8,
                                              mutation_probability=0.4, random_seed=42, max_features_per_bin=None)

    offspring_bins_2 = crossover_and_mutation(max_population_of_bins=50, elitism_parameter=0.8,
                                              feature_list=feature_list, binned_feature_groups=binned_feature_groups,
                                              bin_scores=bin_scores,
                                              crossover_probability=0.8,
                                              mutation_probability=0.4, random_seed=42, max_features_per_bin=None)

    assert (offspring_bins_2 == offspring_bins_1)


def test_regroup_feature_matrix():
    data = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    original_feature_matrix = data.drop('TrueRiskGroup', axis=1)
    label_name = "Censoring"
    duration_name = "Duration"
    feature_matrix_no_empty_variables, maf_0_features, nonempty_feature_list \
        = remove_empty_variables(original_feature_matrix, label_name, duration_name)
    feature_list, binned_feature_groups = random_feature_grouping(feature_matrix_no_empty_variables, label_name,
                                                                  duration_name, number_of_groups=50,
                                                                  min_features_per_group=2,
                                                                  max_number_of_groups_with_feature=10,
                                                                  random_seed=42, max_features_per_bin=None)

    bin_feature_matrix = grouped_feature_matrix(feature_matrix_no_empty_variables, label_name, duration_name,
                                                binned_feature_groups)

    bin_scores = log_rank_test_feature_importance(bin_feature_matrix, binned_feature_groups, label_name, duration_name,
                                                  informative_cutoff=0.2)

    offspring_bins = crossover_and_mutation(max_population_of_bins=50, elitism_parameter=0.8,
                                            feature_list=feature_list, binned_feature_groups=binned_feature_groups,
                                            bin_scores=bin_scores,
                                            crossover_probability=0.8,
                                            mutation_probability=0.4, random_seed=42, max_features_per_bin=None)

    feature_bin_list = create_next_generation(binned_feature_groups, bin_scores=bin_scores, max_population_of_bins=50,
                                              elitism_parameter=0.8, offspring_list=offspring_bins)

    bin_feature_matrix_1, amino_acid_bins_1 = regroup_feature_matrix(feature_list, original_feature_matrix, label_name,
                                                                     duration_name, feature_bin_list,
                                                                     random_seed=42)

    bin_feature_matrix_2, amino_acid_bins_2 = regroup_feature_matrix(feature_list, original_feature_matrix, label_name,
                                                                     duration_name, feature_bin_list,
                                                                     random_seed=42)

    assert (bin_feature_matrix_2.equals(bin_feature_matrix_1))
    assert (amino_acid_bins_2 == amino_acid_bins_1)


def test_experiment_random_seed():
    data = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    fibers_1 = FIBERS(given_starting_point=False, amino_acid_start_point=None,
                      amino_acid_bins_start_point=None,
                      iterations=100,
                      label_name="Censoring",
                      duration_name="Duration",
                      set_number_of_bins=50, min_features_per_group=2,
                      max_number_of_groups_with_feature=5, crossover_probability=0.8,
                      mutation_probability=0.1, elitism_parameter=0.4,
                      random_seed=42)

    fibers_1 = fibers_1.fit(data)

    fibers_2 = FIBERS(given_starting_point=False, amino_acid_start_point=None,
                      amino_acid_bins_start_point=None,
                      iterations=100,
                      label_name="Censoring",
                      duration_name="Duration",
                      set_number_of_bins=50, min_features_per_group=2,
                      max_number_of_groups_with_feature=5, crossover_probability=0.8,
                      mutation_probability=0.1, elitism_parameter=0.4,
                      random_seed=42)

    fibers_2 = fibers_2.fit(data)
    assert (fibers_1.bins == fibers_2.bins)
    assert (fibers_1.bin_feature_matrix.equals(fibers_2.bin_feature_matrix))
    assert (fibers_1.bin_scores == fibers_2.bin_scores)


@pytest.mark.skip(reason="Big Runtime")
def test_experiments():
    data = create_data_simulation_bin()
    results = list()
    for replicate in range(0, 2):
        print('Replication ' + str(replicate))
        fibers = FIBERS(given_starting_point=False, amino_acid_start_point=None,
                        amino_acid_bins_start_point=None,
                        iterations=100,
                        label_name="Censoring",
                        duration_name="Duration",
                        set_number_of_bins=50, min_features_per_group=2,
                        max_number_of_groups_with_feature=5, crossover_probability=0.8,
                        mutation_probability=0.1, elitism_parameter=0.4,
                        random_seed=42)

        fibers = fibers.fit(data)
        fibers, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features = fibers.transform(data)
        results.append((fibers, bin_feature_matrix_internal, amino_acid_bins_internal,
                        amino_acid_bin_scores_internal, maf_0_features))
    for i in range(1, len(results)):
        assert (results[i][1].equals(results[i - 1][1]))
        assert (results[i][2] == results[i - 1][2])
        assert (results[i][3] == results[i - 1][3])
