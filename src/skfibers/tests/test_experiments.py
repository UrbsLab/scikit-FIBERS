import logging
import pytest
from skfibers.fibers import FIBERS
from skfibers.experiments.datagen import create_data_simulation_bin


def test_create_data_simulation_bin():
    data1 = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    data2 = create_data_simulation_bin(number_of_instances=1000, random_seed=42)
    assert (data1.equals(data2))


def test_experiment():
    for replicate in range(0, 1):
        print('Replication ' + str(replicate))

        # Creating the simulated dataset with 1000 instances, 10 features to bin, 50 total features
        data = create_data_simulation_bin()

        fibers = FIBERS(given_starting_point=False, amino_acid_start_point=None,
                        amino_acid_bins_start_point=None,
                        iterations=100,
                        label_name="Censoring",
                        duration_name="Duration",
                        set_number_of_bins=50, min_features_per_group=2,
                        max_number_of_groups_with_feature=5, crossover_probability=0.8,
                        mutation_probability=0.1, elitism_parameter=0.4,
                        random_seed=None)

        fibers.fit(data)
        fibers, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features = fibers.transform(data)
        logging.warning(amino_acid_bins_internal)
        logging.warning(amino_acid_bin_scores_internal)
        logging.warning(maf_0_features)
        print(amino_acid_bins_internal)
        print(amino_acid_bin_scores_internal)
        print(maf_0_features)


@pytest.mark.skip(reason="Test Later")
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

    fibers_1.fit(data)

    fibers_2 = FIBERS(given_starting_point=False, amino_acid_start_point=None,
                      amino_acid_bins_start_point=None,
                      iterations=100,
                      label_name="Censoring",
                      duration_name="Duration",
                      set_number_of_bins=50, min_features_per_group=2,
                      max_number_of_groups_with_feature=5, crossover_probability=0.8,
                      mutation_probability=0.1, elitism_parameter=0.4,
                      random_seed=42)

    fibers_2.fit(data)
    assert (fibers_1.bins == fibers_2.bins)
    assert (fibers_1.bin_feature_matrix == fibers_2.bin_feature_matrix)
    assert (fibers_1.bin_scores == fibers_2.bin_scores)
