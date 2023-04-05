import logging
import pytest
from skfibers.fibers import FIBERS
from skfibers.experiments.datagen import create_data_simulation_bin


@pytest.mark.skip(reason="Omitting for github")
def test_experiment_1():
    for replicate in range(0, 1):
        print('Experiment 1')

        # Creating the simulated dataset with 1000 instances, 10 features to bin, 50 total features
        simdata, cutoff = create_data_simulation_bin(1000, 50, 10, 0.05, 'mean', 0)

        fibers = FIBERS(given_starting_point=False, amino_acid_start_point=None, algorithm="FIBERS",
                        amino_acid_bins_start_point=None, iterations=1000, label_name="Class",
                        duration_name="Duration", rare_variant_maf_cutoff=0.05,
                        set_number_of_bins=50, min_features_per_group=5,
                        max_number_of_groups_with_feature=25,
                        scoring_method='Relief',
                        score_based_on_sample=True, score_with_common_variables=False,
                        instance_sample_size=50, crossover_probability=0.8,
                        mutation_probability=0.1, elitism_parameter=0.4,
                        random_seed=None, bin_size_variability_constraint=None)

        fibers.fit(simdata)
        fibers, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features = fibers.transform(simdata)
        logging.warning(amino_acid_bins_internal)
        logging.warning(amino_acid_bin_scores_internal)
        logging.warning(maf_0_features)
        print(amino_acid_bins_internal)
        print(amino_acid_bin_scores_internal)
        print(maf_0_features)
