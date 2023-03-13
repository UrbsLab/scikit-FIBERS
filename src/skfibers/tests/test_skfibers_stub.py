import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from skfibers import FIBERS


@pytest.mark.skip(reason="Problem With Dataset")
def test_fibers():
    data = pd.read_csv('data/Imp1.csv')
    fibers_obj = FIBERS(given_starting_point=False, amino_acid_start_point=None, algorithm="FIBERS",
                        amino_acid_bins_start_point=None, iterations=1000, label_name="Class",
                        duration_name="grf_yrs", rare_variant_maf_cutoff=0.05,
                        set_number_of_bins=50, min_features_per_group=5,
                        max_number_of_groups_with_feature=25,
                        scoring_method='Relief',
                        score_based_on_sample=True, score_with_common_variables=False,
                        instance_sample_size=50, crossover_probability=0.8,
                        mutation_probability=0.1, elitism_parameter=0.4,
                        random_seed=None, bin_size_variability_constraint=None)

    pipe_fibers = Pipeline(steps=[("FIBERS", fibers_obj)])
    pipe_fibers.fit(data)
    pipe_fibers.transform(data)


@pytest.mark.skip(reason="Already Tested")
def test_rare():
    data = pd.read_csv('data/Experiment1.csv')
    rare_obj = FIBERS(given_starting_point=False, amino_acid_start_point=None, algorithm="RARE",
                      amino_acid_bins_start_point=None, iterations=1000, label_name="Class",
                      duration_name=False, rare_variant_maf_cutoff=0.05,
                      set_number_of_bins=50, min_features_per_group=5,
                      max_number_of_groups_with_feature=25,
                      scoring_method='Relief',
                      score_based_on_sample=True, score_with_common_variables=False,
                      instance_sample_size=50, crossover_probability=0.8,
                      mutation_probability=0.1, elitism_parameter=0.4,
                      random_seed=None, bin_size_variability_constraint=None)

    pipe_fibers = Pipeline(steps=[("RARE", rare_obj)])
    pipe_fibers.fit(data)
    pipe_fibers.transform(data)
