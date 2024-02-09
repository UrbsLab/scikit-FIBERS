#import logging
#import pytest


import pandas as pd
import os
current_working_directory = os.getcwd()
print(current_working_directory)

#from sklearn.pipeline import Pipeline
from skfibers.fibers import FIBERS

from skfibers.experiments.datagen import create_data_simulation_bin
from skfibers.experiments.datagen_evolvable_threshold import create_data_simulation_bin_evolve


data = create_data_simulation_bin_evolve(number_of_instances=10000, number_of_features=100, number_of_features_in_bin=10,
                                  no_fail_proportion=0.5, mm_frequency_range=(0.4, 0.5), noise_frequency=0.0,
                                  class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2),
                                  censoring_frequency=0.5, random_seed=42, negative=False, threshold=2)

data.to_csv('sampledata.csv', index=False)
data = pd.read_csv('sampledata.csv')
true_risk_group = data[['TrueRiskGroup']]
data = data.drop('TrueRiskGroup', axis=1)

fibers = FIBERS(label_name="Censoring", duration_name="Duration", 
                given_starting_point=False, start_point_feature_list=None, feature_bins_start_point=None,
                iterations=100, set_number_of_bins=50, 
                min_features_per_group=2, max_number_of_groups_with_feature=4,
                informative_cutoff=0.2, crossover_probability=0.5, 
                mutation_probability=0.4, elitism_parameter=0.8,
                mutation_strategy="Regular", random_seed=None, 
                set_threshold=0, evolving_probability=1,
                min_threshold=0, max_threshold=3, merge_probability=1, 
                adaptable_threshold=True, covariates=None,
                scoring_method="log_rank")

fibers = fibers.fit(data)

bin_summary, logrank_results = fibers.get_bin_summary(prin=True)
print(logrank_results)


#pipe_fibers = Pipeline(steps=[("FIBERS", fibers_obj)])
#pipe_fibers.fit(data)
#transformed_df = pipe_fibers.transform(data)
#assert isinstance(transformed_df, pd.DataFrame)
