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
                                  censoring_frequency=0.5, random_seed=42, negative=False, threshold=1)

data.to_csv('sampledata.csv', index=False)
data = pd.read_csv('sampledata.csv')
true_risk_group = data[['TrueRiskGroup']]
data = data.drop('TrueRiskGroup', axis=1)

fibers = FIBERS(outcome_label="Duration", outcome_type="survival",iterations=50,
                    pop_size = 50, crossover_prob=0.5, mutation_prob=0.1, new_gen=1.0, elitism=0.1, min_bin_size=1,
                    fitness_metric="log_rank", log_rank_weighting=None,censor_label="Censoring", group_strata_min=0.2,
                    group_thresh=None, min_thresh=0, max_thresh=5, int_thresh=True, thresh_evolve_prob=0.5,
                    manual_bin_init=None, covariates=None, report=[0,10], random_seed=None, verbose=True)

fibers = fibers.fit(data)

print(fibers.top_perform_df)

tdf = fibers.transform(data)
print(tdf)

predictions = fibers.predict(data,bin_number=0)
print(predictions)

predictions = fibers.predict(data)
print(predictions)

low_outcome, high_outcome, low_censor, high_censor, bin_report_df = fibers.get_bin_groups(data,0)

feature_names, feature_tracking = fibers.get_feature_tracking()
print(feature_names)
print(feature_tracking)