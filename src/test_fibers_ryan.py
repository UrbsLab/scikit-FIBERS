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
                                  censoring_frequency=0.5, random_seed=42, negative=False, threshold=0)

data.to_csv('sampledata.csv', index=False)
data = pd.read_csv('sampledata.csv')
true_risk_group = data[['TrueRiskGroup']]
data = data.drop('TrueRiskGroup', axis=1)

fibers = FIBERS(outcome_label="Duration", outcome_type="survival",iterations=100,
                    pop_size = 50, crossover_prob=0.5, mutation_prob=0.1, new_gen=1.0, min_bin_size=1,
                    fitness_metric="log_rank", censor_label="Censoring", group_strata_min=0.2,
                    group_thresh=0, min_thresh=0, max_thresh=3, int_thresh=True, thresh_evolve_prob=0.5,
                    manual_bin_init=None, covariates=['R_77','R_80'], random_seed=None)

fibers = fibers.fit(data)

#bin_summary, logrank_results = fibers.get_bin_summary(prin=True)
#print(logrank_results)


#pipe_fibers = Pipeline(steps=[("FIBERS", fibers_obj)])
#pipe_fibers.fit(data)
#transformed_df = pipe_fibers.transform(data)
#assert isinstance(transformed_df, pd.DataFrame)
