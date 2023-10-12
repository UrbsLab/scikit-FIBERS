# !/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import argparse
import pandas as pd
from sklearn.metrics import classification_report

from skfibers import FIBERS
from skfibers.experiments.datagen_evolvable_threshold import create_data_simulation_bin_evolve


def save_fibers_object(fibers):
    if not os.path.exists('run_history/'):
        os.makedirs('run_history/')
    with open('run_history/' + 'adaptive_' + str(fibers.adaptable_threshold) +
              '_evolving_prob_' + str(fibers.evolving_probability) +
              '_thresh_' + str(fibers.threshold) + '.pickle', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(fibers, outp, pickle.HIGHEST_PROTOCOL)


def experiment(number_of_instances=10000, number_of_features=750, number_of_features_in_bin=10,
               no_fail_proportion=0.5, mm_frequency_range=(0.4, 0.5), noise_frequency=0.0,
               class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2),
               censoring_frequency=0.5, random_seed=42, negative=False, threshold=2,
               iterations=1000, set_number_of_bins=50,
               min_features_per_group=2, max_number_of_groups_with_feature=4,
               informative_cutoff=0.2, crossover_probability=0.5,
               mutation_probability=0.4, elitism_parameter=0.8,
               mutation_strategy="Regular",
               set_threshold=0, evolving_probability=1,
               min_threshold=0, max_threshold=3, merge_probability=0.0,
               adaptable_threshold=True,
               scoring_method="log_rank"):
    # Creating Simulation Data by provided functionality
    data = create_data_simulation_bin_evolve(number_of_instances=number_of_instances,
                                             number_of_features=number_of_features,
                                             number_of_features_in_bin=number_of_features_in_bin,
                                             no_fail_proportion=no_fail_proportion,
                                             mm_frequency_range=mm_frequency_range,
                                             noise_frequency=noise_frequency,
                                             class0_time_to_event_range=class0_time_to_event_range,
                                             class1_time_to_event_range=class1_time_to_event_range,
                                             censoring_frequency=censoring_frequency, random_seed=random_seed,
                                             negative=negative, threshold=threshold)
    if not os.path.exists('run_history/'):
        os.makedirs('run_history/')
    # data.to_csv('run_history/' + 'sampledata.csv')
    # data = pd.read_csv('run_history/' + 'sampledata.csv')
    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    # Running Evaluation
    fibers = FIBERS(label_name="Censoring", duration_name="Duration",
                    given_starting_point=False, start_point_feature_list=None, feature_bins_start_point=None,
                    iterations=iterations, set_number_of_bins=set_number_of_bins,
                    min_features_per_group=min_features_per_group,
                    max_number_of_groups_with_feature=max_number_of_groups_with_feature,
                    informative_cutoff=informative_cutoff, crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability, elitism_parameter=elitism_parameter,
                    mutation_strategy=mutation_strategy, random_seed=random_seed,
                    set_threshold=set_threshold, evolving_probability=evolving_probability,
                    min_threshold=min_threshold, max_threshold=max_threshold, merge_probability=merge_probability,
                    adaptable_threshold=adaptable_threshold, covariates=None,
                    scoring_method=scoring_method)
    fibers = fibers.fit(data)

    # Saving FIBERS Object
    save_fibers_object(fibers)

    # Summary of Top Bin Statistics
    bin_summary, logrank_results = fibers.get_bin_summary(
        save='run_history/' + 'adaptive_' + str(fibers.adaptable_threshold) +
             '_evolving_prob_' + str(fibers.evolving_probability) + '_thresh_' + str(fibers.threshold)
             + '_bin_summary.csv')
    print(logrank_results)

    score_df = fibers.get_bin_scores(save='run_history/' + 'adaptive_' + str(fibers.adaptable_threshold) +
                                          'evolving_prob_' + str(fibers.evolving_probability) + '_thresh_'
                                          + str(fibers.threshold) + 'bin_scores.csv')
    print(score_df.head(10))

    fibers.get_bin_survival_plot(show=False, save='run_history/' + 'adaptive_' + str(fibers.adaptable_threshold) +
                                                  'evolving_prob_' + str(fibers.evolving_probability) + '_thresh_'
                                                  + str(fibers.threshold)
                                                  + 'survival_plot.png')

    print("Accuracy: ", fibers.score(data, true_risk_group))

    y = fibers.predict(data)
    print(classification_report(y, true_risk_group))


def main(config_dict):
    experiment(**config_dict)


def parser_function(argv):
    parser = argparse.ArgumentParser(description="scikitFIBERS: \n"
                                                 "A scikit-learn compatible implementation of "
                                                 "FIBERS (Feature Inclusion Bin "
                                                 "Evolver for Risk Stratification)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_dict = {'number_of_instances': 10000, 'number_of_features': 750, 'number_of_features_in_bin': 10,
                    'no_fail_proportion': 0.5, 'noise_frequency': 0.0,
                    'censoring_frequency': 0.5, 'random_seed': 42, 'threshold': 2,
                    'iterations': 1000, 'set_number_of_bins': 50,
                    'min_features_per_group': 2, 'max_number_of_groups_with_feature': 4,
                    'informative_cutoff': 0.2, 'crossover_probability': 0.5,
                    'mutation_probability': 0.4, 'elitism_parameter': 0.8,
                    'mutation_strategy': "Regular",
                    'set_threshold': 0, 'evolving_probability': 1,
                    'min_threshold': 0, 'max_threshold': 3, 'merge_probability': 0.0,
                    'adaptable_threshold': True,
                    'scoring_method': "log_rank"}

    for key, value in default_dict.items():
        parser.add_argument('--' + key.replace('_', '-'), nargs='?', default=value, type=type(value))

    args, unknown = parser.parse_known_args(argv[1:])
    parse_dict = vars(args)
    return parse_dict


if __name__ == '__main__':
    # NOTE: All keys must be small
    param_dict = parser_function(sys.argv)
    print("Run Params:")
    print(param_dict)
    sys.exit(main(param_dict))
