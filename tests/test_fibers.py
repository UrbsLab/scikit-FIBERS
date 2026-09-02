# To test using pytest run `pytest --log-cli-level=DEBUG` from the root folder`

import os
from src.skfibers.fibers import FIBERS
from src.skfibers.simulators.SIM1 import survival_data_simulation
from sklearn.metrics import classification_report

def test_sim1():
    print("------------------------------------------------------")
    print("Test: SIM1 - 1000 instances, 5 predictive features 15 non-predictive features, threshold=0")

    data = survival_data_simulation(instances=1000, total_features=20, predictive_features=3, low_risk_proportion=0.5, 
                                    threshold = 0, feature_frequency_range=(0.1, 0.4), noise_frequency=0.0, 
                                    class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2), 
                                    censoring_frequency=0.2, negative_control=False, random_seed=42)
    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    fibers = FIBERS(outcome_label="Duration", outcome_type="survival", iterations=40, pop_size=50, tournament_prop=0.5, 
                    crossover_prob=0.5, min_mutation_prob=0.1, max_mutation_prob=0.5, merge_prob=0.1, new_gen=1.0, 
                    elitism=0.1, diversity_pressure=3, min_bin_size=1, max_bin_size=None, max_bin_init_size=10, 
                    fitness_metric="log_rank", log_rank_weighting=None, censor_label="Censoring", 
                    group_strata_min=0.2, penalty=0.5, group_thresh=0, min_thresh=0, max_thresh=3, int_thresh=True, 
                    thresh_evolve_prob=0.5, manual_bin_init=None, covariates=None, report=None, 
                    random_seed=42,verbose=False)

    fibers = fibers.fit(data)
    fibers.get_bin_report(0)
    predictions = fibers.predict(data,bin_number=0)
    print(classification_report(predictions, true_risk_group, digits=8))



