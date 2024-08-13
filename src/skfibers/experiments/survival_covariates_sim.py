import random
import math
import pandas as pd
import numpy as np


#def survival_data_simulation(instances=10000, total_features=100, predictive_features=10, low_risk_proportion=0.5, threshold = 0, 
#                             feature_frequency_range=(0.1, 0.5), noise_frequency=0.0, class0_time_to_event_range=(1.5, 0.2), 
#                             class1_time_to_event_range=(1, 0.2), censoring_frequency=0.2, covariates_to_sim=0, covariates_signal_range=(0.2,0.4),random_seed=None):


def survival_data_simulation_covariates(instances=10000,total_features=100,predictive_features=10,feature_frequency_range=(0.1, 0.5),random_seed=None):

    #predictive_features = 2
    patient_censor_prob = 0.1
    random_features = total_features-predictive_features-1
    administrative_censoring_time = 23

    # Initialize lists to store data
    recipient_factors = []
    donor_factors = []
    pred_values = []

    #P1_values = []
    #P2_values = []
    TC1_values = []
    #PC2_values = [] #test
    patient_censoring_times = []
    administrative_censoring_times = []
    graft_failure_times = []
    years_follow_ups = []
    graft_failures = []

    random.seed(random_seed)
    # Generate data for 10,000 observations
    for i in range(0, instances):
        #Assign Covariate Values
        recipient_factor = random.gauss(0, 1)
        donor_factor = random.gauss(0, 1)

        #Assign feature values
        feature_list = []
        for j in range(0,predictive_features):
            #select feature frequency
            feature_frequency = random.uniform(feature_frequency_range[0], feature_frequency_range[1])
            feature_list.append(int(random.random() < feature_frequency))

        #P1 = int(random.random() < 0.3)
        #P2 = int(random.random() < 0.3)
            
        #TC1 = int(random.random() > 0.5 + recipient_factor/2 + donor_factor/2)
        #if random.random() > 0.2:
        #    TC1 = int(random.random() > recipient_factor/2 + donor_factor/2)
        #else:
        #    TC1 = int(random.random() > 0.5)
        TC1 = int(random.random() > recipient_factor/2 + donor_factor/2)
        #feature_frequency = random.uniform(feature_frequency_range[0], feature_frequency_range[1]) #test
        #PC2 = int(random.random() < feature_frequency) #test

        #Calculate True Graft Failure Time
        predictive_contribution = 0
        for each in feature_list:
            predictive_contribution += 1*each  #0.15

        #rate = math.exp(0.2*recipient_factor - 0.3*donor_factor + 0.15*P1 + 0.15*P2- 1)
        rate = math.exp(1*recipient_factor + 1*donor_factor + predictive_contribution - 2)
        #rate = math.exp(1*recipient_factor*PC2 + 1*donor_factor*PC2 + predictive_contribution - 2) #test
        graft_failure_time = random.expovariate(rate)

        #Determine Patient Cencoring Time
        if random.random() < patient_censor_prob:
            patient_censoring_time = random.expovariate(1)
        else:
            patient_censoring_time = administrative_censoring_time

        #Determine Actual Event/Censoring Time (i.e. Duration)
        years_follow_up = min(patient_censoring_time, administrative_censoring_time, graft_failure_time)
        graft_failure = int(years_follow_up < min(patient_censoring_time, administrative_censoring_time))

        #Update lists
        recipient_factors.append(recipient_factor)
        donor_factors.append(donor_factor)
        pred_values.append(feature_list)

        #P1_values.append(P1)
        #P2_values.append(P2)
        TC1_values.append(TC1)
        #PC2_values.append(PC2) #test
        patient_censoring_times.append(patient_censoring_time)
        administrative_censoring_times.append(administrative_censoring_time)
        graft_failure_times.append(graft_failure_time)
        years_follow_ups.append(years_follow_up)
        graft_failures.append(graft_failure)

    #Generate Predictive Feature Dataframe
    columns = []
    for j in range(1,predictive_features+1):
        columns.append('P_'+str(j))
    df_predictive = pd.DataFrame(pred_values, columns=columns)    

    #Generate Random Features
    random_names = ["R_" + str(j + 1) for j in range(random_features)]
    df_random = pd.DataFrame(np.zeros((instances, random_features)))
    df_random.columns = random_names

    # Identify target MAF for each feature in the dataset.
    MAF_list = [random.uniform(feature_frequency_range[0], feature_frequency_range[1]) for _ in range(random_features)] 
    one_count_list = [int(x * instances) for x in MAF_list]

    # Generate Random Feature Values -------------------------------
    feature_index = 0
    for feature in random_names:
        one_count = one_count_list[feature_index]
        row_indexes = [random.randint(0, instances-1) for _ in range(one_count)]
        for row in row_indexes:
            df_random.at[row,feature] = 1
        feature_index += 1 

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'TC_1': TC1_values,
        #'PC_2': PC2_values, #test
        'C_1': recipient_factors,
        'C_2': donor_factors,
        #'P_1': P1_values,
        #'P_2': P2_values,
        'patient_censoring_time': patient_censoring_times,
        'administrative_censoring_time': administrative_censoring_times,
        'graft_failure_time': graft_failure_times,
        'Duration': years_follow_ups,
        'Censoring': graft_failures
    })
    df = pd.concat([df_predictive, df, df_random], axis=1)
   

    data = pd.DataFrame({
        'TC_1': TC1_values,
        #'PC_2': PC2_values, #test
        'C_1': recipient_factors,
        'C_2': donor_factors,
        #'P_1': P1_values,
        #'P_2': P2_values,
        'Duration': years_follow_ups,
        'Censoring': graft_failures
    })
    data = pd.concat([df_predictive, data, df_random], axis=1)


    return df, data