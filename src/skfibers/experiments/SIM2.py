import random
import math
import pandas as pd
import numpy as np

""" Survival data simulators (SIM2) is designed to create simulated survival datasets with censoring as well as include two covariate features
    and a predictive feature (TC1) that is associated with the two covariates.  This is meant to test if FIBERS can detect the simulated
    predictive features 'P's' but ignore the TC1 feature when utilizing deviance residuals based fitness. We have tested this simulator up to 
    a ground truth threshold of 5 (which requires a minimum of 6 predictive features in order to simulate correctly). """

def survival_data_simulation_covariates(instances=10000,total_features=100,predictive_features=6,low_risk_proportion=0.5,threshold=0,
                                        feature_frequency_range=(0.1, 0.4),noise_frequency=0.0,censoring_frequency=0.2,
                                        negative_control=False,random_seed=None):
    random.seed(random_seed)

    lr_count = int(instances*low_risk_proportion)
    #hr_count = instances - lr_count

    # Identify unique binaries (to first populate predicive instance features)
    high_binary_list,low_binary_list = generate_binary_numbers(predictive_features, threshold)
    print("High Risk Unique: "+str(len(high_binary_list)))
    print("Low Risk Unique: "+str(len(low_binary_list)))

    patient_censor_prob = censoring_frequency
    random_features = total_features-predictive_features-1 #the -1 accounts for the TC1 covariate associated feature
    administrative_censoring_time = 23

    # Initialize lists to store data
    recipient_factors = []
    donor_factors = []
    pred_values = []
    TC1_values = []
    risk_values = []
    patient_censoring_times = []
    administrative_censoring_times = []
    graft_failure_times = []
    years_follow_ups = []
    graft_failures = []

    # Generate data (low risk instances first then high risk to ensure group balance)
    lr_instance_counter = 0
    hr_instance_counter = 0
    for i in range(0, instances):
        #Assign Covariate Values
        recipient_factor = random.gauss(0, 1)
        donor_factor = random.gauss(0, 1)

        #Assign feature values
        if lr_instance_counter < lr_count: # Generate low risk instance
            #Select unique low_binary_list examples first
            if lr_instance_counter < len(low_binary_list):
                feature_list = low_binary_list[lr_instance_counter]
            else: #Randomly choose instances from low_binary_list to fill in remaining low risk instances.
                feature_list = random.choice(low_binary_list)
            lr_instance_counter += 1

        else: #Generate high risk instance
            #Select unique high_binary_list examples first
            if hr_instance_counter < len(high_binary_list):
                feature_list = high_binary_list[hr_instance_counter]
            else: #Randomly choose instances from low_binary_list to fill in remaining low risk instances.
                feature_list = random.choice(high_binary_list)
            hr_instance_counter += 1

        #Convert Binary String into List of 'ints'
        feature_list = [int(char) for char in feature_list]

        #Determine value for covariate associated Feature
        TC1 = int(random.random() > recipient_factor/2 + donor_factor/2)

        risk = 0
        if sum(feature_list) > threshold: #instance belongs to high risk group
            risk = 1

        #Calculate True Graft Failure Time
        rate = math.exp(1*recipient_factor + 1*donor_factor + risk - 2)
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
        TC1_values.append(TC1)
        risk_values.append(risk)
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
        'C_1': recipient_factors,
        'C_2': donor_factors,
        'patient_censoring_time': patient_censoring_times,
        'administrative_censoring_time': administrative_censoring_times,
        'graft_failure_time': graft_failure_times,
        'Duration': years_follow_ups,
        'Censoring': graft_failures,
        'TrueRiskGroup': risk_values
    })
    df = pd.concat([df_predictive, df, df_random], axis=1)
   
    data = pd.DataFrame({
        'TC_1': TC1_values,
        'C_1': recipient_factors,
        'C_2': donor_factors,
        'Duration': years_follow_ups,
        'Censoring': graft_failures,
        'TrueRiskGroup': risk_values
    })
    data = pd.concat([df_predictive, data, df_random], axis=1)

    #Add Noise by swapping 
    if noise_frequency > 0:
        columns_to_shuffle = ['Duration','Censoring','TrueRiskGroup']

        # Calculate the number of rows to shuffle
        num_rows_to_shuffle = int(len(data) * noise_frequency *2) #noise multiplied by 2 so the degree of noise is comparable to SIM1

        # Get random indices for the rows to shuffle
        shuffle_indices = np.random.choice(df.index, size=num_rows_to_shuffle, replace=False)

        # Shuffle the values within the specified columns for these rows
        data.loc[shuffle_indices, columns_to_shuffle] = data.loc[shuffle_indices, columns_to_shuffle].sample(frac=1).values

    if negative_control:
        columns_to_shuffle = ['Duration','Censoring','TrueRiskGroup']
        for col in columns_to_shuffle:
            data[col] = np.random.permutation(data[col].values)

    return df, data


def count_ones(binary):
    return sum(int(bit) for bit in binary)


def generate_binary_numbers(predictive_features, threshold):
    high_binary_list = []
    low_binary_list = []
    unique_count = 0
    for i in range(2 ** predictive_features):
        unique_count +=1
        binary = bin(i)[2:]  # Convert the number to binary (remove '0b' prefix)
        # Ensure the binary number has n digits by padding with zeros if necessary
        padded_binary = binary.zfill(predictive_features)
        if count_ones(padded_binary) > threshold:
            high_binary_list.append(padded_binary)
        else:
            low_binary_list.append(padded_binary)
    print("Unique binary numbers: "+str(unique_count))
          
    return high_binary_list,low_binary_list