import random
import math
import pandas as pd
import numpy as np


#def survival_data_simulation(instances=10000, total_features=100, predictive_features=10, low_risk_proportion=0.5, threshold = 0, 
#                             feature_frequency_range=(0.1, 0.5), noise_frequency=0.0, class0_time_to_event_range=(1.5, 0.2), 
#                             class1_time_to_event_range=(1, 0.2), censoring_frequency=0.2, covariates_to_sim=0, covariates_signal_range=(0.2,0.4),random_seed=None):

# NOTE: 45.4% of donors were men and 54.6% were women, while for recipients, 59.7% were men and 40.3% were women.
# ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8875465/#:~:text=Among%2036%20666%20living%20kidney,men%20and%2040.3%25%20were%20women.
# largely consistent globally ^^

# RACE: 
# WHITE: HR (1.00, GF reference) --- Death HR: 1.00 (reference) proportion: 0.593
# AFRICAN AMERICAN: HR (1.23 GF) --- Death HR: 0.84 proportion: proportion: 0.126
# HISPANIC: HR (0.77 GF) --- Death HR: 0.68 proportion: 0.189
# ASIAN: HR (0.7 GF) --- Death HR: 0.62 proportion: 0.059
# other: proportion: 0.033
#GENDER MISMATCH
# MDMR: male donor male recipient -- (HR: 1.00) expected prop: 0.27
# FDFR: female donor female recipient (HR: 1.05) expected prop: 0.22
# MDFR: male donor female recipient (HR: 0.98) expected prop: 0.183
# FDMR: female donor male recipient (HR: 1.07) expected prop:  0.327 
# ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5383387/#:~:text=Although%20findings%20are%20contradictory%20(10,graft%20loss%20in%20unadjusted%20analyses%3B
def survival_data_simulation_categorical_covariates(instances=10000,total_features=100,predictive_features=10,feature_frequency_range=(0.1, 0.5),random_seed=None, threshold=None):

    #predictive_features = 2
    patient_censor_prob = 0.1
    random_features = total_features-predictive_features-5
    administrative_censoring_time = 23

    # Initialize lists to store data
    race_list = []
    gender_list = []
    pred_values = []

    #P1_values = []
    #P2_values = []
    RPC_values = []
    GPC_values = []
    #PC2_values = [] #test
    patient_censoring_times = []
    administrative_censoring_times = []
    graft_failure_times = []
    years_follow_ups = []
    graft_failures = []

    random.seed(random_seed)
    at = 0
    # Generate data for 10,000 observations
    for i in range(0, instances):
        #Assign Covariate Values

        white_coef = np.log(1.00)
        african_american_coef = np.log(1.3)
        hispanic_coef = np.log(0.65)
        asian_coef = np.log(0.6)
        other_coef = np.log(1.1)

        mdmr_coef = np.log(1.00)
        fdfr_coef = np.log(1.15) # previously 1.05
        mdfr_coef = np.log(0.9) # previously 0.98
        fdmr_coef = np.log(1.25) # previously 1.07

        # indicator variables
        african_american = 0
        white = 0
        hispanic = 0
        asian = 0
        other = 0
        mdmr = 0
        fdfr = 0
        mdfr = 0
        fdmr = 0

        # SET RACE COVARIATE
        rand = random.random()
        if (rand < 0.593):
            race  = 'WHITE'
            white = 1
            RPC = int(random.random() < 0.25) # more likely to be 1 if african american vs white,hispanic,asian
        elif (rand >= 0.593 and rand < 0.719):
            race = 'AFRICAN-AMERICAN'
            african_american = 1
            RPC = int(random.random() < 0.9) # more likely to be 1 if african american vs white,hispanic,asian
        elif (rand >= 0.719 and rand < 0.908):
            race = 'HISPANIC'
            hispanic = 1
            RPC = int(random.random() < 0.15) # more likely to be 1 if african american vs white,hispanic,asian
        elif (rand >= 0.908 and rand < 0.967):
            race = 'ASIAN'
            asian = 1
            RPC = int(random.random() < 0.1) # more likely to be 1 if african american vs white,hispanic,asian
        else:
            race = 'OTHER'
            other = 1
            RPC = int(random.random() < 0.25) # more likely to be 1 if african american vs white,hispanic,asian
        # SET GENDER MATCH COVARIATE
        rand = random.random()
        if (rand < 0.27):
            gender_match = 'MDMR'
            mdmr = 1
            GPC = int(random.random() < 0.2) # more likely to be 1 if female donor vs male
        elif (rand >= 0.27 and rand < 0.49):
            gender_match = 'FDFR'
            fdfr = 1
            GPC = int(random.random() < 0.8) # more likely to be 1 if female donor vs male
        elif (rand >= 0.49 and rand < 0.673):
            gender_match = 'MDFR'
            mdfr = 1
            GPC = int(random.random() < 0.1) # more likely to be 1 if female donor vs male
        else:
            gender_match = 'FDMR'
            fdmr = 1
            GPC = int(random.random() < 0.8) # more likely to be 1 if female donor vs male
        

        #Assign predictive feature values
        feature_list = []
        for j in range(0,predictive_features):
            #select feature frequency
            feature_frequency = random.uniform(feature_frequency_range[0], feature_frequency_range[1])
            feature_list.append(int(random.random() < feature_frequency))

        #P1 = int(random.random() < 0.3)
        #P2 = int(random.random() < 0.3)
            
        #PC1 = int(random.random() > 0.5 + recipient_factor/2 + donor_factor/2)
        #if random.random() > 0.2:
        #    PC1 = int(random.random() > recipient_factor/2 + donor_factor/2)
        #else:
        #    PC1 = int(random.random() > 0.5)
        
        # when PC is 1, the recipient factor and donor factor are smaller, which indicates higher survival
        #feature_frequency = random.uniform(feature_frequency_range[0], feature_frequency_range[1]) #test
        #PC2 = int(random.random() < feature_frequency) #test

        #Calculate True Graft Failure Time
        predictive_contribution = 0
        count = 0
        if threshold == None:
            for each in feature_list:
                predictive_contribution += 1*each  #0.15
            scaled_predictive_contribution = float(predictive_contribution / predictive_features)
            graft_failure_time = np.random.exponential(1) / (math.exp(
                white_coef*white + african_american_coef*african_american + hispanic_coef*hispanic + asian_coef*asian + other_coef*other
                        + mdmr*mdmr_coef + fdfr*fdfr_coef + mdfr*mdfr_coef + fdmr*fdmr_coef + 0.35*scaled_predictive_contribution - 1))
        else: 
            for each in feature_list:
                if each == 1:
                    count += 1
            if count > threshold:
                at += 1
                risk = 1
            else:
                risk = 0
            graft_failure_time = np.random.exponential(1) / (math.exp(
                white_coef*white + african_american_coef*african_american + hispanic_coef*hispanic + asian_coef*asian + other_coef*other
                        + mdmr*mdmr_coef + fdfr*fdfr_coef + mdfr*mdfr_coef + fdmr*fdmr_coef + 0.35*risk - 1))
        
        #rate = math.exp(0.2*recipient_factor - 0.3*donor_factor + 0.15*P1 + 0.15*P2- 1)
        # run cox model on generated data to verify coefficients
        #rate = math.exp(1*recipient_factor*PC2 + 1*donor_factor*PC2 + predictive_contribution - 2) #test
        # desired mean is 1/rate (the higher the rate, the lower the survival)

        #Determine Patient Cencoring Time
        if random.random() < patient_censor_prob:
            patient_censoring_time = random.expovariate(1)
        else:
            patient_censoring_time = administrative_censoring_time

        #Determine Actual Event/Censoring Time (i.e. Duration)
        years_follow_up = min(patient_censoring_time, administrative_censoring_time, graft_failure_time)
        graft_failure = int(years_follow_up < min(patient_censoring_time, administrative_censoring_time))

        #Update lists
        race_list.append(race)
        gender_list.append(gender_match)
        pred_values.append(feature_list)

        #P1_values.append(P1)
        #P2_values.append(P2)
        RPC_values.append(RPC)
        GPC_values.append(GPC)
        #PC2_values.append(PC2) #test
        patient_censoring_times.append(patient_censoring_time)
        administrative_censoring_times.append(administrative_censoring_time)
        graft_failure_times.append(graft_failure_time)
        years_follow_ups.append(years_follow_up)
        graft_failures.append(graft_failure)

    print(at)
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

    df_race = pd.get_dummies(race_list, dtype=int)
    df_race = df_race.drop(columns=['WHITE'])
    df_gender_match = pd.get_dummies(gender_list, dtype=int)
    df_gender_match = df_gender_match.drop(columns='MDMR')
    print(df_random)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'RPC': RPC_values,
        'GPC': GPC_values,
        #'PC_2': PC2_values, #test
        #'P_1': P1_values,
        #'P_2': P2_values,
        'patient_censoring_time': patient_censoring_times,
        'administrative_censoring_time': administrative_censoring_times,
        'graft_failure_time': graft_failure_times,
        'Duration': years_follow_ups,
        'Censoring': graft_failures
    })
    df = pd.concat([df_predictive, df, df_random, df_race, df_gender_match], axis=1)
   

    data = pd.DataFrame({
        'RPC': RPC_values,
        'GPC': GPC_values,
        #'PC_2': PC2_values, #test
        #'P_1': P1_values,
        #'P_2': P2_values,
        'Duration': years_follow_ups,
        'Censoring': graft_failures
    })
    data = pd.concat([df_predictive, data, df_random, df_race, df_gender_match], axis=1)


    return df, data