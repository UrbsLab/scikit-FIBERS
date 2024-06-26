import random
import copy
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def rare_SNP_data_simulation(instances=10000, total_features=100, predictive_features=10, class0_proportion=0.5, threshold = 0, 
                             feature_frequency_range=(0.1, 0.5), noise_frequency=0.1, covariates_to_sim=0, covariates_signal_range=(0.2,0.4),random_seed=None):
    """
    Defining a function to create an artificial dataset with parameters, there will be one ideal/strong bin
    Note: MAF (minor allele frequency) cutoff refers to the threshold
    separating rare variant features from common features

    :param instances: dataset size
    :param total_features: total number of features in dataset
    :param predictive_features: total number of predictive features in the ideal bin
    :param low_risk_proportion: the proportion of instances to be labeled as (no fail class)
    :param threshold: The threshold used to deterimine simulated high vs. low risk instance. Any bin sum higher than threshold is high risk.
    :param feature_frequency_range: the max and min freature frequency for a given column in data. (e.g. 0.1 to 0.4)
    :param noise_frequency: Value from 0 to 0.5 representing the proportion of class 0/class 1 instance pairs that \
                            have their outcome switched from 0 to 1
    :param covariates_to_sim: number of covariates to simulate - that are each partially correlated with outcome
    :param covariates_signal_range: range of values determining covariate correlation with True Risk Group
    :param random_seed:

    :return: pandas dataframe of generated data
    """
    #Set random seed if given
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    class_0_count = int(instances*class0_proportion)
    class_1_count = instances - class_0_count

    class_1_binary_list,class_0_binary_list = check_parameters(predictive_features, threshold, class_1_count, class_0_count)

    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    df = pd.DataFrame(np.zeros((instances, total_features + 2)))
    predictive_names = ["P_" + str(i + 1) for i in range(predictive_features)]
    random_names = ["R_" + str(i + 1) for i in range(total_features - predictive_features)]
    df.columns = predictive_names + random_names + ['Class']

    # Assigning class according to low_risk_proportion parameter
    class_list = [1] * class_1_count + [0] * class_0_count
    df['Class'] = class_list

    # Identify target MAF for each feature in the dataset.
    MAF_list = [random.uniform(feature_frequency_range[0], feature_frequency_range[1]) for _ in range(instances)]
    non_zero_count_list = [int(x * instances) for x in MAF_list]

    # Generate Predictive Feature Values -------------------------------
    for i in range(len(class_1_binary_list)): #for each unique binary combo for class 1
        binary_string = class_1_binary_list[i]
        for col, value in zip(predictive_names, [int(bit) for bit in binary_string]):
            df.at[i, col] = value


    #now use 0s, 1s, and 2s to fill in rest based on threshold setting
    #how to randomly pick values to ensure high threshold, but also keep HWE probs randomly preserved. 
    for i in range(len(class_1_binary_list),class_1_count): #random assignment
        one_count = random.randint(threshold+1,predictive_features) #Get random count of 1's
        sampled_cols = random.sample(predictive_names,one_count) # Get columns to add 1's
        for col in sampled_cols:
            df.at[i,col] = 1





    #can start with binary numbers but later switch 1's to 2's within initial binary configuration based on MAF (get one's count and adjust as needed to 2's to get correct counts)
    #generate MAFs for every feature in dataset
    #Calculate HWE counts for each


    # Initialize lists to store data
    data = {'Class': [], 'Sum_of_SNP': []}
    for i in range(1, num_features + 1):
        data[f'SNP_{i}'] = []

    # Simulate SNP features for each instance
    for _ in range(num_instances):
        # Assign class label
        if len(data['Class']) < num_class_1:
            data['Class'].append(1)
        else:
            data['Class'].append(0)

        # Simulate SNP values for each feature
        snp_sum = 0
        for i in range(1, num_features + 1):
            maf = random.uniform(0.01, 0.05)
            snp_value = random.choices([0, 1, 2], weights=[(1-maf)**2, 2*maf*(1-maf), maf**2])[0]
            data[f'SNP_{i}'].append(snp_value)
            snp_sum += snp_value

        # Assign sum of SNP values
        data['Sum_of_SNP'].append(snp_sum)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Assign class labels based on the sum of SNP values
    df.loc[df['Sum_of_SNP'] > threshold, 'Class'] = 1
    df.loc[df['Sum_of_SNP'] <= threshold, 'Class'] = 0

    return df



    high_binary_list,low_binary_list = check_parameters(predictive_features, threshold, hr_count, lr_count)

    # Creating an empty dataframe to use as a starting point for the eventual feature matrix
    df = pd.DataFrame(np.zeros((instances, total_features + 2)))
    predictive_names = ["P_" + str(i + 1) for i in range(predictive_features)]
    random_names = ["R_" + str(i + 1) for i in range(total_features - predictive_features)]
    df.columns = predictive_names + random_names + ['TrueRiskGroup', 'Duration']

    # Assigning class according to low_risk_proportion parameter
    class_list = [1] * hr_count + [0] * lr_count
    df['TrueRiskGroup'] = class_list

    # Identify target MAF for each feature in the dataset.
    MAF_list = [random.uniform(mm_frequency_range[0], mm_frequency_range[1]) for _ in range(instances)]
    one_count_list = [int(x * instances) for x in MAF_list]

    # Generate Predictive Feature Values -------------------------------
    # for high risk instances fill in predictive features
    for i in range(len(high_binary_list)): #for each unique binary combo for high risk
        binary_string = high_binary_list[i]
        for col, value in zip(predictive_names, [int(bit) for bit in binary_string]):
            df.at[i, col] = value

    for i in range(len(high_binary_list),hr_count): #random assignment
        one_count = random.randint(threshold+1,predictive_features) #Get random count of 1's
        sampled_cols = random.sample(predictive_names,one_count) # Get columns to add 1's
        for col in sampled_cols:
            df.at[i,col] = 1

    #for low risk instances, fill in predictive features
    for i in range(hr_count,len(low_binary_list)): #for each unique binary combo for low risk
        binary_string = low_binary_list[i]
        for col, value in zip(predictive_names, [int(bit) for bit in binary_string]):
            df.at[i, col] = value

    for i in range(hr_count+len(low_binary_list),instances): #random assignment
        one_count = random.randint(0,threshold) #Get random count of 1's
        sampled_cols = random.sample(predictive_names,one_count) # Get columns to add 1's
        for col in sampled_cols:
            df.at[i,col] = 1

    # Adjust rows to try and maintain random MAF while maintaining original risk group limitations for 1 counts.
    predictive_one_counts = one_count_list[:predictive_features]
    print("Target predictive feature(s) 'one's counts: "+str(predictive_one_counts[:predictive_features]))
    #get a list of predictive column names ordered by their target one_counts
    ordered_predictive_indexes = sorted(range(len(predictive_one_counts)), key=lambda i: predictive_one_counts[i])

    for index in ordered_predictive_indexes:
        feature_name = predictive_names[index]
        sum_of_ones = int(df[feature_name].sum())
        if sum_of_ones > predictive_one_counts[index]: #if we need to remove 1's from column
            change_count = sum_of_ones - predictive_one_counts[index]
            #get row indexes where the sum is above the thershold
            row_sum_list = []
            original_index_list = []
            for i in range(len(high_binary_list),hr_count): #go through high risk unprotected rows and get row sums
                row_sum_list.append(int(df.iloc[i][predictive_names].sum()))  # Calculate sum of values in the row
                original_index_list.append(i)
            zipped = list(zip(row_sum_list, original_index_list))

            sorted_zipped = sorted(zipped, key=lambda x: x[0],reverse=True) #row sums sorted by decreasing value
            
            sorted_indexes = [t for _, t in sorted_zipped]

            drop_list = []
            for i in sorted_indexes: #narrow sorted_indexes down to rows with a 1 value
                if int(df.at[i,feature_name]) != 1: #zero in this position
                    drop_list.append(i)
            sorted_indexes = [x for x in sorted_indexes if x not in drop_list]

            # change values in this column for these indexed rows 
            changed_count = 0
            no_qualify_count = 0
            for i in sorted_indexes:
                if changed_count < change_count:
                    if int(df.iloc[i][predictive_names].sum()) > threshold+1: 
                        #print(df[feature_name].sum())
                        df.at[i,feature_name] = 0
                        
                        changed_count += 1
                    else:
                        no_qualify_count+= 1

            #check for success
            new_sum = int(df[feature_name].sum())
            if new_sum != predictive_one_counts[index]:
                print("Warning: Feature "+str(feature_name)+"'only reduced to "+str(new_sum)+" one's count.")
                
        if sum_of_ones < predictive_one_counts[index]: #if we need to add 1's from column
            change_count = predictive_one_counts[index] - sum_of_ones
            changed_count = 0
            for i in range(len(high_binary_list),hr_count): #go through high risk unprotected rows and get row sums
                if df.at[i,feature_name] == 0:
                    if changed_count < change_count:
                        df.at[i,feature_name] = 1
                        changed_count += 1
            #check for success
            new_sum = df[feature_name].sum() 
            if new_sum != predictive_one_counts[index]:
                print("Warning: Feature "+str(feature_name)+"'only raised to "+str(new_sum)+" one's count.")


    # Generate Random Feature Values -------------------------------
    feature_index = 0
    for feature in random_names:
        one_count = one_count_list[feature_index]
        row_indexes = [random.randint(0, instances-1) for _ in range(one_count)]
        for row in row_indexes:
            df.at[row,feature] = 1
        feature_index += 1 

    # check that no random column has all 0's for low groups and all 1's for high group
    for feature in random_names:
        sum_of_LR = df.iloc[0:hr_count+1][feature].sum()
        if sum_of_LR == 0:
            print("Warning: Random feature '"+str(feature)+"' has all 0's in low risk group")

    for feature in random_names:
        sum_of_LR = df.iloc[hr_count:instances+1][feature].sum()
        if sum_of_LR == hr_count:
            print("Warning: Random feature '"+str(feature)+"' has all 1's in high risk group")

    #Final Predictive Feature Check
    final_check(df,hr_count,predictive_names,threshold,instances)

    # Assigning Gaussian according to class
    df_0 = df[df['TrueRiskGroup'] == 0].sample(frac=1).reset_index(drop=True)
    df_1 = df[df['TrueRiskGroup'] == 1].sample(frac=1).reset_index(drop=True)
    #df_0 = df[df['TrueRiskGroup'] == 0]
    #df_1 = df[df['TrueRiskGroup'] == 1]
    df_0['Duration'] = np.clip(np.random.normal(class0_time_to_event_range[0],
                                                class0_time_to_event_range[1], size=len(df_0)),
                               a_min=0, a_max=None)
    df_1['Duration'] = np.clip(np.random.normal(class1_time_to_event_range[0],
                                                class1_time_to_event_range[1], size=len(df_1)),
                               a_min=0, a_max=None)
    df = pd.concat([df_1, df_0])

    df = censor(df, censoring_frequency, random_seed)


    #df_0 = df[df['TrueRiskGroup'] == 0]
    #df_1 = df[df['TrueRiskGroup'] == 1]
    df_0 = df[df['TrueRiskGroup'] == 0].sample(frac=1).reset_index(drop=True)
    df_1 = df[df['TrueRiskGroup'] == 1].sample(frac=1).reset_index(drop=True)

    if noise_frequency > 0:
        swap_count = int(min(len(df_0), len(df_1)) * noise_frequency)
        indexes = random.sample(list(range(min(len(df_0), len(df_1)))), swap_count)
        for i in indexes:
            df_0['Censoring'].iloc[i], df_1['Censoring'].iloc[i] = \
                df_1['Censoring'].iloc[i].copy(), df_0['Censoring'].iloc[i].copy()
            df_0['Duration'].iloc[i], df_1['Duration'].iloc[i] = \
                df_1['Duration'].iloc[i].copy(), df_0['Duration'].iloc[i].copy()

    df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)
    print("Random Number Check: "+str(random.randint(0,100000)))

    return df


def count_ones(binary):
    return sum(int(bit) for bit in binary)


def generate_binary_numbers(predictive_features, threshold):
    class_1_binary_list = []
    class_0_binary_list = []
    unique_count = 0
    for i in range(2 ** predictive_features):
        unique_count +=1
        binary = bin(i)[2:]  # Convert the number to binary (remove '0b' prefix)
        # Ensure the binary number has n digits by padding with zeros if necessary
        padded_binary = binary.zfill(predictive_features)
        if count_ones(padded_binary) > threshold:
            class_1_binary_list.append(padded_binary)
        else:
            class_0_binary_list.append(padded_binary)
    print("Unique binary numbers: "+str(unique_count))
          
    return class_1_binary_list,class_0_binary_list


def check_parameters(predictive_features, threshold, class_1_count, class_0_count):
    #calculate number of  P binary combos with 1sum > threshold (high risk)
    class_1_binary_list,class_0_binary_list = generate_binary_numbers(predictive_features, threshold)
    class_1_unique = len(class_1_binary_list)
    print("Unique HR Combos: "+str(class_1_unique))
    #check that number of unique combinations isn't > (high risk instances)
    if class_1_unique > class_1_count:
        print("Warning: not enough high risk instances to include all unique predictive feature combinations")
    #calculate the number of P binary comabos with 1sum <= threshold
    class_0_unique = len(class_0_binary_list)
    print("Unique LR Combos: "+str(class_0_unique))
    #check that the number of unique combinations isnt't > (low risk instances)
    if class_0_unique > class_0_count:
        print("Warning: not enough low risk instances to include all unique predictive feature combinations")

    return class_1_binary_list,class_0_binary_list


def censor(df, censoring_frequency, random_seed=None):
    df['Censoring'] = 1
    inst_to_censor = int(censoring_frequency * len(df))
    max_duration = max(df['Duration'])
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True) #randomly shuffled
    censor_count = 0
    count = 0
    while censor_count < inst_to_censor:
        if random_seed:
            np.random.seed(random_seed + count)
        for index in range(len(df)):
            prob = df['Duration'].iloc[index] / max_duration
            choice = np.random.choice([0, 1], 1, p=[prob, 1 - prob])
            if censor_count >= inst_to_censor:
                break
            if choice == 0:
                censor_count += 1
            df['Censoring'].iloc[index] = choice
            count += 1
    return df


def final_check(df,hr_count,predictive_names,threshold,instances):
    #Final Predictive Feature Check
    lowered_check = 0
    for i in range(0,hr_count): #high risk group check
        if df.iloc[i][predictive_names].sum() <= threshold:
            lowered_check += 1
    if lowered_check > 0:
        print("Warning: this many rows lowered too much: "+str(lowered_check))

    raised_check = 0
    for i in range(hr_count,instances): #low risk group check
        if df.iloc[i][predictive_names].sum() > threshold:
            raised_check +=1
    if raised_check > 0:
        print("Warning: this many rows raised too much: "+str(raised_check))

"""
data = survival_data_simulation(instances=10000, total_features=50, predictive_features=10, low_risk_proportion=0.5, threshold = 1, mm_frequency_range=(0.1, 0.3), 
                         noise_frequency=0.0, class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2), censoring_frequency=0.5, 
                         random_seed=None, negative=False)

data.to_csv('C:/Users/ryanu/Desktop/test_sim_data.csv', index=False)
"""