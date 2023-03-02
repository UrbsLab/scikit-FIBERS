import timeit
import unittest
from typing import Dict, Any
import pandas as pd
from skrare.methods import rvds_one_bin


def rare_and_common_variable_separation2(original_feature_matrix, label_name, rare_variant_maf_cutoff):
    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name])

    # Calculating the MAF of each feature
    maf = list(feature_df.sum() / (2 * len(feature_df.index)))

    # Creating a df of features and their MAFs
    feature_maf_df = pd.DataFrame(feature_df.columns, columns=['feature'])
    feature_maf_df['maf'] = maf

    # If the MAF of the feature is less than the cutoff, it will be designated as a rare variant
    # If the MAF of the feature is greater than or equal to the cutoff, it will be considered as a common feature
    rare_df = feature_maf_df.loc[(feature_maf_df['maf'] < rare_variant_maf_cutoff) & (feature_maf_df['maf'] > 0)]
    common_df = feature_maf_df.loc[feature_maf_df['maf'] > rare_variant_maf_cutoff]
    maf_0_df = feature_maf_df.loc[feature_maf_df['maf'] == 0]

    # Creating lists of rare and common features
    rare_feature_list = list(rare_df['feature'])
    common_feature_list = list(common_df['feature'])
    maf_0_features = list(maf_0_df['feature'])

    # Creating dictionaries of rare and common features, as the MAF of the features will be useful later
    rare_feature_maf_dict: Dict[Any, Any] = dict(zip(rare_df['feature'], rare_df['maf']))
    common_feature_maf_dict = dict(zip(common_df['feature'], common_df['maf']))

    # Creating data frames for feature matrices of rare features and common features
    rare_feature_df = feature_df[rare_feature_list]
    common_feature_df = feature_df[common_feature_list]

    # Adding the class label to each data frame
    rare_feature_df['Class'] = original_feature_matrix[label_name]
    common_feature_df['Class'] = original_feature_matrix[label_name]
    return rare_feature_list, rare_feature_maf_dict, rare_feature_df, \
        common_feature_list, common_feature_maf_dict, common_feature_df, maf_0_features


def rare_and_common_variable_separation(original_feature_matrix, label_name, rare_variant_maf_cutoff):
    # Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns=[label_name])

    # Creating a list of features
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))

    # Creating lists of rare and common features
    rare_feature_list = []
    common_feature_list = []
    maf_0_features = []

    # Creating dictionaries of rare and common features, as the MAF of the features will be useful later
    rare_feature_maf_dict = {}
    common_feature_maf_dict = {}

    # Creating an empty data frames for feature matrices of rare features and common features
    rare_feature_df = pd.DataFrame()
    common_feature_df = pd.DataFrame()

    for i in range(0, len(feature_list)):
        # If the MAF of the feature is less than the cutoff, it will be designated as a rare variant
        if (feature_df[feature_list[i]].sum() / (2 * len(feature_df.index)) < rare_variant_maf_cutoff) \
                and (feature_df[feature_list[i]].sum() / (2 * len(feature_df.index)) > 0):
            rare_feature_list.append(feature_list[i])
            rare_feature_maf_dict[feature_list[i]] = feature_df[feature_list[i]].sum() / (2 * len(feature_df.index))
            rare_feature_df[feature_list[i]] = feature_df[feature_list[i]]

        elif feature_df[feature_list[i]].sum() / (2 * len(feature_df.index)) == 0:
            maf_0_features.append(feature_list[i])

        # Otherwise, it will be considered as a common feature
        elif feature_df[feature_list[i]].sum() / (2 * len(feature_df.index)) > rare_variant_maf_cutoff:
            common_feature_list.append(feature_list[i])
            common_feature_maf_dict[feature_list[i]] = feature_df[feature_list[i]].sum() / (2 * len(feature_df.index))
            common_feature_df[feature_list[i]] = feature_df[feature_list[i]]

        # In case the MAF is exactly the cutoff
        elif feature_df[feature_list[i]].sum() / (2 * len(feature_df.index)) == rare_variant_maf_cutoff:
            common_feature_list.append(feature_list[i])
            common_feature_maf_dict[feature_list[i]] = feature_df[feature_list[i]].sum() / (2 * len(feature_df.index))
            common_feature_df[feature_list[i]] = feature_df[feature_list[i]]

    # Adding the class label to each data frame
    rare_feature_df['Class'] = original_feature_matrix[label_name]
    common_feature_df['Class'] = original_feature_matrix[label_name]
    return rare_feature_list, rare_feature_maf_dict, \
        rare_feature_df, common_feature_list, \
        common_feature_maf_dict, common_feature_df, maf_0_features


times = pd.DataFrame(columns=['RVDS', "SEP2", "SEP1"])
for i in range(0, 100):
    startsim = timeit.default_timer()
    sim_data, cutoff = rvds_one_bin(1000, 50, 10, 0.05, 'mean', 0)
    stopsim = timeit.default_timer()

    startsep2 = timeit.default_timer()
    # Running RARE and checking if 80% of predictive features are reached at each iteration
    # This will be used for Trial 2 to see how fast RARE bins when given/not given partial expert knowledge
    rare_feature_list2, rare_feature_MAF_dict2, \
        rare_feature_df2, common_feature_list2, \
        common_feature_MAF_dict2, \
        common_feature_df2, MAF_0_features2 = rare_and_common_variable_separation2(sim_data, 'Class', 0.05)
    stopsep2 = timeit.default_timer()

    startsep1 = timeit.default_timer()
    rare_feature_list, rare_feature_MAF_dict, \
        rare_feature_df, common_feature_list, \
        common_feature_MAF_dict, \
        common_feature_df, MAF_0_features = rare_and_common_variable_separation(sim_data, 'Class', 0.05)
    stopsep1 = timeit.default_timer()

    times.loc[i] = [stopsim - startsim, stopsep2 - startsep2, stopsep1 - startsep1]
    print(i)

if __name__ == '__main__':
    unittest.main()
