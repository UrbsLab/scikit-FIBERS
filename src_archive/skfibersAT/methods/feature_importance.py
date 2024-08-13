import numpy as np
from scipy.stats import ranksums
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


def log_rank_test_feature_importance(bin_feature_matrix, amino_acid_bins, label_name, duration_name,
                                     informative_cutoff):
    bin_scores = {}
    for bin_name in amino_acid_bins.keys():

        # To not repeat calculations,
        # if this bin has been seen (meaning evaluated before by the log rank test), it will not
        # calculate the score again
        if (not amino_acid_bins[bin_name].was_seen()):
            df_0 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] <=
                                          amino_acid_bins[bin_name].get_threshold()]
            df_1 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] >
                                          amino_acid_bins[bin_name].get_threshold()]

            durations_no = df_0[duration_name].to_list()
            event_observed_no = df_0[label_name].to_list()
            durations_mm = df_1[duration_name].to_list()
            event_observed_mm = df_1[label_name].to_list()

            if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                    event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
                results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                                       event_observed_B=event_observed_mm)
                bin_scores[bin_name] = results.test_statistic
                amino_acid_bins[bin_name].set_score(results.test_statistic)
            else:
                bin_scores[bin_name] = 0
                amino_acid_bins[bin_name].set_score(0)

            amino_acid_bins[bin_name].set_seen()

        else:
            bin_scores[bin_name] = amino_acid_bins[bin_name].get_score()

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0
            amino_acid_bins[bin_name].set_score(0)
            amino_acid_bins[bin_name].set_seen()
    return bin_scores


def cox_feature_importance(bin_feature_matrix, covariate_matrix, amino_acid_bins, label_name, duration_name,
                           informative_cutoff):
    bin_scores = {}
    for bin_name in amino_acid_bins.keys():
        if not amino_acid_bins[bin_name].was_seen():
            df_0 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] <=
                                          amino_acid_bins[bin_name].get_threshold()]
            df_1 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] >
                                          amino_acid_bins[bin_name].get_threshold()]

            event_observed_no = df_0[label_name].to_list()
            event_observed_mm = df_1[label_name].to_list()
            if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                    event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
                column_values = bin_feature_matrix[bin_name].to_list()
                for r in range(0, len(column_values)):
                    if column_values[r] > 0:
                        column_values[r] = 1
                data = covariate_matrix.copy()
                data['Bin'] = column_values
                data = data.loc[:, (data != data.iloc[0]).any()]
                cph = CoxPHFitter()
                cph.fit(data, duration_name, event_col=label_name)

                bin_scores[bin_name] = 0 - cph.AIC_partial_
            else:
                bin_scores[bin_name] = -np.inf
        else:
            bin_scores[bin_name] = amino_acid_bins[bin_name].get_score()

    return bin_scores


def residuals_feature_importance(residuals, bin_feature_matrix, amino_acid_bins, label_name, duration_name,
                                 informative_cutoff):
    bin_scores = {}
    for bin_name in amino_acid_bins.keys():
        if (not amino_acid_bins[bin_name].was_seen()):
            df_0 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] <=
                                          amino_acid_bins[bin_name].get_threshold()]  # SPHIA
            df_1 = bin_feature_matrix.loc[bin_feature_matrix[bin_name] >
                                          amino_acid_bins[bin_name].get_threshold()]
            durations_no = df_0[duration_name].to_list()
            event_observed_no = df_0[label_name].to_list()
            durations_mm = df_1[duration_name].to_list()
            event_observed_mm = df_1[label_name].to_list()

            if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                    event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):

                bin_residuals = residuals.loc[bin_feature_matrix[bin_name] <=
                                              amino_acid_bins[bin_name].get_threshold()]
                bin_residuals = bin_residuals["deviance"]

                non_bin_residuals = residuals.loc[bin_feature_matrix[bin_name] >
                                                  amino_acid_bins[bin_name].get_threshold()]
                non_bin_residuals = non_bin_residuals["deviance"]

                test_statistic = abs(ranksums(bin_residuals, non_bin_residuals).statistic)

                bin_scores[bin_name] = test_statistic
                amino_acid_bins[bin_name].set_score(test_statistic)
                amino_acid_bins[bin_name].set_seen()
            else:
                bin_scores[bin_name] = 0
                amino_acid_bins[bin_name].set_score(0)
                amino_acid_bins[bin_name].set_seen()
        else:
            bin_scores[bin_name] = amino_acid_bins[bin_name].get_score()

    #         print(bin_name, bin_scores[bin_name])

    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]):
            bin_scores[i] = 0
            amino_acid_bins[i].set_score(0)
            amino_acid_bins[i].set_seen()

    return bin_scores