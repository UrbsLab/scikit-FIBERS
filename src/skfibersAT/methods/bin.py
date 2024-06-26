import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import ranksums


class BIN:
    def __init__(self, feature_list, threshold=0, bin_name=None):
        self.feature_list = feature_list
        self.score = 0
        self.threshold = threshold
        self.seen = False
        self.bin_name = bin_name

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.feature_list):
            x = self.feature_list[self.index]
            self.index += 1
            return x
        else:
            raise StopIteration

    def __eq__(self, other):
        if not isinstance(other, BIN):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return sorted(self.feature_list) == sorted(other.feature_list)

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, item):
        return self.feature_list[item]

    def __str__(self):
        return str(self.feature_list)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return other.__lt__(self)

    # Setter methods of the BIN attributes

    def set_name(self, bin_name):
        self.bin_name = bin_name

    def set_score(self, score):
        self.score = score

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list

    def set_seen(self):
        self.seen = True

    def set_not_seen(self):
        self.seen = False

        # Getter methods of the BIN attributes

    def get_feature_list(self):
        return self.feature_list.copy()

    def get_threshold(self):
        return self.threshold

    def get_score(self):
        return self.score

    def get_name(self):
        return self.bin_name

    def was_seen(self):
        return self.seen

    # Log rank test used to score the different thresholds and fitness of the BIN
    def log_rank_test(self, bin_feature_matrix, label_name, duration_name,
                      informative_cutoff, threshold):
        score = 0
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]

        durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        durations_mm = df_1[duration_name].to_list()
        event_observed_mm = df_1[label_name].to_list()

        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                                   event_observed_B=event_observed_mm)
            score = results.test_statistic

        return score

    def residuals_score(self, residuals, bin_feature_matrix, label_name, duration_name,
                        informative_cutoff, threshold):
        score = 0
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]  # SPHIA
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]
        durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        durations_mm = df_1[duration_name].to_list()
        event_observed_mm = df_1[label_name].to_list()

        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            bin_residuals = residuals.loc[bin_feature_matrix[self.bin_name] <= threshold]
            bin_residuals = bin_residuals["deviance"]

            non_bin_residuals = residuals.loc[bin_feature_matrix[self.bin_name] > threshold]
            non_bin_residuals = non_bin_residuals["deviance"]

            score = abs(ranksums(bin_residuals, non_bin_residuals).statistic)
        return score

    def aic_score(self, covariate_matrix, bin_feature_matrix, label_name, duration_name,
                  informative_cutoff, threshold):
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]

        event_observed_no = df_0[label_name].to_list()
        event_observed_mm = df_1[label_name].to_list()
        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            column_values = bin_feature_matrix[self.bin_name].to_list()
            for r in range(0, len(column_values)):
                if column_values[r] > 0:
                    column_values[r] = 1
            data = covariate_matrix.copy()
            data['Bin'] = column_values
            data = data.loc[:, (data != data.iloc[0]).any()]
            cph = CoxPHFitter()
            cph.fit(data, duration_name, event_col=label_name)

            score = 0 - cph.AIC_partial_
        else:
            score = - np.inf
        return score

    # This method will update the threshold for the bin by try all the thresholds from min to max threshold
    # and uses the threshold that will get the highest score
    def try_all_thresholds(self, min_threshold, max_threshold, bin_feature_matrix,
                           label_name, duration_name, informative_cutoff, scoring_method="log_rank",
                           residuals=None, covariate_matrix=None):
        # to avoid unnecessary computations, if they have already tried all the thresholds which is determined by the
        # seen variable, then no need to try all the thresholds
        if not self.seen:
            highest_score = 0
            for threshold in range(min_threshold, max_threshold + 1):
                # Variable that will store the highest score used to determine the threshold
                if scoring_method == "log_rank":
                    score = self.log_rank_test(bin_feature_matrix, label_name,
                                               duration_name, informative_cutoff, threshold)
                elif scoring_method == "AIC":
                    score = self.aic_score(covariate_matrix, bin_feature_matrix, label_name,
                                           duration_name, informative_cutoff, threshold)
                elif scoring_method == "residuals":
                    score = self.residuals_score(residuals, bin_feature_matrix, label_name,
                                                 duration_name, informative_cutoff, threshold)
                if score > highest_score:
                    self.score = score
                    self.threshold = threshold
                    highest_score = score

            self.seen = True
