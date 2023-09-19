import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from .methods.algorithms import fibers_algorithm
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(font="Times New Roman")


class FIBERS(BaseEstimator, TransformerMixin):
    def __init__(self, label_name="Class", duration_name="Duration",
                 given_starting_point=False, start_point_feature_list=None, feature_bins_start_point=None,
                 iterations=1000, set_number_of_bins=50, min_features_per_group=2, max_number_of_groups_with_feature=4,
                 informative_cutoff=0.2, crossover_probability=0.5, mutation_probability=0.4, elitism_parameter=0.8,
                 mutation_strategy="Regular", random_seed=None, set_threshold=0, evolving_probability=1,
                 min_threshold=0, max_threshold=3, merge_probability=0.0, adaptable_threshold=False, covariates=None,
                 scoring_method="log_rank"):
        """
        A Scikit-Learn compatible framework for the FIBERS Algorithm.

        :param label_name: label for the class/endpoint column in the dataset (e.g., 'Class')
        :param duration_name: label to omit extra column in the dataset
        :param given_starting_point: whether or not expert knowledge is being inputted (True or False)
        :param start_point_feature_list: if FIBERS is starting with expert knowledge, input the list
               of features here; otherwise None
        :param feature_bins_start_point: if FIBERS is starting with expert knowledge, input the list of bins of
               features here; otherwise None
        :param iterations: the number of evolutionary cycles FIBERS will run
        :param set_number_of_bins: the population size of candidate bins
        :param min_features_per_group: the minimum number of features in a bin
        :param max_number_of_groups_with_feature: the maximum number of bins containing a feature
        :param crossover_probability: the probability of each feature in an offspring bin to crossover
               to the paired offspring bin (recommendation: 0.5 to 0.8)
        :param mutation_probability: the probability of each feature in a bin to be deleted (a proportionate
               probability is automatically applied on each feature outside the bin to be added
               (recommendation: 0.05 to 0.5 depending on situation and number of iterations run)
        :param elitism_parameter: the proportion of elite bins in the current generation to be
               preserved for the next evolutionary cycle (recommendation: 0.2 to 0.8
               depending on conservativeness of approach and number of iterations run)
        :param random_seed: the seed value needed to generate a random number
        :param covariates:
        :param scoring_method:
        """

        algorithm = "FIBERS"
        if algorithm not in ["FIBERS"]:
            raise Exception("Invalid Algorithm")

        if not self.check_is_int(iterations):
            raise Exception("iterations param must be non-negative integer")

        if iterations < 0:
            raise Exception("iterations param must be non-negative integer")

        # set_number_of_bins
        if not self.check_is_int(set_number_of_bins):
            raise Exception("set_number_of_bins param must be non-negative integer")

        if set_number_of_bins < 1:
            raise Exception("set_number_of_bins param must be non-negative integer 1 or greater")

        # min_features_per_group
        if not self.check_is_int(min_features_per_group):
            raise Exception("min_features_per_group param must be non-negative integer")

        if min_features_per_group < 0:
            raise Exception("min_features_per_group param must be non-negative integer")

        # max_number_of_groups_with_feature
        if not self.check_is_int(max_number_of_groups_with_feature):
            raise Exception("max_number_of_groups_with_feature param must be non-negative integer")

        if max_number_of_groups_with_feature < 0:
            raise Exception("max_number_of_groups_with_feature param must be non-negative integer")

        if max_number_of_groups_with_feature > set_number_of_bins:
            raise Exception(
                "max_number_of_groups_with_feature must be less than or equal to population size of candidate bins")

        # informative_cutoff
        if not self.check_is_float(informative_cutoff):
            raise Exception("informative_cutoff param must be float from 0 - 0.5")

        if informative_cutoff < 0 or informative_cutoff > 0.5:
            raise Exception("informative_cutoff param must be float from 0 - 0.5")

        # crossover_probability
        if not self.check_is_float(crossover_probability):
            raise Exception("crossover_probability param must be float from 0 - 1")

        if crossover_probability < 0 or crossover_probability > 1:
            raise Exception("crossover_probability param must be float from 0 - 1")

        # mutation_probability
        if not self.check_is_float(mutation_probability):
            raise Exception("mutation_probability param must be float from 0 - 1")

        if mutation_probability < 0 or mutation_probability > 1:
            raise Exception("mutation_probability param must be float from 0 - 1")

        # merge probability
        if merge_probability < 0 or merge_probability > 1:
            raise Exception("merge_probability param must be float from 0 - 1")

        # elitism_parameter
        if not self.check_is_float(elitism_parameter):
            raise Exception("elitism_parameter param must be float from 0 - 1")

        if elitism_parameter < 0 or elitism_parameter > 1:
            raise Exception("elitism_parameter param must be float from 0 - 1")

        # given_starting_point
        if not (isinstance(given_starting_point, bool)):
            raise Exception("given_starting_point param must be boolean True or False")
        elif given_starting_point:
            if start_point_feature_list is None or feature_bins_start_point is None:
                raise Exception(
                    "amino_acid_start_point param and amino_acid_bins_start_point param must be a list if expert "
                    "knowledge is being inputted")
            elif not (isinstance(start_point_feature_list, list)):
                raise Exception("amino_acid_start_point param must be a list")
            elif not (isinstance(feature_bins_start_point, list)):
                raise Exception("amino_acid_bins_start_point param must be a list")

        # label_name
        if not (isinstance(label_name, str)):
            raise Exception("label_name param must be str")

        # threshold
        if set_threshold < 0:
            raise Exception("threshold param must not be negative")

        # min and max threshold
        if max_threshold < min_threshold:
            raise Exception("min threshold must be less than or equal to max_threshold")

        self.algorithm = algorithm
        self.given_starting_point = given_starting_point
        self.start_point_feature_list = start_point_feature_list
        self.feature_bins_start_point = feature_bins_start_point
        self.iterations = iterations
        self.label_name = label_name
        self.duration_name = duration_name
        self.set_number_of_bins = set_number_of_bins
        self.min_features_per_group = min_features_per_group
        self.max_number_of_groups_with_feature = max_number_of_groups_with_feature
        self.informative_cutoff = informative_cutoff
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_parameter = elitism_parameter
        self.mutation_strategy = mutation_strategy
        self.random_seed = random_seed
        self.reboot_filename = None
        self.original_feature_matrix = None
        self.threshold = set_threshold
        self.evolving_probability = evolving_probability
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.merge_probability = merge_probability
        self.adaptable_threshold = adaptable_threshold
        if covariates is None:
            covariates = list()
        self.covariates = covariates
        self.scoring_method = scoring_method

        # Reboot Population
        if self.reboot_filename is not None:
            self.reboot_population()
            self.hasTrained = True
        else:
            self.iterationCount = 0

        self.hasTrained = False
        self.bin_feature_matrix = None
        self.bins = None
        self.bin_scores = None
        self.maf_0_features = None

    def reboot_population(self):
        """
        Function to Reboot Population, not Implemented
        :meta private:
        """
        raise NotImplementedError

    @staticmethod
    def check_is_int(num):
        """
        :meta private:
        """
        return isinstance(num, int)

    @staticmethod
    def check_is_float(num):
        """
        :meta private:
        """
        return isinstance(num, float)

    def check_x_y(self, x, y):
        """
        Function to check if x and y input to fit are valid.
        Functionality to support input as both just X as a dataframe
        similar to lifelines package
        x and y similar to scikit survival.

        :meta private:
        """
        if y is None:
            if not (isinstance(x, pd.DataFrame)):
                raise Exception("x must be pandas dataframe")
            if not ((self.label_name in x.columns) or (self.duration_name not in x.columns)):
                raise Exception("x must have column labels as specified")
            original_feature_matrix = x
        else:
            if not (isinstance(x, pd.DataFrame)):
                raise Exception("x must be pandas dataframe")
            if not ((self.label_name in x.columns) or (self.duration_name not in x.columns)):
                labels = pd.DataFrame(y, columns=[self.label_name, self.duration_name])
                original_feature_matrix = pd.concat([x, labels], axis=1)
            else:
                original_feature_matrix = x

        # Check if original_feature_matrix and y are numeric
        try:
            original_feature_matrix.copy() \
                .apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
        except Exception:
            raise Exception("X must be fully numeric")

        if not (self.label_name in original_feature_matrix.columns):
            raise Exception("label_name param must be a column in the dataset")

        if not (self.duration_name in original_feature_matrix.columns):
            raise Exception("duration_name param must be a column in the dataset")

        return original_feature_matrix

    def fit(self, x, y=None):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param x: array-like {n_samples, n_features} Training instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
                OR array-like dataframe {n_samples, n_features} Training instances
                with column name as given in label_name and duration_name.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN when y=None

        :param y: None or list of list/tuples with (censoring, duration) {n_samples, 2} labels.
                ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE

        :return: self
        """
        original_feature_matrix = self.check_x_y(x, y)
        if self.algorithm == "FIBERS":
            self.fibers_fit(original_feature_matrix)
            return self
        else:
            raise Exception("Unknown Algorithm")

    def fibers_fit(self, original_feature_matrix):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param original_feature_matrix: array-like {n_samples, n_features} Training instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :return: self, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features
        """
        self.original_feature_matrix = original_feature_matrix

        bin_feature_matrix_internal, bins_internal, \
            bin_scores_internal, maf_0_features = \
            fibers_algorithm(
                self.given_starting_point,
                self.start_point_feature_list,
                self.feature_bins_start_point,
                self.iterations,
                self.original_feature_matrix,
                self.label_name,
                self.duration_name,
                self.set_number_of_bins,
                self.min_features_per_group,
                self.max_number_of_groups_with_feature,
                self.informative_cutoff,
                self.crossover_probability,
                self.mutation_probability,
                self.elitism_parameter,
                self.mutation_strategy,
                self.random_seed,
                self.threshold,
                self.evolving_probability,
                self.max_threshold,
                self.min_threshold,
                self.merge_probability,
                self.adaptable_threshold,
                self.covariates,
                self.scoring_method,
            )
        self.bin_feature_matrix = bin_feature_matrix_internal
        self.bins = bins_internal
        self.bin_scores = bin_scores_internal
        self.maf_0_features = maf_0_features
        self.hasTrained = True
        return self

    def transform(self, x):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param x: array-like {n_samples, n_features} Transform instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN

        :return: Top Feature Bin Features as a pd.DataFrame
        """
        # x_new = self.check_x_y(x, y)

        if not self.hasTrained:
            raise Exception("Model must be fit first")

        # if not (self.original_feature_matrix.equals(original_feature_matrix)):
        #     raise Exception("X param does not match fitted matrix. Fit needs to be first called on the same matrix.")

        if self.algorithm == "FIBERS":
            sorted_bin_scores = dict(sorted(self.bin_scores.items(), key=lambda item: item[1], reverse=True))
            sorted_bin_list = list(sorted_bin_scores.keys())
            tdf = pd.DataFrame()
            try:
                for i in range(len(sorted_bin_list)):
                    tdf[sorted_bin_list[i]] = x[self.bins[sorted_bin_list[i]]].sum(axis=1)
            except Exception as e:
                print(e)
                raise Exception("Bin Feature not present in dataset")
            return tdf
        else:
            raise Exception("Unknown Algorithm")

    def predict(self, x):
        """
        Function to predict risk on the basis of top OR bin.

        :param x: array-like {n_samples, n_features} Transform instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :return: y: prediction of risk stratification
        """
        if not self.hasTrained:
            raise Exception("Model must be trained first")
        _, _, _, _, top_bin = self.get_duration_event(bin_order=0)
        top_or_rule = self.bins[top_bin]
        # check each column if 
        return (x[top_or_rule].sum(axis=1) > self.bins[top_bin].get_threshold()).astype(int)

    def score(self, x, y):
        """

        :param x: array-like {n_samples, n_features} Transform instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :param y: True Risk Group, y_true
        :return: accuracy score of the risk stratification.
        """
        if not self.hasTrained:
            raise Exception("Model must be fit first")
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)

    def classification_report(self, x, y, prin=False, save=None):
        """
        :meta private:
        """
        y_pred = self.predict(x)
        report = classification_report(y, y_pred)
        if prin:
            print(report)
        if save:
            report = classification_report(y, y_pred, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(save)
        return report

    def get_duration_event(self, bin_order=0):
        """
        :meta private:
        """
        if not self.hasTrained:
            raise Exception("Model must be fit first")
        # Ordering the bin scores from best to worst
        # sorted_bin_scores = dict(sorted(self.bin_scores.items(), key=lambda item: item[1], reverse=True))

        sorted_bin_list = dict(sorted(self.bins.items(), key=lambda item: item[1], reverse=True))

        threshold = list(sorted_bin_list.values())[bin_order].get_threshold()

        sorted_bin_list = list(sorted_bin_list.keys())

        top_bin = sorted_bin_list[bin_order]

        df_0 = self.bin_feature_matrix.loc[self.bin_feature_matrix[top_bin] <= threshold]  # SPHIA
        df_1 = self.bin_feature_matrix.loc[self.bin_feature_matrix[top_bin] > threshold]

        durations_no = df_0[self.duration_name].to_list()
        event_observed_no = df_0[self.label_name].to_list()
        durations_mm = df_1[self.duration_name].to_list()
        event_observed_mm = df_1[self.label_name].to_list()
        return durations_no, durations_mm, event_observed_no, event_observed_mm, top_bin

    def get_bin_summary(self, prin=False, save=None, bin_order=0):
        """
        Function to print statistics summary of given bin

        :param prin: flag to print statistics summary
        :param save: filename to save statistics to or None to skip (default=None)
        :param bin_order: bin index in sorted bins by log rank score (starts from 0, default=0)
        :return: log_rank_results, summary_dataframe
        """
        if not self.hasTrained:
            raise Exception("Model must be trained first")
        # Ordering the bin scores from best to worst
        durations_no, durations_mm, event_observed_no, event_observed_mm, top_bin = self.get_duration_event(bin_order)
        results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                               event_observed_B=event_observed_mm)
        columns = ["Bin #", "Top Bin of Features:", "Log-Rank Score",
                   "Number of Instances with No Mismatches in Bin:",
                   "Number of Instances with Mismatch(es) in Bin:", "p-value from Log Rank Test:", "Threshold"]
        pdf = pd.DataFrame([[top_bin, self.bins[top_bin],
                             self.bin_scores[top_bin], len(durations_no),
                             len(durations_mm), results.p_value, self.bins[top_bin].get_threshold()]],
                           columns=columns).T  # SPHIA
        if prin or save is not None:
            if prin:
                print(pdf)
            if save:
                pdf.to_csv(save)
        return results, pdf

    def get_bin_survival_plot(self, show=False, save=None, bin_order=0):
        """
        Function to plot Kaplan Meier Survival Plot

        :param show: flag to show plot
        :param save: filename to save plot to or None to skip (default=None)
        :param bin_order: bin_order: bin index in sorted bins by log rank score (starts from 0, default=0)
        :return: None
        """

        kmf1 = KaplanMeierFitter()

        durations_no, durations_mm, event_observed_no, event_observed_mm, top_bin = self.get_duration_event(bin_order)

        # fit the model for 1st cohort
        kmf1.fit(durations_no, event_observed_no, label='No Mismatches in Bin')
        a1 = kmf1.plot_survival_function()
        a1.set_ylabel('Survival Probability')

        # fit the model for 2nd cohort
        kmf1.fit(durations_mm, event_observed_mm, label='Mismatch(es) in Bin')
        kmf1.plot_survival_function(ax=a1)
        a1.set_xlabel('Time After Event')
        if show:
            plt.show()
        if save:
            plt.savefig(save, dpi=1200, bbox_inches="tight")
            plt.close()

    def get_bin_scores(self, save=None):
        """
        Function to get all bins and their corresponding log-rank scores

        :param save: filename to save bins to or None to skip (default=None)
        :return: pd.DataFrame of Bins and their corresponding log-rank scores.
        """
        bin_scores_sorted = sorted(self.bin_scores.items(),
                                   key=lambda x: x[1], reverse=True)
        bins_sorted = sorted(self.bins.items(),
                             key=lambda x: len(x[1]), reverse=True)

        tdf1 = pd.DataFrame(bin_scores_sorted, columns=['Bin #', 'Score'])
        tdf2 = pd.DataFrame(bins_sorted, columns=['Bin #', 'Bins'])

        tdf3 = tdf1.merge(tdf2, on='Bin #', how='inner', suffixes=('_1', '_2'))

        tdf3['Threshold'] = tdf3.apply(lambda x: x['Bins'].get_threshold(), axis=1)  # Added column for threshold SPHIA

        if save:
            tdf3.to_csv(save)
        return tdf3

    def print_bins(self, save=None):
        if save:
            self.bins.to_csv(save)

        return self.bin_feature_matrix
