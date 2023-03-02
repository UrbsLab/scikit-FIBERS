import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skfibers.methods import fibers_algorithm


class FIBERS(BaseEstimator, TransformerMixin):
    def __init__(self, label_name="Class", duration_name="grf_yrs",
                 given_starting_point=False, amino_acid_start_point=None, amino_acid_bins_start_point=None,
                 iterations=1000, set_number_of_bins=1, min_features_per_group=1, max_number_of_groups_with_feature=1,
                 informative_cutoff=0.2, crossover_probability=0.5, mutation_probability=0.05, elitism_parameter=0.2,
                 random_seed=None, bin_size_variability_constraint=None, max_features_per_bin=None):
        """
        A Scikit-Learn compatible framework for the FIBERS Algorithm

        :param label_name: described as below
        :param duration_name: another column name in dataset, not used
        :param given_starting_point: whether or not expert knowledge is being inputted (True or False)
        :param amino_acid_start_point: if RARE is starting with expert knowledge, input the list
                of features here; otherwise None
        :param amino_acid_bins_start_point: if RARE is starting with expert knowledge, input the list of bins of
                features here; otherwise None
        :param iterations: the number of evolutionary cycles RARE will run
        :param label_name: label for the class/endpoint column in the dataset (e.g., 'Class')
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
        :param bin_size_variability_constraint: sets the max bin size of children to be n
                times the size of their sibling (recommendation: 2, with larger or smaller
                values the population would trend heavily towards small or large bins without
                exploring the search space)
        :param max_features_per_bin: sets a max value for the number of features per bin
        """

        # iterations
        self.original_feature_matrix = None

        if not self.check_is_int(iterations):
            raise Exception("iterations param must be nonnegative integer")

        if iterations < 0:
            raise Exception("iterations param must be nonnegative integer")

        # set_number_of_bins
        if not self.check_is_int(set_number_of_bins):
            raise Exception("set_number_of_bins param must be nonnegative integer")

        if set_number_of_bins < 1:
            raise Exception("set_number_of_bins param must be nonnegative integer 1 or greater")

        # min_features_per_group
        if not self.check_is_int(min_features_per_group):
            raise Exception("min_features_per_group param must be nonnegative integer")

        if min_features_per_group < 0:
            raise Exception("min_features_per_group param must be nonnegative integer")

        # max_number_of_groups_with_feature
        if not self.check_is_int(max_number_of_groups_with_feature):
            raise Exception("max_number_of_groups_with_feature param must be nonnegative integer")

        if max_number_of_groups_with_feature < 0:
            raise Exception("max_number_of_groups_with_feature param must be nonnegative integer")

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

        # elitism_parameter
        if not self.check_is_float(elitism_parameter):
            raise Exception("elitism_parameter param must be float from 0 - 1")

        if elitism_parameter < 0 or elitism_parameter > 1:
            raise Exception("elitism_parameter param must be float from 0 - 1")

        # given_starting_point
        if not (isinstance(given_starting_point, bool)):
            raise Exception("given_starting_point param must be boolean True or False")
        elif given_starting_point:
            if amino_acid_start_point is None or amino_acid_bins_start_point is None:
                raise Exception(
                    "amino_acid_start_point param and amino_acid_bins_start_point param must be a list if expert "
                    "knowledge is being inputted")
            elif not (isinstance(amino_acid_start_point, list)):
                raise Exception("amino_acid_start_point param must be a list")
            elif not (isinstance(amino_acid_bins_start_point, list)):
                raise Exception("amino_acid_bins_start_point param must be a list")

        # label_name
        if not (isinstance(label_name, str)):
            raise Exception("label_name param must be str")

        # duration_name
        if not (isinstance(duration_name, str)):
            raise Exception("duration_name param must be str")

        # bin_size_variability_constraint
        if not (bin_size_variability_constraint is None) or not self.check_is_float(bin_size_variability_constraint):
            raise Exception("bin_size_variability_constraint param must be None, an integer 1 or greater, or a float "
                            "1.0 or greater")
        if not (bin_size_variability_constraint is None) and bin_size_variability_constraint < 1:
            raise Exception("bin_size_variability_constraint is less than 1")

        # max_features_per_bin
        if not (max_features_per_bin is None):
            if not self.check_is_int(max_features_per_bin):
                raise Exception("max_features_per_bin param must be nonnegative integer")

            if min_features_per_group <= max_features_per_bin:
                raise Exception("max_features_per_bin param must be greater or equal to min_features_per_group param")

        self.given_starting_point = given_starting_point
        self.amino_acid_start_point = amino_acid_start_point
        self.amino_acid_bins_start_point = amino_acid_bins_start_point
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
        self.random_seed = random_seed
        self.bin_size_variability_constraint = bin_size_variability_constraint
        self.max_features_per_bin = max_features_per_bin

        # TODO: Implement Reboot Population method
        self.reboot_filename = None

        # Reboot Population
        if self.reboot_filename is not None:
            self.reboot_population()
            self.hasTrained = True
        else:
            self.iterationCount = 0

        self.hasTrained = False

    def reboot_population(self):
        """
        Function to Reboot Population, not Implemented

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

    def fit(self, original_feature_matrix, y=None):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param original_feature_matrix: array-like {n_samples, n_features} Training instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :param y: array-like {n_samples} Training labels.
                ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE

        :return self
        """
        # original_feature_matrix
        if not (isinstance(original_feature_matrix, pd.DataFrame)):
            raise Exception("original_feature_matrix must be pandas dataframe")

        # Check if original_feature_matrix and y are numeric
        try:
            for instance in original_feature_matrix:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)

        except Exception:
            raise Exception("X and y must be fully numeric")

        if not (self.label_name in original_feature_matrix.columns):
            raise Exception("label_name param must be a column in the dataset")

        if not (self.duration_name in original_feature_matrix.columns):
            raise Exception("duration_name param must be a column in the dataset")

        self.original_feature_matrix = original_feature_matrix

        return self

    def transform(self, original_feature_matrix, y=None):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param original_feature_matrix: array-like {n_samples, n_features} Training instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :param y: array-like {n_samples} Training labels.
                ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
        :return: self, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features
        """

        if y is not None:
            pass

        if not (self.original_feature_matrix == original_feature_matrix):
            raise Exception("X param does not match fitted matrix. Fit needs to be first called on the same matrix.")

        bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features = \
            fibers_algorithm(
                self.given_starting_point,
                self.amino_acid_start_point,
                self.amino_acid_bins_start_point,
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
                self.random_seed,
                self.bin_size_variability_constraint,
                self.max_features_per_bin)
        return self, bin_feature_matrix_internal, amino_acid_bins_internal, \
            amino_acid_bin_scores_internal, maf_0_features
