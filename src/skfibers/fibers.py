#import numpy as np
import pandas as pd
import random

from sklearn.base import BaseEstimator, TransformerMixin
from .methods.data_handling import prepare_data
from .methods.data_handling import calculate_residuals
from .methods.population import BIN_SET

from tqdm import tqdm

#from sklearn.metrics import classification_report, accuracy_score
#from lifelines.statistics import logrank_test
#from lifelines import KaplanMeierFitter
#from .methods.algorithms import fibers_algorithm
#from matplotlib import pyplot as plt
#import seaborn as sns
#sns.set_theme(font="Times New Roman")

class FIBERS(BaseEstimator, TransformerMixin):
    def __init__(self, outcome_label="Duration", outcome_type="survival",iterations=1000,
                    pop_size = 50, tournament_prop=0.5,crossover_prob=0.5, mutation_prob=0.1, new_gen=1.0, min_bin_size=1, max_bin_init_size=10,
                    fitness_metric="log_rank", pareto_fitness=False, censor_label="Censoring", group_strata_min=0.2, penalty=0.5,
                    group_thresh=0, min_thresh=0, max_thresh=3, int_thresh=True, thresh_evolve_prob=0.5,
                    manual_bin_init=None, covariates=None, random_seed=None):

        """
        A Scikit-Learn compatible implementation of the FIBERS Algorithm.
        #General Parameters:
        :param outcome_label: label indicating the outcome column in the dataset (e.g. 'SurvivalTime', 'Class')
        :param outcome_type: defines the type of outcome in the dataset ['survival','class']
        :param iterations: the number of evolutionary cycles FIBERS will run
        :param pop_size: the maximum bin population size
        :param tournament_prop: the proportion of the popultion randomly selected for each parent pair selection
        :param crossover_prob: the probability of each specified feature in a pair of offspring bins to swap between bins
        :param mutation_prob: the probability of further offspring bin modification (i.e. feature addition, removal or swap)
        :param new_gen: proportion that determines the number of offspring generated each iteration based new_gen*pop_size
        :param min_bin_size: minimum number of features to be specified within a bin
        :param max_bin_init_size: maximum number of features within initialized bins
        :param fitness_metric: the fitness metric used by FIBERS to evaluate candidate bins ['log_rank','residuals','aic']
        :param pareto_fitness: boolean determining whether multi-objective pareto-front-based fitness is utilized combining fitness_metric and bin simplicity as objectives)

        #Survival Analysis Parameters:
        :param censor_label: label indicating the censoring column in the datasets (e.g. 'Censoring')
        :param group_strata_min: the minimum cuttoff for risk group sizes (instance count) below which bins have fitness penalizaiton applied
        :param penalty: the penalty multiplier applied to the fitness of bins that go beneith the group_strata_min
        :param group_thresh: the bin sum (e.g. mismatch count) for an instance over which that instance is assigned to the high-risk group

        #Adaptive Bin Threshold Parameters:
        :param min_thresh: for adaptive bin thresholding - the minimum group_thresh allowed
        :param max_thresh: for adaptive bin thresholding - the maximum group_thresh allowed
        :param int_thresh: boolean indicating that adaptive bin thresholds are limited to positive intergers
        :param thresh_evolve_prob: probability that adaptive bin thresholding will evolve vs. be selected for the bin deterministically

        #Manual Bin Initialization Parameters:
        :param manual_bin_init: a dictionary giving bin-name:feature list to manually initialize the bin populaExceptiontion with a specific population of bins

        #Covariate Adjustment Parameters:
        :param covariates: list of feature names in the data to be treated as covariates (not included in binning)

        #Other Parameters
        :param random_seed: the seed value needed to generate a random number
        """
        #Basic run parameter checks
        if not isinstance(outcome_label,str):
            raise Exception("'outcome_label' param must be a str")
        
        if outcome_type!="survival" and not outcome_type!="class":
            raise Exception("'outcome_type' param can only have values of 'survival' or 'class'")
        
        if not self.check_is_int(iterations) or iterations < 0:
            raise Exception("'iterations' param must be a non-negative integer")

        if not self.check_is_int(pop_size) or pop_size < 10:
            raise Exception("'pop_size' param must be non-negative integer larger than 10")

        if not self.check_is_float(tournament_prop) or tournament_prop < 0 or tournament_prop > 1:
            raise Exception("'tournament_prop' param must be float from 0 - 1")

        if not self.check_is_float(crossover_prob) or crossover_prob < 0 or crossover_prob > 1:
            raise Exception("'crossover_prob' param must be float from 0 - 1")

        if not self.check_is_float(mutation_prob) or mutation_prob < 0 or mutation_prob > 1:
            raise Exception("'mutation_prob' param must be float from 0 - 1")

        if not self.check_is_float(new_gen) or new_gen < 0 or new_gen > 1:
            raise Exception("'new_gen' param must be float from 0 - 1")
        
        if not self.check_is_int(min_bin_size) or min_bin_size < 0:
            raise Exception("'min_bin_size' param must be non-negative integer (and no larger then the number of features in the dataset)")

        if not self.check_is_int(max_bin_init_size) or max_bin_init_size < 0:
            raise Exception("'max_bin_init_size' param must be non-negative integer (and no larger then the number of features in the dataset)")

        if fitness_metric!="log_rank" and fitness_metric!="residuals" and fitness_metric!="aic":
            raise Exception("'fitness_metric' param can only have values of 'log_rank', 'residuals', or 'aic'")
        
        if fitness_metric == "residuals" or fitness_metric == "aic":
            if covariates == None:
                raise Exception("list of covariates must be specified when fitness_metric is 'residuals' or 'aic'")

        if not pareto_fitness == True and not pareto_fitness == False and not pareto_fitness == 'True' and not pareto_fitness == 'False':
            raise Exception("'pareto_fitness' param must be a boolean, i.e. True or False")

        if not isinstance(censor_label,str) and censor_label != None:
            raise Exception("'censor_label' param must be a str or None")
        
        if not self.check_is_float(group_strata_min) or group_strata_min < 0 or group_strata_min > 0.5:
            raise Exception("'group_strata_min' param must be float from 0 - 0.5")

        if not self.check_is_float(penalty) and not self.check_is_int(penalty):
            raise Exception("'penalty' param must be an int or float from 0 - 1")
        if penalty < 0 or penalty > 1:
            raise Exception("'penalty' param must be an int or float from 0 - 1")

        if not self.check_is_int(group_thresh) and not self.check_is_float(group_thresh) and group_thresh != None:
            raise Exception("'group_thresh' param must be a non-negative int or float, or None, for adaptive thresholding")
        if group_thresh < 0: 
            raise Exception("'group_thresh' param must be a non-negative int or float, or None, for adaptive thresholding")
        
        if not self.check_is_int(min_thresh) and not self.check_is_float(min_thresh) or min_thresh < 0:
            raise Exception("'min_thresh' param must be a non-negative int or float")

        if not self.check_is_int(max_thresh) and not self.check_is_float(max_thresh) or max_thresh < 0 or max_thresh <= min_thresh:
            raise Exception("'max_thresh' param must be a non-negative int or float")
        if max_thresh <= min_thresh:
            raise Exception("'max_thresh' param must be larger than min_thresh param")
        
        if not int_thresh == True and not int_thresh == False and not int_thresh == 'True' and not int_thresh == 'False':
            raise Exception("'int_thresh' param must be a boolean, i.e. True or False")

        if not self.check_is_float(thresh_evolve_prob) and not self.check_is_int(thresh_evolve_prob):
            raise Exception("'thresh_evolve_prob' param must be an int or float from 0 - 1")
        if thresh_evolve_prob < 0 or thresh_evolve_prob > 1:
            raise Exception("'thresh_evolve_prob' param must be an int or float from 0 - 1")
        
        if not self.check_is_list(manual_bin_init) and not manual_bin_init == None:
            raise Exception("'manual_bin_init' param must be either None or a list of feature name lists")
        
        if not self.check_is_list(covariates) and not covariates == None:
                raise Exception("'covariates' param must be either None or a list of feature names")

        if not self.check_is_int(random_seed) and not random_seed == None:
            raise Exception("'random_seed' param must be an int")
        
        #Initialize global variables
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.iterations = iterations
        self.pop_size = pop_size
        self.tournament_prop = tournament_prop
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob 
        self.new_gen = new_gen
        self.min_bin_size = min_bin_size
        self.max_bin_init_size = max_bin_init_size
        self.fitness_metric = fitness_metric
        self.pareto_fitness = pareto_fitness
        self.censor_label = censor_label
        self.group_strata_min = group_strata_min
        self.penalty = penalty
        self.group_thresh = group_thresh
        self.min_thresh = min_thresh 
        self.max_thresh = max_thresh 
        self.int_thresh = int_thresh
        self.thresh_evolve_prob = thresh_evolve_prob
        self.manual_bin_init = manual_bin_init
        self.covariates = covariates
        self.random_seed = random_seed
        if self.covariates is None:
            self.covariates = list()

        #self.hasTrained = False
        #self.bin_feature_matrix = None
        #self.bins = None
        #self.bin_scores = None
        #self.maf_0_features = None

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

    @staticmethod
    def check_is_list(num):
        """
        :meta private:
        """
        return isinstance(num, list)
 
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
        
        self.df = self.check_x_y(x, y)
        print("Data Shape: "+str(self.df.shape))

        # PREPARE DATA ---------------------------------------
        self.feature_df,self.outcome_df,self.censor_df,self.covariate_df = prepare_data(self.df,self.outcome_label,self.censor_label,self.covariates)

        # Calculate residuals for covariate adjustment
        if self.fitness_metric == "residuals":
            self.residuals = calculate_residuals(self.covariate_df,self.outcome_label,self.censor_label)
        else:
            self.residuals = None

        # Make feature dataframe without covariates
        self.feature_df = self.feature_df.drop(self.covariates, axis=1)
        print("Feature Data Shape: "+str(self.feature_df.shape))

        # Creating a list of features
        self.feature_names = list(self.feature_df.columns)
        print(len(self.feature_names))


        #Initialize bin population
        threshold_evolving = False #Adaptive thresholding - evolving thresholds is off by default for bin initialization 
        self.set = BIN_SET(self.manual_bin_init,self.feature_df,self.outcome_df,self.censor_df,self.feature_names,self.pop_size,
                           self.min_bin_size,self.max_bin_init_size,self.group_thresh,self.min_thresh,self.max_thresh,
                           self.int_thresh,self.outcome_type,self.fitness_metric,self.pareto_fitness,self.group_strata_min,
                           self.outcome_label,self.censor_label,threshold_evolving,self.penalty,self.random_seed)
        self.set.report_pop()

        #EVOLUTIONARY LEARNING ITERATIONS
        random.seed(self.random_seed)  # You can change the seed value as desired
        for iteration in tqdm(range(0, self.iterations)):
            if self.group_thresh == None:
                evolve = random.random()
                if self.thresh_evolve_prob > evolve:
                    threshold_evolving = True
            else:
                threshold_evolving = False

            # GENETIC ALGORITHM 
            target_offspring_count = int(self.pop_size*self.new_gen) #Determine number of offspring to generate
            while len(self.set.bin_pop) < self.pop_size + target_offspring_count: #Generate offspring until we hit the target number
                print(len(self.set.bin_pop))
                # Parent Selection
                parent_list = self.set.select_parent_pair(self.tournament_prop,self.random_seed)

                # Generate Offspring - clone, crossover, mutation, evaluation, add to population
                self.set.generate_offspring(self.crossover_prob,self.mutation_prob,iteration,parent_list,self.feature_names,
                                            threshold_evolving,self.min_bin_size,self.max_bin_init_size,self.min_thresh,
                                            self.max_thresh,self.feature_df,self.outcome_df,self.censor_df,self.outcome_type,
                                            self.fitness_metric,self.outcome_label,self.censor_label,self.int_thresh,self.group_thresh,self.random_seed)
            #Bin Deletion
            if iteration == self.iterations - 1: #Last iteration
            
            else:
                self.set.bin_deletion_probabilistic(self.pop_size)
        
        #Final bin population fitness evaluation (using deterministic adaptive thresholding)
        #last iteration deletion should be deterministic

        #add code to generate performance tracking of top bin over iterations

        print('Made it')
   
        """
        bin_feature_matrix_internal, bins_internal, \
            bin_scores_internal, maf_0_features = \
            fibers_algorithm(
                self.given_starting_point,
                self.start_point_feature_list,
                self.feature_bins_start_point,
                self.iterations,
                self.original_data,
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
        """

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
                if not ((self.outcome_label in x.columns) or (self.censor_label not in x.columns)):
                    raise Exception("x must have column labels as specified")
                feature_df = x
            else:
                if not (isinstance(x, pd.DataFrame)):
                    raise Exception("x must be pandas dataframe")
                if not ((self.outcome_label in x.columns) or (self.censor_label not in x.columns)):
                    labels = pd.DataFrame(y, columns=[self.outcome_label, self.censor_label])
                    feature_df = pd.concat([x, labels], axis=1)
                else:
                    feature_df = x

            # Check if original_feature_matrix and y are numeric
            try:
                feature_df.copy() \
                    .apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
            except Exception:
                raise Exception("X must be fully numeric")

            if not (self.outcome_label in feature_df.columns):
                raise Exception("label_name param must be a column in the dataset")

            if not (self.censor_label in feature_df.columns):
                raise Exception("duration_name param must be a column in the dataset")

            return feature_df


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

        # if not (self.original_data.equals(original_feature_matrix)):
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

    def predict(self, x, bin_order=0):
        """
        Function to predict risk on the basis of top OR bin.

        :param x: array-like {n_samples, n_features} Transform instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :param bin_order: the top nth bin to consider
        :return: y: prediction of risk stratification
        """
        if not self.hasTrained:
            raise Exception("Model must be trained first")
        _, _, _, _, top_bin = self.get_duration_event(bin_order)
        top_or_rule = self.bins[top_bin]
        # check each column if 
        return (x[top_or_rule].sum(axis=1) > self.bins[top_bin].get_threshold()).astype(int)

    def score(self, x, y, bin_order=0):
        """

        :param x: array-like {n_samples, n_features} Transform instances.
                ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        :param y: True Risk Group, y_true
        :param bin_order use top nth bin
        :return: accuracy score of the risk stratification.
        """
        if not self.hasTrained:
            raise Exception("Model must be fit first")
        y_pred = self.predict(x, bin_order)
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

        bin_name_list, duration_mm_list, duration_no_list = list(), list(), list()

        for i in range(len(bins_sorted)):
            durations_no, durations_mm, \
                event_observed_no, event_observed_mm, top_bin = self.get_duration_event(i)
            duration_mm_list.append(len(durations_mm))
            duration_no_list.append(len(durations_no))
            bin_name_list.append(top_bin)
        duration_mm_list = np.array(duration_mm_list)
        duration_no_list = np.array(duration_no_list)
        high_risk_ratio = list(duration_mm_list/(duration_no_list+duration_mm_list))
        high_risk_tuple = list(zip(bin_name_list, high_risk_ratio))
        tdf1 = pd.DataFrame(bin_scores_sorted, columns=['Bin #', 'Score'])
        tdf2 = pd.DataFrame(bins_sorted, columns=['Bin #', 'Bins'])

        tdf3 = tdf1.merge(tdf2, on='Bin #', how='inner', suffixes=('_1', '_2'))

        tdf3['Threshold'] = tdf3.apply(lambda x: x['Bins'].get_threshold(), axis=1)  # Added column for threshold SPHIA

        tdf4 = pd.DataFrame(high_risk_tuple, columns=['Bin #', 'High Risk Ratio'])

        tdf5 = tdf3.merge(tdf4, on='Bin #', how='inner', suffixes=('_1', '_2'))

        if save:
            tdf5.to_csv(save)
        return tdf5

    def print_bins(self, save=None):
        if save:
            self.bins.to_csv(save)

        return self.bin_feature_matrix
