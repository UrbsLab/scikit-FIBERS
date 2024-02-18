import numpy as np
import pandas as pd
import random
import time
from sklearn.base import BaseEstimator, TransformerMixin
from .methods.data_handling import prepare_data
from .methods.data_handling import calculate_residuals
from .methods.population import BIN_SET
from .methods.util import plot_pareto
from .methods.util import plot_feature_tracking
from .methods.util import plot_kaplan_meir
from .methods.util import plot_fitness_progress
from .methods.util import plot_perform_progress
from .methods.util import plot_misc_progress
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
                    pop_size = 50, tournament_prop=0.5,crossover_prob=0.5, mutation_prob=0.1, new_gen=1.0, elitism=0.1, min_bin_size=1, max_bin_init_size=10,
                    fitness_metric="log_rank", log_rank_weighting=None, pareto_fitness=False, censor_label="Censoring", group_strata_min=0.2, penalty=0.5,
                    group_thresh=0, min_thresh=0, max_thresh=3, int_thresh=True, thresh_evolve_prob=0.5,
                    manual_bin_init=None, covariates=None, report=None, random_seed=None,verbose=False):

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
        :param elitism: proportion of pop_size that is protected from deletion each generation
        :param min_bin_size: minimum number of features to be specified within a bin
        :param max_bin_init_size: maximum number of features within initialized bins
        :param fitness_metric: the fitness metric used by FIBERS to evaluate candidate bins ['log_rank','residuals','aic']
        :param log_rank_weighting: an optional weighting of the log-rank test ['wilcoxon','tarone-ware','peto','fleming-harrington'] 
        :param pareto_fitness: boolean determining whether multi-objective pareto-front-based fitness is utilized combining fitness_metric and bin simplicity as objectives)

        #Survival Analysis Parameters:
        :param censor_label: label indicating the censoring column in the datasets (e.g. 'Censoring')
        :param group_strata_min: the minimum cuttoff for group-strata sizes (instance count) below which bins have fitness penalizaiton applied
        :param penalty: the penalty multiplier applied to the fitness of bins that go beneith the group_strata_min
        :param group_thresh: the bin sum (e.g. mismatch count) for an instance over which that instance is assigned to the above threshold group

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
        :param report: list of integers, indicating iterations where the population will be printed out for viewing
        :param random_seed: the seed value needed to generate a random number
        :param verbose: Boolean flag to run in 'verbose' mode - display run details
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
        
        if not self.check_is_float(elitism) or elitism < 0 or elitism > 1:
            raise Exception("'elitism' param must be float from 0 - 1")
        
        if not self.check_is_int(min_bin_size) or min_bin_size < 0:
            raise Exception("'min_bin_size' param must be non-negative integer (and no larger then the number of features in the dataset)")

        if not self.check_is_int(max_bin_init_size) or max_bin_init_size < 0:
            raise Exception("'max_bin_init_size' param must be non-negative integer (and no larger then the number of features in the dataset)")

        if fitness_metric!="log_rank" and fitness_metric!="residuals" and fitness_metric!="aic":
            raise Exception("'fitness_metric' param can only have values of 'log_rank', 'residuals', or 'aic'")
        
        if log_rank_weighting!="wilcoxon" and log_rank_weighting!="tarone-ware" and log_rank_weighting!="peto" and log_rank_weighting!='fleming-harrington'and log_rank_weighting != None:
            raise Exception("'log_rank_weighting' param can only have values of 'wilcoxon', 'tarone-wares', 'peto' or 'fleming-harrington'")

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
        if group_thresh != None and group_thresh < 0: 
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

        if not self.check_is_list(report) and not report == None:
            raise Exception("'report' param must be an list of positive integers or None")
        
        if not self.check_is_int(random_seed) and not random_seed == None:
            raise Exception("'random_seed' param must be an int or None")

        if not verbose == True and not verbose == False and not verbose == 'True' and not verbose == 'False':
            raise Exception("'verbose' param must be a boolean, i.e. True or False")
        
        #Initialize global variables
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.iterations = iterations
        self.pop_size = pop_size
        self.tournament_prop = tournament_prop
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob 
        self.new_gen = new_gen
        self.elitism = elitism
        self.min_bin_size = min_bin_size
        self.max_bin_init_size = max_bin_init_size
        self.fitness_metric = fitness_metric
        self.log_rank_weighting = log_rank_weighting
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
        self.report = report
        self.random_seed = random_seed
        self.verbose = verbose
        if self.covariates is None:
            self.covariates = list()

        self.hasTrained = False


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

        # Check if original_feature_matrix and y are numeric (Ryan - extend to check if all values > 0)
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
    

    def fit(self, x, y=None):
        """
        Scikit-learn required function for supervised training of FIBERS

        :param x: array-like {n_samples, n_features} training instances.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE, and can include 'covariates'
                OR array-like dataframe {n_samples, n_features} training instances that can include 'covariates'
                and with column name as given by outcome_label and censor_label when y=None.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE 

        :param y: None or list of list/tuples with (censoring, duration) {n_samples, 2} labels.
                ALL INSTANCE OUTCOMES MUST BE NUMERIC NOT NAN OR OTHER TYPE

        :return: self
        """
        self.start_time = time.time() # Record the start time
        random.seed(self.random_seed) # Set random seed

        # PREPARE DATA ---------------------------------------
        self.df = self.check_x_y(x, y)
        self.feature_df,self.outcome_df,self.censor_df,self.covariate_df = prepare_data(self.df,self.outcome_label,self.censor_label,self.covariates)

        # Calculate residuals for covariate adjustment
        if self.fitness_metric == "residuals":
            self.residuals = calculate_residuals(self.covariate_df,self.outcome_label,self.censor_label)
        else:
            self.residuals = None

        # Make feature dataframe without covariates
        self.feature_df = self.feature_df.drop(self.covariates, axis=1)

        # Creating a list of features
        self.feature_names = list(self.feature_df.columns)

        print("Beginning FIBERS Fit:")
        #Initialize bin population
        threshold_evolving = False #Adaptive thresholding - evolving thresholds is off by default for bin initialization 
        self.set = BIN_SET(self.manual_bin_init,self.feature_df,self.outcome_df,self.censor_df,self.feature_names,self.pop_size,
                           self.min_bin_size,self.max_bin_init_size,self.group_thresh,self.min_thresh,self.max_thresh,
                           self.int_thresh,self.outcome_type,self.fitness_metric,self.log_rank_weighting,self.pareto_fitness,self.group_strata_min,
                           self.outcome_label,self.censor_label,threshold_evolving,self.penalty,self.iterations,0,self.residuals,self.covariate_df,random)
        #Global fitness update
        self.set.global_fitness_update() #Exerimental

        # Update feature tracking
        self.set.update_feature_tracking(self.feature_names)

        # Initialize training performance tracking
        self.performance_tracking(True,-1)

        # Report initial population
        if self.report != None and 0 in self.report: 
            self.set.report_pop()

        #EVOLUTIONARY LEARNING ITERATIONS
        for iteration in tqdm(range(0, self.iterations)):
            if self.group_thresh == None:
                evolve = random.random()
                if self.thresh_evolve_prob > evolve:
                    threshold_evolving = True
            else:
                threshold_evolving = False

            # GENETIC ALGORITHM 
            target_offspring_count = int(self.pop_size*self.new_gen) #Determine number of offspring to generate
            while len(self.set.offspring_pop) < target_offspring_count: #Generate offspring until we hit the target number
                # Parent Selection
                parent_list = self.set.select_parent_pair(self.tournament_prop,random)

                # Generate Offspring - clone, crossover, mutation, evaluation, add to population
                self.set.generate_offspring(self.crossover_prob,self.mutation_prob,self.iterations,iteration,parent_list,self.feature_names,
                                            threshold_evolving,self.min_bin_size,self.max_bin_init_size,self.min_thresh,
                                            self.max_thresh,self.feature_df,self.outcome_df,self.censor_df,self.outcome_type,
                                            self.fitness_metric,self.log_rank_weighting,self.outcome_label,self.censor_label,self.int_thresh,
                                            self.group_thresh,self.pareto_fitness,self.group_strata_min,self.penalty,self.residuals,self.covariate_df,random)
            # Add Offspring to Population
            self.set.add_offspring_into_pop()

            #Global fitness update
            self.set.global_fitness_update() #Exerimental

            #Bin Deletion
            if iteration == self.iterations - 1: #Last iteration
                self.set.bin_deletion_deterministic(self.pop_size) # Elitism not needed
            else:
                self.set.bin_deletion_probabilistic(self.pop_size,self.elitism,random)

            # Update feature tracking
            self.set.update_feature_tracking(self.feature_names)
            
            # Training performance tracking
            self.performance_tracking(False,iteration)

            # DEBUGGING FUNCTION - view the population at specified iterations during training
            if self.report != None and iteration in self.report and iteration != 0:
                print("ITERATION: "+str(iteration))
                self.set.report_pop()    

        #Output a final population report
        if self.report != None: 
            self.set.report_pop()

        # Time keeping
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time

        print("Random Seed Check - End: "+ str(random.random()))
        print('FIBERS Run Complete!')
        print("Elapsed Time (sec): ", self.elapsed_time, "seconds")

        self.hasTrained = True
        return self


    def transform(self, x, y=None):
        """
        Scikit-learn required function for Supervised training of FIBERS

        :param x: array-like {n_samples, n_features} training instances.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE, and can include 'covariates'
                OR array-like dataframe {n_samples, n_features} training instances that can include 'covariates'
                and with column name as given by outcome_label and censor_label when y=None.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE 

        :param y: None or list of list/tuples with (censoring, duration) {n_samples, 2} labels.
                ALL INSTANCE OUTCOMES MUST BE NUMERIC NOT NAN OR OTHER TYPE
        :return: Dataset transformed instances into bin-defined features (i.e. value sums of bin-specified features) as a pd.DataFrame
        """
        if not self.hasTrained:
            raise Exception("FIBERS must be fit first")

        # PREPARE DATA ---------------------------------------
        self.df = self.check_x_y(x, y)
        self.feature_df,self.outcome_df,self.censor_df,self.covariate_df = prepare_data(self.df,self.outcome_label,self.censor_label,self.covariates)
        tdf = pd.DataFrame()

        #Create transformed dataset
        bin_count = 0
        for bin in self.set.bin_pop: #for each bin in the population - apply it to creating a bin 'feature' in the dataset for each instance
            # Sum instance values across features specified in the bin
            feature_sums = self.feature_df[bin.feature_list].sum(axis=1)
            tdf['Bin_'+str(bin_count)] = feature_sums
            bin_count += 1

        tdf = pd.concat([tdf,self.outcome_df,self.censor_df],axis=1)
        return tdf


    def predict(self, x, bin_number=None):
        """
        Function to predict strata on the basis of top OR bin.

        :param x: array-like {n_samples, n_features} training instances.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE, and can include 'covariates'
                OR array-like dataframe {n_samples, n_features} training instances that can include 'covariates'
                and with column name as given by outcome_label and censor_label when y=None.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE 

        :param bin_number: the top nth bin (0 is bin with highest fitness) to consider as the predictor, 
                or if [None] uses a bin-population weighted voting scheme as the predictor

        :return: y: prediction of group (0 or 1), e.g. strata group --> low vs. high
        """
        if not self.hasTrained:
            raise Exception("FIBERS must be trained first")

        if not self.check_is_int(bin_number) and not bin_number == None:
            raise Exception("'random_seed' param must be an int")
        
        # PREPARE DATA ---------------------------------------
        y = None
        self.df = self.check_x_y(x, y)
        if self.covariates:
            try:
                for covariate in self.covariates:
                    feature_df = feature_df.drop(columns=covariate)
            except:
                pass
        
        # Make Predition
        if bin_number != None: #Make prediction with single selected bin
            # Sum instance values across features specified in the bin
            feature_sums = self.df[self.set.bin_pop[bin_number].feature_list].sum(axis=1)
            return (self.df[self.set.bin_pop[bin_number].feature_list].sum(axis=1) > self.set.bin_pop[bin_number].group_threshold).astype(int).values
        
        else: #Make prediction using entire bin population (weighted voting scheme)
            temp_df = pd.DataFrame()
            bin_count = 0
            for bin in self.set.bin_pop: #for each bin in the population 
                # Sum instance values across features specified in the bin
                feature_sums = self.feature_df[bin.feature_list].sum(axis=1)
                temp_df['Bin_'+str(bin_count)] = feature_sums
                bin_count += 1

            # Transform values greater than 0 to 1
            #transformed_df = temp_df.applymap(lambda x: 1 if x > 0 else x)

            # Count
            bt_vote = [0]*len(temp_df) #votesum stored for each instance
            at_vote = [0]*len(temp_df) #votesum stored for each instance

            # Iterate through each row of the DataFrame
            row_count = 0
            for index, row in temp_df.iterrows():
                bin_count = 0
                # Iterate through each value in the row
                for value in row:
                    if value <= self.set.bin_pop[bin_count].group_threshold:
                        bt_vote[row_count] += self.set.bin_pop[bin_count].pre_fitness
                    else:
                        at_vote[row_count] += self.set.bin_pop[bin_count].pre_fitness
                    bin_count += 1
                row_count += 1
            # Convert votes into predictions
            prediction_list = []

            for i in range(0,len(bt_vote)):
                if bt_vote[i] < at_vote[i]:
                    prediction_list.append(1)
                else:
                    prediction_list.append(0)
            return np.array(prediction_list) 


    def performance_tracking(self,initialize,iteration):
        current_time = time.time()
        self.elapsed_time = current_time - self.start_time
        #self.set.bin_pop = sorted(self.set.bin_pop, key=lambda x: x.fitness,reverse=True)
        top_bin = self.set.bin_pop[0]
        if initialize:
            col_list = ['Iteration','Top Bin', 'Threshold', 'Fitness', 'Pre-Fitness', 'Metric', 'p-value', 'Bin Size', 'Group Ratio', 'Count At/Below Threshold', 
                        'Count Below Threshold','Birth Iteration','Residuals Score','Residuals p-value','Elapsed Time']
            self.perform_track_df = pd.DataFrame(columns=col_list)
            if self.verbose:
                print(col_list)

        tracking_values = [iteration,top_bin.feature_list,top_bin.group_threshold,top_bin.fitness,top_bin.pre_fitness,top_bin.metric,top_bin.p_value,top_bin.bin_size,
                        top_bin.group_strata_prop,top_bin.count_bt,top_bin.count_at,top_bin.birth_iteration,top_bin.residuals_score,
                        top_bin.residuals_p_value,self.elapsed_time]
        if self.verbose:
            print(tracking_values)
        # Add the row to the DataFrame
        self.perform_track_df.loc[len(self.perform_track_df)] = tracking_values


    def get_performance_tracking(self):
        return self.perform_track_df
    

    def get_top_bins(self):
        return self.set.get_all_top_bins()
    

    def report_ties(self):
        top_bin_list = self.get_top_bins()
        count = len(top_bin_list)
        if count > 1:
            print(str(len(top_bin_list))+" bins were tied for best fitness")
            for bin in top_bin_list:
                #print("Features in Bin: "+str(bin.feature_list))
                report = bin.bin_short_report()
                print(report)
        else:
            print("Only one top performing bin found")


    def get_bin_groups(self, x, y=None, bin_index=0):
        """
        Function for FIBERS that returns the variables needed to construct survival curves for the two instance 
        groups defined by a given bin (low_outcome, high_outcome, low_censor, high_censor)

        :param x: array-like {n_samples, n_features} training instances.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE, and can include 'covariates'
                OR array-like dataframe {n_samples, n_features} training instances that can include 'covariates'
                and with column name as given by outcome_label and censor_label when y=None.
                ALL INSTANCE ATTRIBUTES MUST BE >0 NUMERIC NOT NAN OR OTHER TYPE 

        :param y: None or list of list/tuples with (censoring, duration) {n_samples, 2} labels.
                ALL INSTANCE OUTCOMES MUST BE NUMERIC NOT NAN OR OTHER TYPE

        :param bin_index: population index of the bin to return group information for

        :return: low_outcome, high_outcome, low_censor, and high_censor
        """   
        if not self.hasTrained:
            raise Exception("FIBERS must be fit first")

        # PREPARE DATA ---------------------------------------
        self.df = self.check_x_y(x, y)
        self.feature_df,self.outcome_df,self.censor_df,self.covariate_df = prepare_data(self.df,self.outcome_label,self.censor_label,self.covariates)

        # Sum instance values across features specified in the bin
        feature_sums = self.feature_df[self.set.bin_pop[bin_index].feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})

        # Create evaluation dataframe including bin sum feature with 
        bin_df = pd.concat([bin_df,self.outcome_df,self.censor_df],axis=1)

        low_df = bin_df[bin_df['feature_sum'] <= self.set.bin_pop[bin_index].group_threshold]
        high_df = bin_df[bin_df['feature_sum'] > self.set.bin_pop[bin_index].group_threshold]

        low_outcome = low_df[self.outcome_label].to_list()
        high_outcome = high_df[self.outcome_label].to_list()
        low_censor = low_df[self.censor_label].to_list()
        high_censor =high_df[self.censor_label].to_list()
        return low_outcome, high_outcome, low_censor, high_censor
    

    def get_bin_report(self, bin_index):
        # Generates a bin summary report as a transposed dataframe
        return self.set.bin_pop[bin_index].bin_report().T


    def get_feature_tracking(self):
        return self.feature_names, self.set.feature_tracking
    

    def get_pop(self):
        self.set.sort_feature_lists()
        pop_df = pd.DataFrame([vars(instance) for instance in self.set.bin_pop])
        return pop_df


    def get_pareto_plot(self,show=True,save=False,output_folder=None,data_name=None):
        plot_pareto(self.set.bin_pop,show=show,save=save,output_folder=output_folder,data_name=data_name)


    def get_feature_tracking_plot(self,max_features=50,show=True,save=False,output_folder=None,data_name=None):
        feature_names, feature_tracking = self.get_feature_tracking()
        plot_feature_tracking(feature_names,feature_tracking,max_features,show=show,save=save,output_folder=output_folder,data_name=data_name)


    def get_kaplan_meir(self,data,bin_index,show=True,save=False,output_folder=None,data_name=None):
        low_outcome, high_outcome, low_censor, high_censor = self.get_bin_groups(data, bin_index)
        plot_kaplan_meir(low_outcome,low_censor,high_outcome, high_censor,show=show,save=save,output_folder=output_folder,data_name=data_name)


    def get_fitness_progress_plot(self,show=True,save=False,output_folder=None,data_name=None):
        plot_fitness_progress(self.perform_track_df,show=show,save=save,output_folder=output_folder,data_name=data_name)


    def get_perform_progress_plot(self,show=True,save=False,output_folder=None,data_name=None):
        plot_perform_progress(self.perform_track_df,show=show,save=save,output_folder=output_folder,data_name=data_name)


    def get_misc_progress_plot(self,show=True,save=False,output_folder=None,data_name=None):
        plot_misc_progress(self.perform_track_df,show=show,save=save,output_folder=output_folder,data_name=data_name)