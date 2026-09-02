# Hyperparameters

While scikit_FIBERS has a number of available hyperparameters only a few are considered to be essential or useful to check or set. 

* Essential hyperparameters are given in the first table. 
  * For survival data without covariate columns, the *fitness_metric* should be set to 'log_rank' and *covariates* should be set to None. 
  * For survival data with covariate colums (that need to be adjusted for), the *fitness_metric* should be set to either 'residuals' or to 'log_rank_residuals' (otherwise referred to as the 'product' fitness metric), and *covariates* should be set to a list of all covariate column lables in the dataset. 
  * For survival data without censoring, *censor_label* should be set to None, otherwise it should be set the censoring column lable. 
  * While *outcome_type* is also an important hyperparameter to check, currently only the 'survival' option (for survial data analysis) has been fully implemented and tested.

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| *outcome_label* | Data column label for time-to-event (outcome) | str | 'Duration' |
| *outcome_type* | Defines the type of outcome in the dataset | 'survival','class' | 'survival' |
| *censor_label* | Data column label for censoring | str/None |'Censoring', None | 
| *fitness_metric* | Pre-fitness metric driving fitness ranking | 'log_rank','residuals','log_rank_residuals' | 'log_rank' |
| *covariates* | List of data column labels to be treated as covariates (i.e. not considered for bin inclusion) | list, None | None |

* This second table includes hyperparameters that are not essential but can have a significant impact on algorithm performance. 
  * In general, setting *iterations* and *pop_size* to larger integers is expected to improve training performance, but will require longer run times. 
  * The *group_thresh* hyperparameter controls the adaptive burden thresholding in FIBERS, where 'None' activates this mechanism, and an integer value (e.g. 0), will enforce a specific burden threshold for all discovered bins. Related to this, *max_thresh* controls the maximum burden threshold allowed in bins, assuming *group_thresh* = None. Also related to adaptive burden thresholding, the *thresh_evolve_prob* (set between 0.0 and 1.0) can lead to better performance when set closer to 1.0, but requires significantly more runtime. 
  * The *group_strata_min* hyperparameter (set as > 0.0 and < 0.5) enforces a minimum instance count balance between risk groups, where 0.5 describes two risk groups with the same instance count. 
  * The *manual_bin_init* hyperparameter allows users to load an existing set of candidate bins for FIBERS to start learning from, rather than starting from randomly initialized bins. The value of using this function depends on the quality of the loaded bins, however utilizing expert (i.e. domain) knowledge to design candidate bins before running FIBERS has the potential to dramatically improve or speed up learning in larger or more complex tasks. 
  * Lastly, *pop_clean* = 'group_strata' applies a post-hoc cleaning of the bin population, removing any remaining bins that have a risk group instance count ratio below the *group_strata_min*. 

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| *iterations* | Number of training/optimization cycles | int | 100 |
| *pop_size* | Maximum bin population size at end of each cycle | int | 50 |
| *group_thresh* | Optionally specify a group threshold for bins to use | int, None | None |
| *max_thresh* | Maximum group threshold for adaptive thresholding | int | 5 |
| *thresh_evolve_prob* | Probability that an optimization cycle will evolve vs. deterministically select a group threshold for new bin evaluation | float | 0.5 |
| *group_strata_min* | Min. cutoff for group strata sizes below which a pre-fitness penalty is applied to bin | float | 0.2 |
| *manual_bin_init* | Dataframe of FIBERS-formatted bin population used to initialize the bin population | dataframe, None | None |
| *pop_clean* | Optional bin population cleanup strategy | 'group_strata', None | None |

* The remaining hyperparameters in the table below can largely be left to their default values by most users. 

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| *tournament_prop* | Population proportion randomly selected for tournament | float | 0.2 |
| *crossover_prob* | Uniform crossover operator probability | float | 0.5 |
| *min_mutation_prob* | Minimum mutation operator probability | float | 0.1 |
| *max_mutation_prob* | Maximum mutation operator probability | float | 0.3 |
| *merge_prob* | Merge operator probability | float | 0.1 |
| *new_gen* | Proportion of *pop_size*. Determines the number of offspring to generate each generation | float | 1.0 |
| *elitism* | Proportion of *pop_size* protected from deletion | float | 0.1 |
| *diversity_pressure* | Number of bin similarity clusters driving bin deletion to encourage bin diversity | int | 3 |
| *min_bin_size* | Min. number of features to be specified in a bin | int | 1 |
| *max_bin_size* | Max. number of features to be specified in a bin | int/None | None |
| *max_bin_init_size* | Max. number of features in initialized bins | int | 10 |
| *log_rank_weightning* | Optional weighting of log-rank test | 'wilcoxon', 'tarone-ware', 'peto', 'fleming-harrington', None | None |
| *penalty* | Penalty multiplier applied to pre-fitness when bin’s group strata ratio goes below the minimum | float | 0.5 |
| *min_thresh* | Minimum group threshold for adaptive thresholding | int | 0 |
| *int_thresh* | Boolean indicating that adaptive bin thresholds are limited to positive intergers | Boolean | True |
| *random_seed* | Seed value used to generate random numbers or make random selections | int, None | None |
| *report* | List of integers, indicating iterations where the population will be printed out for viewing | list, None | None |
| *verbose* | Boolean flag to run in 'verbose' mode - display run details | Boolean | False |