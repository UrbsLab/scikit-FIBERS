# scikit-FIBERS 2.0

**Table of contents:**
 - [Introduction](#item-one)
 - [Installation](#item-two)
 - [Input Data](#item-three)
 - [Using scikit-FIBERS](#item-four)
 - [Hyperparameters](#item-five)
 - [Algorithm History](#item-six)
 - [Citing scikit-FIBERS](#item-seven)
 - [Futher Documentation](#item-eight)
 - [Contact](#item-nine)
 - [Acknowledgements](#item-ten)

<!-- headings -->
<a id="item-one"></a>
## Introduction
**FIBERS (Feature Inclusion Bin Evolver for Risk Stratification)** is an evolutionary machine learning algorithm designed for modeling or feature learning in survival analyses. It can be applied to survival datasets (1) with or without right-censoring, and (2) with or without target covariate features that need to be adjusted for. This algorithm is designed for target problems where the 'burden' (i.e. sum) of specific feature values in the dataset may be predictive of a time-to-event outcome (e.g. the burden of certain HLA amino-acid mismatches (between kidney donor and recipient pairs) can be predictive of kidney graft failure time).

FIBERS can be used directly as a **modeling strategy**, by training a bin population and using the predict() function to apply the discovered bin with the highest fitness as a predictive model of risk group assigment. It can also be used as a **feature learning algorithm**, by training a bin population and using the transform() function to convert each discovered bin in the population into corresponding dataset features for additional downstream machine learning modeling. 

The FIBERS algorithm seeks to automatically identify and optimize a population of 'candidate bins' that maximize time-to-event differences between high and low risk groups. A 'bin' is a subset of features and an associated 'burden threshold' that together define high vs. low risk instance groups, where instances that have a bin sum (of feature values) greater than the threshold are assigend to the high-risk group, and all others to the low-risk group. The fitness (i.e. quality) of bins in the candidate bin population drives evolutionary algorithm learning. 

A schematic detailing how the FIBERS algorithm works is given below:

![alttext](https://github.com/UrbsLab/scikit-FIBERS/blob/dev/Pictures/FIBERS2.0_paper_vertical_color.png?raw=true)

FIBERS currently offers three fitness function options: (1) log-rank fitness, for data without covariates, seeks to maximize the separation between high and low-risk survival curves (2) residuals fitness, for data with covariates, seeks to maximize the difference between deviance residuals between high and low-risk instance groups, and (3) product fitness, for data with covariates, calculates both log-rank and residuals metrics and assignes fitness as the product of both scores. 

***
<a id="item-two"></a>
## Installation
scikit-FIBERS can be installed easily via pip, or by cloning this repository. Either way, make sure that you have also installed all prerequisite packages included in requirements.txt prior to running. 

### Pip Install
Note: Not yet updated to the most recent release.
```
pip install scikit-fibers
```

### Clone Respository
```
git clone --single-branch https://github.com/UrbsLab/scikit-FIBERS
cd scikit-FIBERS
pip install -r requirements.txt
```

***
<a id="item-three"></a>
## Input Data
scikit-FIBERS takes a pandas dataframe of the target dataset as input (including a header). This dataset can include columns for (1) one or more potentially predictive features (2) a time-to-event outcome, (3) a censoring column (optional), where 0 values indicate that the time-to-event is the censoring time, and 1 values indicate that the time-to-event is the actual observed event time, (4) one or more covariate features (optional) that exclude any perfect correlations among them. 

Note that bins sum the feature values of potentially predictive features to determine risk group assignment, so all potentially predictive features should either (1) have the same value range (2) be scaled/normalized ahead of time, and or (3) should have value magnitudes that reflect their weight of importance with respect to other features. All values in the dataset should be numeric.

scikit-FIBERS can also take in a previously trained or manually generated bin population to partially or fully initialize the bin population for training. This bin population is passed as a dataframe of a FIBERS-formatted bin population using the 'manual_bin_init' hyperparameter. This allows users to continue evolving a previously trained bin population for futher iterations, or to use domain/expert knowledge to initialize the evolutionary algorithm with candidate bins. 

***
<a id="item-four"></a>
## Using scikit-FIBERS

### Demonstration Notebooks
Two Jupyter Notebooks have been included to demonstrate how scikit-FIBERS (and it's functions) can be applied to data with or without covariates. These demonstrations utilize two survival data simulators that are also included in the scikit-FIBERS repository (i.e. SIM1 and SIM2, for simulating right-censored survival data without or with covariates, respectively). 
* [DEMO Notebook 1 (no covariates)](https://github.com/UrbsLab/scikit-FIBERS/blob/dev/FIBERS_Survival_Demo_SIM1.ipynb)
* [DEMO Notebook 2 (with covariates)](https://github.com/UrbsLab/scikit-FIBERS/blob/dev/FIBERS_Survival_Demo_SIM2.ipynb)

These notebooks are currently set up to run by downloading this repository and running the included notebook. 

### Basic Run Command Walk-Through
As a simple example of FIBERS training, the following code first loads a hypothetical survival dataset including potentially predictive features, a time-to-event (outcome) column labeled as "Duration", and a censoring indicator column, labeled as "Censoring". Next the FIBERS algorithm is initialized with some basic hyperparameters settings, followed by algorithm training.

```
train_data = pd.read_csv('my_survival_training_data.csv')
fibers = FIBERS(outcome_label="Duration", iterations=100, pop_size=50, fitness_metric="log_rank", censor_label="Censoring")
fibers = fibers.fit(train_data)
```

Once trained, FIBERS can be applied to make risk group predictions on a testing dataset including only potentially predictive feature columns (i.e. no time-to-event, censoring, or covariate columns). First the feature columns alone are loaded as a dataframe. Next, FIBERS's predict() function is called with the 'bin_number' parameter set to the bin index in the bin population to be used as a predictive model. Index '0' is the bin with the highest fitness by default. Lastly, assuming we have the true risk groups of each testing instance (saved as a single-column dataframe) we can generate a classification report comparing risk group predictions to true risk group values. 

```
test_data = pd.read_csv('my_survival_testing_data.csv')
predictions = fibers.predict(test_data,bin_number=0)
print(classification_report(predictions, true_risk_group, digits=3))
```

Lastly, a trained FIBERS bin population can also be converted to learned features in a newly generated dataset using FIBERS transform() function. This feature learning can convert each bin to a new feature by encoding instances as either (1) the sum of bin feature values or (2) the binary risk strata assignment (0=low, 1=high). In the first example below, we transform bins into new features that represent the sum of bin feature values and save this new dataset as a .csv file. 

```
tdf = fibers.transform(train_data,full_sums=True)
tdf.to_csv('my_transformed_dataset_full_sums.csv', index=False)
```

In this next example, we transform bins into new features that represent the binary risk strata assignment and save this new dataset as a .csv file.

```
tdf = fibers.transform(train_data,full_sums=False)
tdf.to_csv('my_transformed_dataset_strata.csv', index=False)
```

***
<a id="item-five"></a>
## Hyperparameters
While scikit_FIBERS has a number of available hyperparameters only a few are considered to be essential or useful to check or set. 

* Essential hyperparameters are given in the first table. 
* For survival data without covariate columns, the $${\color{red}fitness$\_$metric}$$ should be set to 'log_rank' and covariates should be set to None. For survival data with covariate colums (that need to be adjusted for), the fitness_metric should be set to either 'residuals' or to 'log_rank_residuals' (otherwise referred to as the 'product' fitness metric), and covariates should be set to a list of all covariate column lables in the dataset. For survival data without censoring, censor_label should be set to None, otherwise it should be set the censoring column lable. While outcome_type is also an important hyperparameter to check, currently only the 'survival' option (for survial data analysis) has been fully implemented and tested.

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| outcome_label | Data column label for time-to-event (outcome) | str | 'Duration' |
| outcome_type | defines the type of outcome in the dataset | 'survival','class' | 'survival' |
| censor_label | Data column label for censoring | str/None |'Censoring', None | 
| fitness_metric | Pre-fitness metric driving fitness ranking | 'log_rank','residuals','log_rank_residuals' | 'log_rank' |
| covariates | List of data column labels to be treated as covariates (i.e. not considered for bin inclusion) | list, None | None |

* This second table includes hyperparameters that are not essential but can have a significant impact on algorithm performance. In general, setting iterations and pop_size to larger integers is expected to improve training performance, but will require longer run times. The group_thresh hyperparameter controls the adaptive burden thresholding in FIBERS, where None activates this mechanism, and an integer value (e.g. 0), will enforce a specific burden threshold for all discovered bins. Related to this, max_thresh controls the maximum burden threshold allowed in bins, assuming group_thresh = None. Also related to adaptive burden thresholding, the thresh_evolve_prob (set between 0.0 and 1.0) and lead to better performance when set closer to 1.0, but require significantly more runtime. The group_strata_min

Lastly, manual_bin_init allows users to load an existing set of candidate bins for FIBERS to start learning from, rather than starting from randomly initialized bins. The value of using this function depends on the quality of the loaded bins, however utilizing expert (i.e. domain) knowledge to design candidate bins before running FIBERS has the potential to dramatically improve or speed up learning in larger or more complex tasks. 

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| iterations | Number of training/optimization cycles | int | 100 |
| pop_size | Maximum bin population size at end of each cycle | int | 50 |
| group_thresh | Optionally specify a group threshold for bins to use | int, None | None |
| max_thresh | Maximum group threshold for adaptive thresholding | int | 5 |
| thresh_evolve_prob | Probability that an optimization cycle will evolve vs. deterministically select a group threshold for new bin evaluation | float | 0.5 |
| group_strata_min | Min. cutoff for group strata sizes below which a pre-fitness penalty is applied to bin | float | 0.2 |
| manual_bin_init | Dataframe of FIBERS-formatted bin population used to initialize the bin population | dataframe, None | None |
| pop_clean | Optional bin population cleanup strategy | 'group_strata', None | None |

* The remaining hyperparameters in the table below can largely be left to their default values by most users. 

| Hyperparameter | Description | Type/Options | Default Value |
| -------------- | ----------- | ------------- | ------------- |
| tournament_prop | Population proportion randomly selected for tournament | float | 0.2 |
| crossover_prob | Uniform crossover operator probability | float | 0.5 |
| min_mutation_prob | Minimum mutation operator probability | float | 0.2 |
| max_mutation_prob | Maximum mutation operator probability | float | 0.2 |
| merge_prob | Merge operator probability | float | 0.1 |
| new_gen | Proportion of pop_size. Determines the number of offspring to generate each generation | float | 1.0 |
| elitism | Proportion of pop_size protected from deletion | float | 0.1 |
| diversity_pressure | Number of bin similarity clusters driving bin deletion to encourage bin diversity | int | 3 |
| min_bin_size | Min. number of features to be specified in a bin | int | 1 |
| max_bin_size | Max. number of features to be specified in a bin | int/None | None |
| max_in_init_size | Max. number of features in initialized bins | int | 10 |
| log_rank_weightning | Optional weighting of log-rank test | 'wilcoxon', 'tarone-ware', 'peto', 'fleming-harrington', None | None |
| penalty | Penalty multiplier applied to pre-fitness when bin’s group strata ratio goes below the minimum | float | 0.5 |
| min_thresh | Minimum group threshold for adaptive thresholding | int | 0 |
| int_thresh | Boolean indicating that adaptive bin thresholds are limited to positive intergers | Boolean | True |
| random_seed | Seed value used to generate random numbers or make random selections | int, None | None |
| report | List of integers, indicating iterations where the population will be printed out for viewing | list, None | None |
| verbose | Boolean flag to run in 'verbose' mode - display run details | Boolean | False |

***
<a id="item-six"></a>
## Algorithm History
FIBERS was originally based on the [RARE](https://github.com/UrbsLab/RARE) algorithm, an evolutionary algorithm for rare variant binning. (Dasariraju, S. and Urbanowicz, R.J., 2021, July. [RARE: evolutionary feature engineering for rare-variant bin discovery.](https://dl.acm.org/doi/abs/10.1145/3449726.3463174?casa_token=0MRY0eLfZW0AAAAA:PD75rM0SB_V37prY2Ey1CPCu5twUrWMoPn5C6tD9sBRuQy5TJ_TeqhzWwmvp41gbrsPtQerZpPI56A) In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 1335-1343).)

The first implementation of FIBERS was developed within it's own [GitHub repository](https://github.com/UrbsLab/FIBERS), and was applied to an investigation of graft failure in kidney transplantation. (Dasariraju, S., Gragert, L., Wager, G.L., McCullough, K., Brown, N.K., Kamoun, M. and Urbanowicz, R.J., 2023. [HLA amino acid Mismatch-Based risk stratification of kidney allograft failure using a novel Machine learning algorithm.](https://www.sciencedirect.com/science/article/pii/S1532046423000953?casa_token=HP4rI5N9iFkAAAAA:-NgwMAlLUWlvLzzBHU9qz08mv-evC19YxIsFH5RTiGpSiXEd-uBuOkfZbuBShTwstT50vDnIsrM) Journal of Biomedical Informatics, 142, p.104374.)

The first publication detailing scikit-FIBERS (release 0.9.3) was applied and evaluated on simulated right-censored survival data with amino acid mismatch features.
The code for that is available [here](https://github.com/UrbsLab/scikit-FIBERS/tree/gecco_dev). (Urbanowicz, R., Bandhey, H., Kamoun, M., Fogarty, N. and Hsieh, Y.A., 2023, July. [Scikit-FIBERS: An'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analyses.](https://dl.acm.org/doi/abs/10.1145/3583133.3596393?casa_token=jZEPXXznvuUAAAAA:IdV4u-Q07p8_AEfvnTtLpBJePZzmdR2DsImvtpN0z2mge0tgLwqutEF18q74afpj9pOnQ8OnlxPKjw) In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (pp. 1846-1854).) This is the synonmous to [FIBERS 1.0 Release](https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v1.0-beta).

scikit-FIBERS was extended with a prototype adaptive burden thresholding using '[FIBERS-AT](https://github.com/UrbsLab/scikit-FIBERS/tree/evostar_24)' approach to allow bins to simulaneously identify the best bin threshold to apply. (Bandhey, H., Sadek, S., Kamoun, M. and Urbanowicz, R., 2024, March. [Evolutionary Feature-Binning with Adaptive Burden Thresholding for Biomedical Risk Stratification.](https://link.springer.com/chapter/10.1007/978-3-031-56855-8_14) In International Conference on the Applications of Evolutionary Computation (Part of EvoStar) (pp. 225-239). Cham: Springer Nature Switzerland.)

Most recently scikit-FIBERS 2.0 was released, as a completely redesigned, refactored and expanded implementation. Expansions include (1) a merge operator, (2) variable mutation rate, (3) improved adaptive burden thresholding, (4) a bin diversity pressure deletion mechanism, (5) fitness options based on deviance residuals to estimate covariate adjustments throughout algorithm training, (6) a bin population cleanup option, and (7) a number of other helpful functions to report/save the underlying bin population and generate various visualizations. A publication on scikit-FIBERS 2.0 is in preparation. 

### FIBERS 2.0 Paper Analysis Notes
The repository contains high-performance-cluster running scripts for different version of FIBERS for the anaysis.
Specifically this repository compares 3 versions of scikit-FIBERS
1. FIBERS 1.0: https://github.com/UrbsLab/scikit-FIBERS/tree/gecco_dev / https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v1.0-beta
2. FIBERS AT: https://github.com/UrbsLab/scikit-FIBERS/tree/evostar_24
3. FIBERS 2.0: https://github.com/UrbsLab/scikit-FIBERS/tree/dev / https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v2.0.0

***
<a id="item-seven"></a>
## Citing scikit-FIBERS
The manuscript for scikit-FIBERS 2.0 is currently in preparation.

If you use scikit-FIBERS in a scientific publication, please consider citing one of the following papers:

Harsh Bandhey, Nolan Fogarty, Yi-An Hsieh, Malek Kamoun, Ryan J. Urbanowicz (2023). [Scikit-FIBERS: An 'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analyses](https://dl.acm.org/doi/abs/10.1145/3583133.3596393).

BibTeX entry:
```bibtex
@inproceedings{urbanowicz2023scikit,
  title={Scikit-FIBERS: An'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analyses},
  author={Urbanowicz, Ryan and Bandhey, Harsh and Kamoun, Malek and Fogarty, Nolan and Hsieh, Yi-An},
  booktitle={Proceedings of the Companion Conference on Genetic and Evolutionary Computation},
  pages={1846--1854},
  year={2023}
}
```

Harsh Bandhey, Sphia Sadek, Malek Kamoun, Ryan J. Urbanowicz (2024). [Evolutionary Feature-Binning with Adaptive Burden Thresholding for Biomedical Risk Stratification](https://link.springer.com/chapter/10.1007/978-3-031-56855-8_14).

BibTeX entry:
```bibtex
@inproceedings{bandhey2024evolutionary,
  title={Evolutionary Feature-Binning with Adaptive Burden Thresholding for Biomedical Risk Stratification},
  author={Bandhey, Harsh and Sadek, Sphia and Kamoun, Malek and Urbanowicz, Ryan},
  booktitle={International Conference on the Applications of Evolutionary Computation (Part of EvoStar)},
  pages={225--239},
  year={2024},
  organization={Springer}
}
```

***
<a id="item-eight"></a>
## Futher Documentation:
Further code documentation regarding the scikit-FIBERS API can be found [here](https://urbslab.github.io/scikit-FIBERS/skfibers.html).

***
<a id="item-nine"></a>
## Contact
Please email Ryan.Urbanowicz@cshs.org or Harsh.Bandhey@cshs.org for any inquiries related to scikit-FIBERS.


***
<a id="item-ten"></a>
## Acknowledgements
The development of FIBERS benefited from feedback across multiple biomedical research collaborators at the University of Pennsylvania, Tulane University, and the Arbor Research Collaborative for Health. 

The bulk of the coding for the current version of FIBERS was completed by Ryan Urbanowicz and Harsh Bandhey, with credit to Satvik Dasariraju for his implementation of the original FIBERS 1.0 algorithm. Other algorithm/coding contributions have also made by Nolan Fogarty, Yi-An Hsieh, Sphia Sadek, Brian Ling, Gabe Lipschutz-Villa, and Praneel Vashney.

Funding supporting this work comes from NIH grant: R01 AI173095