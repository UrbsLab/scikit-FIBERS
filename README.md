# scikit-FIBERS

Feature Inclusion Bin Evolver for Risk Stratification (FIBERS) (under development, publication forthcoming) is 
an evolutionary algorithm for binning features to stratify risk in biomedical datasets.

FIBERS utilizes an evolutionary algorithm approach to optimizing bins of features based on their stratification of event risk through the following steps:

1) Random bin initialization or expert knowledge input; the bin value at an instance is the sum of the instance's values for the features included in the bin
2) Repeated evolutionary cycles consisting of: 
   - Candidate bin evaluation with log-rank test to evaluate for significant difference in survival curves of the low risk group (instances for which bin value = 0) and high risk group (instances for which bin value > 0).
   - Genetic operations (elitism, parent selection, crossover, and mutation) for new bin discovery and generation of the next generation of candidate bins
3) Final bin evaluation and summary of risk stratification provided by top bins


## Installation
We can easily install scikit-rare using the following command:
```
pip install scikit-fibers
```

## Parameters for FIBERS Class:
1) given_starting_point: whether expert knowledge is being inputted (True or False)
2) amino_acid_start_point: if RARE is starting with expert knowledge, input the list of features here; otherwise None
3) amino_acid_bins_start_point: if RARE is starting with expert knowledge, input the list of bins of features here; otherwise None
4) iterations: the number of evolutionary cycles RARE will run
5) original_feature_matrix: the dataset (containing only features eligible for binning, the variable indicating event occurence, and the variable indicating time to event)
6) label_name: label of the variable indicating event/endpoint occurrence (e.g., 'Graft Failure'); should be a column in the dataset
7) duration_name: name of the variable indicating time to event or time to observation; should be a column in the dataset
8) set_number_of_bins: the population size of candidate bins
9) min_features_per_group: the minimum number of features in a bin
10) max_number_of_groups_with_feature: the maximum number of bins containing a feature
11) informative_cutoff: the minimum proportion allowed for a risk group, all bins that result in a smaller risk group representing a proportion below this cutoff will automatically be assigned a fitness score of 0 (e.g., 0.2 means that both the low risk group and high risk group must represent over 20% of the total population)
12) crossover_probability: the probability of each feature in an offspring bin to crossover to the paired offspring bin (recommendation: 0.5 to 0.8)
13) mutation_probability: the probability of each feature in a bin to be deleted (a proportionate probability is automatically applied on each feature outside the bin to be added (recommendation: 0.05 to 0.5 depending on situation and number of iterations run)
14) elitism_parameter: the proportion of elite bins in the current generation to be preserved for the next evolutionary cycle (recommendation: 0.2 to 0.8 depending on conservativeness of approach and number of iterations run)

## Contact

Please email sdasariraju23@lawrenceville.org and Ryan.Urbanowicz@cshs.org for any 
inquiries related to FIBERS.