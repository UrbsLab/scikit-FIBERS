# scikit-FIBERS
## About
To Be Added

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
5) original_feature_matrix: the dataset 
6) label_name: label for the class/endpoint column in the dataset (e.g., 'Class')
7) rare_variant_MAF_cutoff: the minor allele frequency cutoff separating common features from rare variant features
8) set_number_of_bins: the population size of candidate bins
9) min_features_per_group: the minimum number of features in a bin
10) max_number_of_groups_with_feature: the maximum number of bins containing a feature
11) scoring_method: 'Univariate', 'Relief', or 'Relief only on bin and common features'
12) score_based_on_sample: if Relief scoring is used, whether or not bin evaluation is done based on a sample of instances rather than the whole dataset
13) score_with_common_variables: if Relief scoring is used, whether or not common features should be used as context for evaluating rare variant bins
14) instance_sample_size: if bin evaluation is done based on a sample of instances, input the sample size here
15) crossover_probability: the probability of each feature in an offspring bin to crossover to the paired offspring bin (recommendation: 0.5 to 0.8)
16) mutation_probability: the probability of each feature in a bin to be deleted (a proportionate probability is automatically applied on each feature outside the bin to be added (recommendation: 0.05 to 0.5 depending on situation and number of iterations run)
17) elitism_parameter: the proportion of elite bins in the current generation to be preserved for the next evolutionary cycle (recommendation: 0.2 to 0.8 depending on conservativeness of approach and number of iterations run)
18) random_seed: the seed value needed to generate a random number
19) bin_size_variability_constraint: sets the max bin size of children to be n times the size of their sibling (recommendation: 2, with larger or smaller values the population would trend heavily towards small or large bins without exploring the search space)
20) max_features_per_bin: sets a max value for the number of features per bin


## Citation
To be added