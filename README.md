# scikit-FIBERS

scikit-FIBERS is a scikit-learn compatible implementation of the Feature Inclusion Bin Evolver for Risk Stratification [(FIBERS)](https://doi.org/10.1016/j.jbi.2023.104374),
an evolutionary algorithm for binning features to stratify risk in biomedical datasets.

It utilizes an evolutionary algorithm approach to optimizing bins of features based on their stratification of event risk through the following steps:

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

## How to Use:
An [Example Notebook](ExampleNotebook.ipynb) is given with sample codes what functions are available
in scikit-FIBERS and how to use them.

## Documentation:
Extensive code documentation about the scikit-FIBERS API 
can be found [here](https://urbslab.github.io/scikit-FIBERS/skfibers.html) in the guide.

## Contact

Please email Ryan.Urbanowicz@cshs.org and Harsh.Bandhey@cshs.org for any 
inquiries related to scikit-FIBERS.