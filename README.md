# scikit-FIBERS

**Feature Inclusion Bin Evolver for Risk Stratification (FIBERS)** is an evolutionary algorithm for automatically binning features to stratify risk in right-censored survival data. In particular it was designed for features that correspond to mismatches between donor and recipients for transplantation. This repository focuses on a scikit-learn compatible implementation of the original [(FIBERS)](https://doi.org/10.1016/j.jbi.2023.104374) algorithm. Further development of the FIBERS algorithm will take place via this repository.

![alttext](https://github.com/UrbsLab/FIBERS/blob/main/Pictures/FIBERS_GECCO_Fig1.png?raw=true)

It utilizes an evolutionary algorithm approach to optimizing bins of features based on their stratification of event risk through the following steps:

1) Random bin initialization or expert knowledge input; the bin value at an instance is the sum of the instance's values for the features included in the bin
2) Repeated evolutionary cycles consisting of:
   - Candidate bin evaluation with log-rank test to evaluate for significant difference in survival curves of the low risk group (instances for which bin value = 0) and high risk group (instances for which bin value > 0).
   - Genetic operations (elitism, parent selection, crossover, and mutation) for new bin discovery and generation of the next generation of candidate bins
3) Final bin evaluation and summary of risk stratification provided by top bins

## Installation
We can easily install scikit-FIBERS using the following command:
```
pip install scikit-fibers
```

## Read More About scikit-FIBERS
The first publication detailing scikit-FIBERS (release 0.9.3) and applying it to simulated right-censored survival data with amino acid mismatch features is currently in press:

Bandhey, H., Fogarty, N., Hsieh, Y., Kamoun, M., Urbanowicz, R. Scikit-FIBERS: An 'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analysis. In Proceedings of the 25th annual conference on Genetic and evolutionary computation. 2023. (In Press)

## How to Use:
An [Example Notebook](ExampleNotebook.ipynb) is given with sample code that shows what functions are available
in scikit-FIBERS and how to use them.

## Documentation:
Extensive code documentation about the scikit-FIBERS API
can be found [here](https://urbslab.github.io/scikit-FIBERS/skfibers.html) in the guide.

## Contact
Please email Ryan.Urbanowicz@cshs.org and Harsh.Bandhey@cshs.org for any
inquiries related to scikit-FIBERS.
