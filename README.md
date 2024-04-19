# scikit-FIBERS

**Feature Inclusion Bin Evolver for Risk Stratification (FIBERS)** is an evolutionary algorithm for automatically binning features to stratify risk in right-censored survival data. In particular it was designed for features that correspond to mismatches between donor and recipients for transplantation. This repository focuses on a scikit-learn compatible implementation and ongoing improvement/expansion of the of the original [(FIBERS)](https://doi.org/10.1016/j.jbi.2023.104374) algorithm. Further development of the FIBERS algorithm will take place via this repository. The schematic below outlines how this algorithm works.

![alttext](https://github.com/UrbsLab/scikit-FIBERS/blob/main/Pictures/FIBERS2.0_paper_vertical_color.png?raw=true)

## Installation
We can easily install scikit-FIBERS using the following command:
```
pip install scikit-fibers
```

## How to Use:
An [Example Notebook](FIBERS_Survival_Demo.ipynb) is given with sample code that shows what functions are available
in scikit-FIBERS and how to use them by utilizing a built in survival data simulator. This notebook is currently set up to run by downloading this repository and running the included notebook, however you can also set up scikit-fibers to be installed and applied using pip install (above).

## Read About and Cite FIBERS and scikit-FIBERS
FIBERS was originally based on the [RARE](https://github.com/UrbsLab/RARE) algorithm, an evolutionary algorithm for rare variant binning.

Dasariraju, S. and Urbanowicz, R.J., 2021, July. [RARE: evolutionary feature engineering for rare-variant bin discovery.](https://dl.acm.org/doi/abs/10.1145/3449726.3463174?casa_token=0MRY0eLfZW0AAAAA:PD75rM0SB_V37prY2Ey1CPCu5twUrWMoPn5C6tD9sBRuQy5TJ_TeqhzWwmvp41gbrsPtQerZpPI56A) In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 1335-1343).

The first implementation of [FIBERS](https://github.com/UrbsLab/FIBERS) was developed within it's own GitHub repository, and was applied to an investigation of graft failure in kidney transplantation. 

Dasariraju, S., Gragert, L., Wager, G.L., McCullough, K., Brown, N.K., Kamoun, M. and Urbanowicz, R.J., 2023. [HLA amino acid Mismatch-Based risk stratification of kidney allograft failure using a novel Machine learning algorithm.](https://www.sciencedirect.com/science/article/pii/S1532046423000953?casa_token=HP4rI5N9iFkAAAAA:-NgwMAlLUWlvLzzBHU9qz08mv-evC19YxIsFH5RTiGpSiXEd-uBuOkfZbuBShTwstT50vDnIsrM) Journal of Biomedical Informatics, 142, p.104374.

The first publication detailing scikit-FIBERS (release 0.9.3) was applied and evaluated on simulated right-censored survival data with amino acid mismatch features.

Urbanowicz, R., Bandhey, H., Kamoun, M., Fogarty, N. and Hsieh, Y.A., 2023, July. [Scikit-FIBERS: An'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analyses.](https://dl.acm.org/doi/abs/10.1145/3583133.3596393?casa_token=jZEPXXznvuUAAAAA:IdV4u-Q07p8_AEfvnTtLpBJePZzmdR2DsImvtpN0z2mge0tgLwqutEF18q74afpj9pOnQ8OnlxPKjw) In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (pp. 1846-1854).

FIBERS was extended with a prototype '[adaptive burden thresholding](https://github.com/UrbsLab/scikit-FIBERS/tree/evostar_24)' approach to allow bins to simulaneously identify the best bin threshold to apply.

Bandhey, H., Sadek, S., Kamoun, M. and Urbanowicz, R., 2024, March. [Evolutionary Feature-Binning with Adaptive Burden Thresholding for Biomedical Risk Stratification.](https://link.springer.com/chapter/10.1007/978-3-031-56855-8_14) In International Conference on the Applications of Evolutionary Computation (Part of EvoStar) (pp. 225-239). Cham: Springer Nature Switzerland.

Most recently FIBERS 2.0 was released, as a completely redesigned, refactored and expanded implementation. Expansions include (1) a merge operator, (2) variable mutation rate, (3) improved adaptive burden thresholding, (4) a bin diversity pressure mechanism, (5) a fitness option based on deviance residuals to estimate covariate adjustments throught algorithm training, and (6) a bin population cleanup option. This paper is currently submitted (under review). Code to run the analyses in this paper is available [here](https://github.com/UrbsLab/scikit-FIBERS/tree/ppsn).

Urbanowicz, R., Bandhey, H., McCullough, K., Chang, A., Gragert, L., Brown, N., Kamoun, M., 2024, April. FIBERS 2.0: Evolutionary Feature Binning For Biomedical Risk Stratification in Right-Censored Survival Analyses With Covariates.

## Documentation:
Extensive code documentation about the scikit-FIBERS API
can be found [here](https://urbslab.github.io/scikit-FIBERS/skfibers.html) in the guide. (this guide is currently being updated for version 2.0)

## Contact
Please email Ryan.Urbanowicz@cshs.org and Harsh.Bandhey@cshs.org for any
inquiries related to scikit-FIBERS.
