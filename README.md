# scikit-FIBERS 2.0

**FIBERS (Feature Inclusion Bin Evolver for Risk Stratification)** is an evolutionary machine learning algorithm designed for modeling or feature learning in survival analyses. It can be applied to survival datasets (1) with or without right-censoring, and (2) with or without target covariate features that need to be adjusted for. This algorithm is designed for target problems where the 'burden' (i.e. sum) of specific feature values in the dataset may be predictive of a time-to-event outcome (e.g. the burden of certain HLA amino-acid mismatches (between kidney donor and recipient pairs) can be predictive of kidney graft failure time).

The FIBERS algorithm seeks to automatically identify and optimize a population of 'candidate bins' that maximize time-to-event differences between high and low risk groups. A 'bin' is a subset of features and an associated 'burden threshold' that together define high vs. low risk instance groups, where instances that have a bin sum (of feature values) greater than the threshold are assigend to the high-risk group, and all others to the low-risk group. The fitness (i.e. quality) of bins in the candidate bin population drives evolutionary algorithm learning. 

A schematic detailing how the FIBERS algorithm works is given below:

![alttext](https://github.com/UrbsLab/scikit-FIBERS/blob/main/Pictures/FIBERS2.0_paper_vertical_color.png?raw=true)

FIBERS currently offers three fitness function options: (1) log-rank fitness, for data without covariates, seeks to maximize the separation between high and low-risk survival curves (2) residuals fitness, for data with covariates, seeks to maximize the difference between deviance residuals between high and low-risk instance groups, and (3) product fitness, for data with covariates, calculates both log-rank and residuals metrics and assignes fitness as the product of both scores. 

FIBERS can be used directly as a **modeling strategy**, by training a bin population and using the predict() function to apply the discovered bin with the highest fitness as a predictive model of risk group assigment. It can also be used as a **feature learning algorithm**, by training a bin population and using the transform() function to convert each discovered bin in the population into corresponding dataset features for additional downstream machine learning modeling. 



The repository contains high-performance-cluster running scripts for different version of FIBERS for the anaysis.
Specifically this repository compares 3 versions of scikit-FIBERS
1. FIBERS 1.0: https://github.com/UrbsLab/scikit-FIBERS/tree/gecco_dev / https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v1.0-beta
2. FIBERS AT: https://github.com/UrbsLab/scikit-FIBERS/tree/evostar_24
3. FIBERS 2.0: https://github.com/UrbsLab/scikit-FIBERS/tree/dev / https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v2.0.0

## History of FIBERS and scikit-FIBERS
FIBERS was originally based on the [RARE](https://github.com/UrbsLab/RARE) algorithm, an evolutionary algorithm for rare variant binning. (Dasariraju, S. and Urbanowicz, R.J., 2021, July. [RARE: evolutionary feature engineering for rare-variant bin discovery.](https://dl.acm.org/doi/abs/10.1145/3449726.3463174?casa_token=0MRY0eLfZW0AAAAA:PD75rM0SB_V37prY2Ey1CPCu5twUrWMoPn5C6tD9sBRuQy5TJ_TeqhzWwmvp41gbrsPtQerZpPI56A) In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 1335-1343).)

The first implementation of FIBERS was developed within it's own [GitHub repository](https://github.com/UrbsLab/FIBERS), and was applied to an investigation of graft failure in kidney transplantation. (Dasariraju, S., Gragert, L., Wager, G.L., McCullough, K., Brown, N.K., Kamoun, M. and Urbanowicz, R.J., 2023. [HLA amino acid Mismatch-Based risk stratification of kidney allograft failure using a novel Machine learning algorithm.](https://www.sciencedirect.com/science/article/pii/S1532046423000953?casa_token=HP4rI5N9iFkAAAAA:-NgwMAlLUWlvLzzBHU9qz08mv-evC19YxIsFH5RTiGpSiXEd-uBuOkfZbuBShTwstT50vDnIsrM) Journal of Biomedical Informatics, 142, p.104374.)

The first publication detailing scikit-FIBERS (release 0.9.3) was applied and evaluated on simulated right-censored survival data with amino acid mismatch features.
The code for that is available [here](https://github.com/UrbsLab/scikit-FIBERS/tree/gecco_dev). (Urbanowicz, R., Bandhey, H., Kamoun, M., Fogarty, N. and Hsieh, Y.A., 2023, July. [Scikit-FIBERS: An'OR'-Rule Discovery Evolutionary Algorithm for Risk Stratification in Right-Censored Survival Analyses.](https://dl.acm.org/doi/abs/10.1145/3583133.3596393?casa_token=jZEPXXznvuUAAAAA:IdV4u-Q07p8_AEfvnTtLpBJePZzmdR2DsImvtpN0z2mge0tgLwqutEF18q74afpj9pOnQ8OnlxPKjw) In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (pp. 1846-1854).) This is the synonmous to [FIBERS 1.0 Release](https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v1.0-beta).

scikit-FIBERS was extended with a prototype adaptive burden thresholding using '[FIBERS-AT](https://github.com/UrbsLab/scikit-FIBERS/tree/evostar_24)' approach to allow bins to simulaneously identify the best bin threshold to apply. (Bandhey, H., Sadek, S., Kamoun, M. and Urbanowicz, R., 2024, March. [Evolutionary Feature-Binning with Adaptive Burden Thresholding for Biomedical Risk Stratification.](https://link.springer.com/chapter/10.1007/978-3-031-56855-8_14) In International Conference on the Applications of Evolutionary Computation (Part of EvoStar) (pp. 225-239). Cham: Springer Nature Switzerland.)

Most recently scikit-FIBERS 2.0 was released, as a completely redesigned, refactored and expanded implementation. Expansions include (1) a merge operator, (2) variable mutation rate, (3) improved adaptive burden thresholding, (4) a bin diversity pressure mechanism, (5) a fitness option based on deviance residuals to estimate covariate adjustments throughout  algorithm training, and (6) a bin population cleanup option. The code release for this is available [here](https://github.com/UrbsLab/scikit-FIBERS/releases/tag/v2.0.0).

<!-- Urbanowicz, R., Bandhey, H., McCullough, K., Chang, A., Gragert, L., Brown, N., Kamoun, M., 2024, April. FIBERS 2.0: Evolutionary Feature Binning For Biomedical Risk Stratification in Right-Censored Survival Analyses With Covariates. -->

## How to Use

### Pre-requisites and Installation

Steps and commands to set up and run experiments as per the manuscript:
1. Clone the reposioriy with all the code and runner files `git clone --single-branch --branch sim_paper https://github.com/UrbsLab/scikit-FIBERS`
2. Install pre-requisites using `pip install -r requirements.txt`
3. Run the experiments using commands specified in the next section (assumes you have access to an HPC, change HPC type and queue name accordingly). The sample commands below are specified for the UPenn I2C2 Cluster.

### Running the experiments

Commands to simulate datasets:
```
python run_simple_sim.py
python run_covariate_sim.py (covariate sim hpc file to be made)
```

Commands to run separate FIBERS 2.0 experiments:
```
python run_sim_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_default --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.5 --dp 0 --f log_rank --t 0 --cl None
python run_sim_sum_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_default --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibers_hpc.py --w /project/kamoun_shared/ryanurb/ --o Fibers2.0_sim_default

python run_sim_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.5 --dp 0 --f log_rank --t None --cl None
python run_sim_sum_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibers_hpc.py --w /project/kamoun_shared/ryanurb/ --o Fibers2.0_sim_t_0_5

python run_sim_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5_ma_0.1 --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.1 --dp 0 --f log_rank --t None --cl None
python run_sim_sum_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5_ma_0.1 --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibers_hpc.py --w /project/kamoun_shared/ryanurb/ --o Fibers2.0_sim_t_0_5_ma_0.1

python run_sim_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5_d_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.5 --dp 5 --f log_rank --t None --cl None
python run_sim_sum_fibers_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/ryanurb/ --o sim_t_0_5_d_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibers_hpc.py --w /project/kamoun_shared/ryanurb/ --o Fibers2.0_sim_t_0_5_d_5
```

Commands to run separate FIBERS 1.0 and FIBERS AT experiments:
```
python run_sim_fibersv1_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/bandheyh/ --o sim_default_v1 --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.5 --dp 0 --f log_rank --t 0 --cl None
python run_sim_sum_fibersv1_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/bandheyh/ --o sim_default_v1 --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibersv1_hpc.py --w /project/kamoun_shared/ryanurb/ --o Fibers1.0_sim_default

python run_sim_fibersAT_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/bandheyh/ --o FAT_sim_t_0_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal --ol Duration --c Censoring --ma 0.5 --dp 0 --f log_rank --t None --cl None
python run_sim_sum_fibersAT_hpc.py --d /project/kamoun_shared/ryanurb/data/simple_sim_datasets --w /project/kamoun_shared/bandheyh/ --o FAT_sim_t_0_5 --rc LSF --rs 30 --rm 4 --q i2c2_normal
python run_sim_master_sum_fibersAT_hpc.py --w /project/kamoun_shared/ryanurb/ --o FibersAT_sim_t_0_5
```

More to be added

## Documentation:
Extensive code documentation about the scikit-FIBERS 2.0 API can be found [here](https://urbslab.github.io/scikit-FIBERS/skfibers.html) in the guide.
An [Example Notebook](FIBERS_Survival_Demo.ipynb) is given with sample code that shows what functions are available
in scikit-FIBERS 2.0 and how to use them by utilizing a built in survival data simulator. This notebook is currently set up to run by downloading this repository and running the included notebook, however you can also set up scikit-fibers to be installed and applied using pip install.

## Contact
Please email Ryan.Urbanowicz@cshs.org and Harsh.Bandhey@cshs.org for any
inquiries related to scikit-FIBERS.
