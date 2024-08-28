import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('/project/kamoun_shared/data_shared/simulation_study_covariates_old/BasePC_i_10000_tf_100_p_5_t_0_n_0_c_0_nc_False.csv')

# creating dummies
# data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})

# the independent variables set
covariates = ['AFRICAN-AMERICAN','ASIAN','HISPANIC','WHITE','OTHER','FDFR','FDMR','MDFR','MDMR']
X = data[covariates]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

print(vif_data)