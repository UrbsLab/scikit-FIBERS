import logging
from lifelines import CoxPHFitter

def prepare_data(df,outcome_label,censor_label,covariates):
    outcome_df = df[outcome_label]
    print("Outcome Data Shape: "+str(outcome_df.shape))
    feature_df = df.drop(columns=outcome_label)
    if censor_label != None:
        censor_df = df[censor_label]
        print("Censor Data Shape: "+str(censor_df.shape))
        feature_df = feature_df.drop(columns=censor_label)
    else:
        censor_df = None
    print("Init Feature Data Shape: "+str(feature_df.shape))

    # Remove invariant features (data cleaning)
    cols_to_drop = []
    for col in feature_df.columns:
        if len(feature_df[col].unique()) == 1:
            cols_to_drop.append(col)
    feature_df.drop(columns=cols_to_drop, inplace=True)
    print("Cleaned Feature Data Shape: "+str(feature_df.shape))

    # Make covariate dataframe
    if covariates:  
        covariate_df = feature_df[covariates]
        print("Covariate Shape: "+str(covariate_df.shape))
    else:
        covariate_df = None

    return feature_df,outcome_df,censor_df,covariate_df

"""
def get_covariates(df,covariates):
    covariate_matrix = df.copy()   
    covariate_matrix = covariate_matrix[covariates]
    covariate_matrix[outcome_label] = df[outcome_label]
    if censor_label != None:
        covariate_matrix[censor_label] = df[censor_label]
    return covariate_matrix
"""

def remove_invariant_features(df):
    cols_to_drop = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def calculate_residuals(covariate_df,outcome_label,censor_label):
    # Fit a Cox proportional hazards model to the DataFrame
    logging.info("Fitting COX Model")
    cph = CoxPHFitter()
    cph.fit(covariate_df, duration_col=outcome_label, event_col=censor_label, show_progress=True)

    # Calculate the residuals using the Schoenfeld residuals method
    residuals = cph.compute_residuals(covariate_df, kind='deviance')
    return residuals