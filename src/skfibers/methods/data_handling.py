import logging
from lifelines import CoxPHFitter

def prepare_data(df,outcome_label,censor_label,covariates):
    outcome_df = df[outcome_label]
    feature_df = df.drop(columns=outcome_label)
    if censor_label != None:
        censor_df = df[censor_label]
        feature_df = feature_df.drop(columns=censor_label)
    else:
        censor_df = None

    # Remove invariant features (data cleaning)
    cols_to_drop = []
    for col in feature_df.columns:
        if len(feature_df[col].unique()) == 1:
            cols_to_drop.append(col)
    feature_df.drop(columns=cols_to_drop, inplace=True)

    # Make covariate dataframe
    if covariates:  
        covariate_df = feature_df[covariates]
        for covariate in covariates:
            feature_df = feature_df.drop(columns=covariate)
    else:
        covariate_df = None


    return feature_df,outcome_df,censor_df,covariate_df


def calculate_residuals(covariate_df,outcome_label,censor_label):
    # Fit a Cox proportional hazards model to the DataFrame
    logging.info("Fitting COX Model")
    cph = CoxPHFitter()
    cph.fit(covariate_df, duration_col=outcome_label, event_col=censor_label, show_progress=True)

    # Calculate the residuals using the Schoenfeld residuals method
    residuals = cph.compute_residuals(covariate_df, kind='deviance')
    return residuals