import logging
from lifelines import CoxPHFitter

def prepare_data(df, outcome_label, censor_label, covariates):
    # Make list of feature names (i.e. columns that are not outcome, censor, or covariates)
    feature_names = list(df.columns)
    if covariates != None:
        exclude = covariates + [outcome_label,censor_label]
    else:
        exclude = [outcome_label,censor_label]
    feature_names = [item for item in feature_names if item not in exclude]

    # Remove invariant feature columns (data cleaning)
    cols_to_drop = []
    for col in feature_names:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True)
    feature_names = [item for item in feature_names if item not in cols_to_drop]
    print("Dropped "+str(len(cols_to_drop))+" invariant feature columns.")

    return df, feature_names


def calculate_residuals(df,covariates, feature_names, outcome_label,censor_label, penalizer=0.0001): #Ryan - do we need to handle categorical variables like when calculating Cox PH??
    # Fit a Cox proportional hazards model to the DataFrame
    var_list = covariates+[outcome_label,censor_label]
    logging.info("Fitting COX Model")
    no_penalizer = True #Hard coded over-ride
    if covariates == ['AFRICAN-AMERICAN','ASIAN','HISPANIC','WHITE','OTHER','FDFR','FDMR','MDFR','MDMR'] and not no_penalizer:
        cph = CoxPHFitter(penalizer=penalizer)
    else:
        cph = CoxPHFitter()
    cph.fit(df.loc[:,var_list], duration_col=outcome_label, event_col=censor_label, show_progress=True)

    # Calculate the residuals using the Schoenfeld residuals method
    residuals = cph.compute_residuals(df.loc[:,var_list], kind='deviance')
    return residuals


