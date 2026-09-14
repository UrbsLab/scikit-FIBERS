# Running scikit-FIBERS
## Training
As a simple example of FIBERS training, the following code first loads a hypothetical survival dataset (without covariates) including potentially predictive features, a time-to-event (outcome) column labeled as "Duration", and a censoring indicator column, labeled as "Censoring". Next the FIBERS algorithm is initialized with some basic hyperparameters settings, followed by algorithm training.

```
train_data = pd.read_csv('my_survival_training_data.csv')
fibers = FIBERS(outcome_label="Duration", iterations=100, pop_size=50, fitness_metric="log_rank", censor_label="Censoring")
fibers = fibers.fit(train_data)
```

## Testing Evaluation
Once trained, FIBERS can be applied to make risk group predictions on a testing dataset that only includes potentially predictive feature columns (i.e. no time-to-event, censoring, or covariate columns). First, the feature columns alone are loaded as a dataframe. Next, FIBERS's predict() function is called with the *bin_number* parameter set to the bin 'index' in the bin population to be used as a predictive model. Index '0' is the bin with the highest fitness by default. Lastly, assuming we have the true risk groups of each testing instance (saved as a single-column dataframe) we can optionally generate a classification report comparing risk group predictions to true risk group values. 

```
test_data = pd.read_csv('my_survival_testing_data.csv')
predictions = fibers.predict(test_data,bin_number=0)
print(classification_report(predictions, true_risk_group, digits=3))
```
## Feature Learning
Lastly, a trained FIBERS bin population can also be converted to learned features in a newly generated dataset using FIBERS transform() function. This feature learning can convert each bin to a new feature by encoding instances as either (1) the sum of bin feature values or (2) the binary risk strata assignment (0=low, 1=high). In the first example below, we transform bins into new features that represent the sum of bin feature values (i.e. *full_sums*=True) and save this new dataset as a .csv file. 

```
tdf = fibers.transform(train_data,full_sums=True)
tdf.to_csv('my_transformed_dataset_full_sums.csv', index=False)
```

In this next example, we transform bins into new features that represent the binary risk strata assignment and save this new dataset as a .csv file.

```
tdf = fibers.transform(train_data,full_sums=False)
tdf.to_csv('my_transformed_dataset_strata.csv', index=False)
```