import pandas as pd
from sklearn.model_selection import KFold
import os
import argparse
import sys

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafile', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    options=parser.parse_args(argv[1:])

    datafile= options.datafile
    outputfolder = options.outputfolder

    # Load and split the dataset into 3 folds
    load_and_split_dataset(datafile,outputfolder)

# Function to save the datasets to CSV files
def save_fold(train_data, test_data, fold_idx, output_folder):
    # Create the output folder if it doesn't exist
    #os.makedirs(output_folder, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_folder, f'train_fold_{fold_idx}.csv')
    train_data.to_csv(train_file, index=False)
    
    # Save testing data
    test_file = os.path.join(output_folder, f'test_fold_{fold_idx}.csv')
    test_data.to_csv(test_file, index=False)

# Load the dataset
def load_and_split_dataset(csv_file,output_folder):
    # Load the dataset
    data = pd.read_csv(csv_file)

    # Set up 3-fold cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Loop over the splits
    for fold_idx, (train_index, test_index) in enumerate(kf.split(data), start=1):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        # Save each fold's training and testing data
        save_fold(train_data, test_data, fold_idx, output_folder)

if __name__=="__main__":
    sys.exit(main(sys.argv))

    # Provide the path to your dataset here
    #dataset_path = '/project/kamoun_shared/data_shared/harsh_new_imp/NewImp_8.csv'
    #output_folder = '/project/kamoun_shared/data_shared/harsh_new_imp_CV/'

