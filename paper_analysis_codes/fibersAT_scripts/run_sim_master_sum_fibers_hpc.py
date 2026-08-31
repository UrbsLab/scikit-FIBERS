import os
import argparse
import sys
import pandas as pd

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename

    options=parser.parse_args(argv[1:])

    writepath = options.writepath+'/output'
    outputfolder = options.outputfolder
    targetfolder = writepath +'/'+outputfolder

    #open each output subfolder and grab the second row
    master_files = []
    for folder in os.listdir(targetfolder):
        if os.path.isdir(os.path.join(targetfolder,folder)):
            subfolder = os.path.join(targetfolder,folder)
            master_files.append(subfolder+'/summary/'+folder+'_master_summary.csv')

    #Read the header from the first CSV file
    header = pd.read_csv(master_files[0], nrows=0).columns.tolist()

    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame(columns=header)

    # Iterate over each CSV file (excluding the first one) and add its first row to the combined DataFrame
    for master_file in master_files:
        data = pd.read_csv(master_file, nrows=1, header=0, names=header)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Save the combined data to a new CSV file
    combined_data.to_csv(targetfolder+'/'+outputfolder+'_master_summary.csv', index=False)

    
if __name__=="__main__":
    sys.exit(main(sys.argv))
