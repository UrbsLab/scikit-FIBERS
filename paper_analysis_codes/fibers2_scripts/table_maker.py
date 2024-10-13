import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
directory = 'C:/Users/ryanu/Desktop/FIBERS2.0_SIM2_Analyses/'
dataname = 'Fibers2.0_sim2_log_rank_residuals_master_summary'
file_path = directory+dataname+'.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

results_list = []
for i in range (0,len(df)): #each row of dataframe
    experiment = []
    experiment.append(df.iloc[i,2])
    experiment.append("'"+str(df.iloc[i,27])+'/30') #TC1
    experiment.append(str(round(df.iloc[i,3],3))+' ('+str(round(df.iloc[i,4],3))+')') #Accuracy
    experiment.append(str(round(df.iloc[i,5],3))+' ('+str(round(df.iloc[i,6],3))+')') #Num Pred
    experiment.append(str(round(df.iloc[i,7],3))+' ('+str(round(df.iloc[i,8],3))+')') #Num Rand
    experiment.append("'"+str(df.iloc[i,9])+'/30') #Top bin
    experiment.append(str(round(df.iloc[i,10],2))+' ('+str(round(df.iloc[i,11],2))+')') #Ideal Iter
    experiment.append("'"+str(df.iloc[i,12])+'/30') #True Thresh
    experiment.append(str(round(df.iloc[i,15],1))+' ('+str(round(df.iloc[i,16],1))+')') #log rank
    experiment.append(str(round(df.iloc[i,17],2))+' ('+str(round(df.iloc[i,18],2))+')') #residual
    experiment.append(str(round(df.iloc[i,21],2))+' ('+str(round(df.iloc[i,22],2))+')') #adj HR
    experiment.append(str(round(df.iloc[i,23],3))+' ('+str(round(df.iloc[i,24],3))+')') #Group ratio
    experiment.append(str(round(df.iloc[i,25],2))+' ('+str(round(df.iloc[i,26],2))+')') #runtime

    results_list.append(experiment)

#Turn results list into a transposed dataframe
newdf = pd.DataFrame(results_list)
transposed_df = newdf.T
#print(transposed_df)

out_file_path = directory+dataname+'_Table.csv'
transposed_df.to_csv(out_file_path, index=False)