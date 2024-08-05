import os
import shutil

# Define the paths to the directories
logs_dir = '/project/kamoun_shared/bandheyh/hpcfiles/logs'
scratch_dir = '/project/kamoun_shared/bandheyh/hpcfiles/scratch'

# Function to delete contents of a directory
def delete_contents(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'{directory} does not exist.')

# Delete contents of 'logs' and 'scratch' directories
delete_contents(logs_dir)
delete_contents(scratch_dir)

print("Contents of 'logs' and 'scratch' directories have been deleted.")
