import os
import shutil
import argparse

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

def main():
    parser = argparse.ArgumentParser(description='Delete logs and scratch files for one results root.')
    parser.add_argument('--w', required=True, help='Same results root passed to the run scripts')
    args = parser.parse_args()
    root = os.path.abspath(args.w)
    delete_contents(os.path.join(root, 'logs'))
    delete_contents(os.path.join(root, 'scratch'))
    print("Contents of the selected logs and scratch directories have been deleted.")


if __name__ == '__main__':
    main()
