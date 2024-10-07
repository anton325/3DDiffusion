import os
from pathlib import Path
import shutil

def delete_files(directory):
    # List all files and directories in the given directory
    all_files = sorted([os.path.join(directory, file) for file in os.listdir(directory)],
                       key=lambda x: x.lower(), reverse=True)  # Sort alphabetically in reverse
    
    # Check if the number of files/directories is greater than 50,000
    print(f"Number of files/directories in {directory}: {len(all_files)}")
    while len(all_files) > 50000:
        # Remove the last file/directory in the list (alphabetically last)
        file_to_delete = all_files.pop()
        if os.path.isfile(file_to_delete):

            # os.remove(file_to_delete)
            # print(f"Deleting: {file_to_delete}")
            pass
        # print(f"Deleted: {file_to_delete}")

# Replace 'your_directory_path' with the path to the directory you want to manage
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638932_gen/epoch_20")
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638814_gen/epoch_20")
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638753_gen/epoch_20")
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638753_stochgen/epoch_20")
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638932_stochgen/epoch_20")
directory_path = Path("/globalwork/giese/evals/gen_gaussians_eval/48638814_stochgen/epoch_20")
delete_files(directory_path)
