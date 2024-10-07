import os
import shutil

def copy_directory_contents(src, dst, exclude_dirs):
    """
    Copy the contents from src to dst, excluding any directories in exclude_dirs.
    If the destination directory already exists and is not empty, skip the copy for that directory.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)  # Create destination directory if it doesn't exist
    elif os.listdir(dst):  # Check if the destination directory is not empty
        print(f"Skipping {dst} as it is already populated.")
        return  # Skip copying if destination directory exists and has content

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        
        if os.path.isdir(s):
            if item not in exclude_dirs:
                copy_directory_contents(s, d, exclude_dirs)  # Recursively copy directories
        else:
            print(f"Copying {s} to {d}")
            shutil.copy2(s, d)  # Copy files

def main():
    base_src_dir = "/home/giese/claix_work/gecco_logs/"
    base_dst_dir = "/globalwork/giese/gecco_shapenet/logs/"
    exclude_dirs = ["benchmark-checkpoints", "checkpoints", "meshes", "projection"]

    for x in os.listdir(base_src_dir):
        src_dir = os.path.join(base_src_dir, x)
        dst_dir = os.path.join(base_dst_dir, x)
        
        if os.path.isdir(src_dir):
            print(f"Checking contents of {src_dir} for copying to {dst_dir}")
            copy_directory_contents(src_dir, dst_dir, exclude_dirs)

if __name__ == "__main__":
    main()
