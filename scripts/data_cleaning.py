import glob, re, os
import sys

def rename_directories(path):
    #renames subdirectories in path to all lowercase and removes numbers 
    for file in os.listdir(path):
        new_name = re.sub('[n0-9]*-', "", file)
        os.rename(path + file, path + new_name.lower())

    print("subdirectories in " + (path) + " have been updated successfully")


if __name__ == "__main__":
    path = str(sys.argv[1])
    rename_directories(path)