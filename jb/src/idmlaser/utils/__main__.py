# src/idmlaser/utils/__main__.py

import os
import glob
import sys

def list_available_utils():
    utils_dir = os.path.dirname(__file__)
    utils_files = glob.glob(os.path.join(utils_dir, "*.py"))
    utils_names = [os.path.splitext(os.path.basename(file))[0] for file in utils_files]
    utils_names.remove("__main__")  # Remove this file from the list

    if utils_names:
        print("Available utils:")
        for util in utils_names:
            print(f"- {util}")
    else:
        print("No utils found in this package.")

def main():
    list_available_utils()

if __name__ == "__main__":
    main()

