# Access the paths of the files
# Save the files in an intermediate step as train and test

from zipfile import ZipFile
import os
from os import listdir
   

def main():

    # get a path to the main input
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    print(listdir(INPUTS_DIR))
    pathtozip = os.path.join(INPUTS_DIR,'zipfileishere')
    print(listdir(pathtozip))
    os.chdir(pathtozip)
    print(os.getcwd())
    print(listdir())
    # archive_path = os.path.join(INPUTS_DIR, archive_file)

    # unzip the folder
    pathtozip = os.path.join(INPUTS_DIR,'zipfileishere/archive.zip')
    print(os.path.isfile(pathtozip))
    with ZipFile(pathtozip, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall()
    print('Files extracted')

    # Take a path to each one of the folder+file
    archive_path = os.path.join(INPUTS_DIR, 'archive/garbage_classification')
    paths=[]
    labels=[]
    for folder in listdir(archive_path):
        for picture in listdir(folder):
            _ = os.path.join(os.path.join(archive_path, folder), picture)
            paths.append(_)
            labels.append(folder)
        print('Completed for ',folder)


if __name__ == '__main__':
    main()