""" RUN THIS PROGRAM ON YOUR allPictures FOLDER BEFORE RUNNING 'pickDataset.py'. 
    IT HELPS TO HAVE NUMERICAL NAMING FOR ALL PICTURES"""

# This program renames all files under each person to be numerical
# ex: 0000.jpg, 0001.jpg, 0002.jpg
# Will skip over files already numerically named
import os 
from os import walk
import sys

try:
    str_folderLoc = str(sys.argv[1])
except IndexError:
    print("Format: renamePictures.py [loc of picture directory tree]")

try:
    for root, dirs, files in walk(str_folderLoc, topdown=True):
        count = 0
        files.sort()
        for name in files:
            origFileName = str(os.path.join(root, name))
            print("File: ", origFileName)
            ext = name[-4:]
            rename = str(count).zfill(4) + ext
            if name == rename:
                count += 1
                continue

            print("Change to: ", rename)
            os.rename(origFileName, root + "/" + rename)
            count += 1

except NotADirectoryError:
    print("Directory Not Found Exception")
