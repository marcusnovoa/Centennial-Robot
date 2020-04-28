""" RUN 'renamePictures.py' BEFORE RUNNING THIS PROGRAM.
    'renamePictures.py' will keep picture naming consistant"""
# This program automatically picks your train and val datasets
# It will pick 20% of the total pictures associated to each 
#   person in the folder with all your pictures to be your val set

import os
from os import walk
import sys
import math
import random
import shutil

# str_allPicLoc = "./Pictures/dataset/allPictures/"
# str_trainLoc = "./Pictures/dataset/train/"
# str_valLoc = "./Pictures/dataset/val/"

try:
    str_allPicLoc = str(sys.argv[1])
    str_trainLoc = str(sys.argv[2])
    str_valLoc = str(sys.argv[3])
except IndexError:
    print("Format: pickValidationSet.py [loc of ./dataset/allPictures/ tree] [loc of ./dataset/train/ tree] [loc of ./dataset/val/ tree]")


def deleteDataset(dir_val): # Delete val and train sets before picking new data sets from all pictures
    os.chdir(dir_val)
    for root, dirs, files in walk(dir_val, topdown=True):
        for name in files:
            str_fil_fullPath = str(os.path.join(root, name))
            print("Deleting... ", str_fil_fullPath)
            os.remove(str_fil_fullPath)

def pickSet(dir_dataset):
    os.chdir(dir_dataset)
    dict_valFiles = {}
    dict_trainFiles = {}
    for root, dirs, files in walk(dir_dataset, topdown=True):
        files.sort()
        l_files = list(files)
        ct_files = len(l_files)

        if ct_files == 0: continue

        ct_valFiles = ct_files * 0.2  # Only use 20% of all pictures for each person for val set
        ct_valFiles = math.ceil(ct_valFiles)

        valFiles, trainFiles = pickFilesFromList(l_files, ct_valFiles) # Picks random files from list of l_files
        person = os.path.split(root)[-1]
        dict_valFiles[person] = tuple(valFiles)
        dict_trainFiles[person] = tuple(trainFiles)

    return dict_valFiles, dict_trainFiles # Return dictionary set of training and val datasets

def pickFilesFromList(l_files, numRandFiles):
    valSet = random.sample(l_files, numRandFiles)
    trainSet = []
    for f in l_files:
        if f in valSet: continue
        trainSet.append(f)
    return valSet, trainSet

        
def moveSet(dict_dataset, str_origDir, str_newDir): # Move dictionary dataset to final train or val folder location
    for key in dict_dataset:
        newLoc = str(os.path.join(str_newDir, key))
        files = dict_dataset[key]
        for f in files:
            shutil.copy(str_origDir + key + '/' + f, newLoc)
            print("Moved {} to {}".format(str_origDir+key+'/'+f, newLoc))


deleteDataset(str_valLoc)
deleteDataset(str_trainLoc)
dict_valSet, dict_trainSet = pickSet(str_allPicLoc)
moveSet(dict_trainSet, str_allPicLoc, str_trainLoc)
moveSet(dict_valSet, str_allPicLoc, str_valLoc)