import random
import os
import numpy as np
import shutil

root_dir = '/home/kien/Downloads/archive/Chessman-image-dataset/Chess/'  # data root path
classes_dir = ['Rook', 'Pawn', 'Knight', 'King', 'Bishop', 'Queen'] # total labels

val_ratio = 0.2
test_ratio = 0.2

for cls in classes_dir:
    os.makedirs(root_dir + 'train/' + cls)
    os.makedirs(root_dir + 'val/' + cls)
    os.makedirs(root_dir + 'test/' + cls)

    src = root_dir + cls  # /Downloads/archive/Chessman-image-dataset/Chess/ + "Rook", "Pawn", "Knight", "King", "Bishop", "Queen"
    allFileNames = os.listdir(src)
    # Creating partitions of the data after shuffeling
    # Folder to copy images from

    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + 'train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir + 'val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + 'test/' + cls)