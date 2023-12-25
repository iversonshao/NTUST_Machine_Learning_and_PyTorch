import os
import shutil

train_path = 'vangogh2photo'
trainA_path = os.path.join(train_path, 'trainA')
targetA_path = os.path.join(train_path, 'new_trainA')
trainB_path = os.path.join(train_path, 'trainB')
targetB_path = os.path.join(train_path, 'new_trainB')

if os.path.exists(targetA_path) == False:
    os.makedirs(targetA_path)
    print('Create dir : ', targetA_path)
    shutil.move(trainA_path, targetA_path)

if os.path.exists(targetB_path) == False:
    os.makedirs(targetB_path)
    print('Create dir : ', targetB_path)
    shutil.move(trainB_path, targetB_path)
