"""
This script has the only porpouse of creating CSV's containing the pixel information of 
each face from the folder of train and test data.
"""

# L I B R A R I E S
import numpy as np
import pandas as pd
import cv2 as cv
from glob import glob


# F U N C T I O N S
def images_to_dataframe(label,file_path):
    dataframe = pd.DataFrame()
    count = 1
    list_all_images = glob(pathname=file_path)
    for image in list_all_images:
        img = cv.imread(image)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        row = np.append(label,img_gray.flatten())
        row = row.reshape(1,-1)
        dataframe = dataframe.append(pd.DataFrame(row))
        print(f'{count} out of {len(list_all_images)} from {file_path}')
        count += 1
    return dataframe

def split_train_test_dataframes(dataframe, train_prop=0.75, test_prop=0.25):
    """
    ...
    """

    validate_prop = train_prop + test_prop
    if validate_prop != 1:
        print('train_prop + test_prop should greater than zero. Instead is {}'.format(validate_prop))
        return None,None,None

    n_obs = dataframe.shape[0]

    idx_train = int(n_obs * train_prop)

    df_train = dataframe.iloc[ :idx_train, : ]
    df_test = dataframe.iloc[ idx_train: , : ]

    return df_train, df_test

def create_final_dataframe(list_dataframes):
    df = pd.concat(list_dataframes)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# C O D E 
file_path_positive = '../data/faces/positives/*.jpg'
file_path_negative = '../data/faces/negatives/*.jpg'

file_path_train_data = '../data/training_data.csv'
file_path_test_data = '../data/testing_data.csv'

# Create positive and negative dataframes
df_positive = images_to_dataframe(label=1,file_path=file_path_positive)
print(' ')
df_negative = images_to_dataframe(label=0, file_path=file_path_negative)
print(' ')

# Create train, test and validation data
df_positive_train, df_positive_test = split_train_test_dataframes(dataframe=df_positive)
df_negative_train, df_negative_test = split_train_test_dataframes(dataframe=df_negative)

# Concatenate into three dataframes
df_train = create_final_dataframe(list_dataframes=[df_positive_train, df_negative_train])
df_test = create_final_dataframe(list_dataframes=[df_positive_test, df_negative_test])


df_train.to_csv(file_path_train_data, index=False, header=False)
print('Training data -> DONE')
df_test.to_csv(file_path_test_data, index= False, header= False)
print('Test data -> DONE')