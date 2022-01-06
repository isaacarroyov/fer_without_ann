# Scripts

## Create the database: From images to CSV files
You can find a larger dataset [**here**](https://www.kaggle.com/msambare/fer2013)



### Transform images to `pandas.DataFrame`


```Python
file_path_positive = '../data/faces/positives/*.jpg'
file_path_negative = '../data/faces/negatives/*.jpg'

df_positive = images_to_dataframe(label=1,file_path=file_path_positive)
df_negative = images_to_dataframe(label=0, file_path=file_path_negative)
```


### Split the `pandas.DataFrame` into train, test and validation sets


```Python
df_positive_train, df_positive_test, df_positive_val = split_train_test_val_dataframes(dataframe=df_positive)
df_negative_train, df_negative_test, df_negative_val = split_train_test_val_dataframes(dataframe=df_negative)
```

### Concatenate the all the data frames into a single one

```Python
def create_final_dataframe(list_dataframes):
    df = pd.concat(list_dataframes)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

df_train = create_final_dataframe(list_dataframes=[df_positive_train, df_negative_train])
df_test = create_final_dataframe(list_dataframes=[df_positive_test, df_negative_test])
df_val = create_final_dataframe(list_dataframes=[df_positive_val, df_negative_val])
```



```Python
file_path_train_data = '../data/train_data.csv'
file_path_test_data = '../data/test_data.csv'
file_path_val_data = '../data/validation_data.csv'

df_train.to_csv(file_path_train_data, index=False, header=False)
df_test.to_csv(file_path_test_data, index= False, header= False)
df_val.to_csv(file_path_val_data, index= False, header= False)
```
