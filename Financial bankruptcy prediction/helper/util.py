import pandas as pd
import numpy as np

from sklearn.utils import shuffle

def df_explore_helper(df):
    print("Shape of dataframe: {} ".format(df.shape))
    df_summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    df_summary = df_summary.reset_index()
    df_summary['Name'] = df_summary['index']
    df_summary = df_summary[['Name', 'dtypes']]

    # explore this dataframe
    df_summary['Missing values'] = df.isnull().sum().values
    df_summary['Unique values'] = df.nunique().values
    df_summary['First Value'] = df.iloc[0].values
  
    for name in df_summary['Name'].value_counts().index:
        isNumeric = (df[name].dtype == np.float64 or df[name].dtype == np.int64)
    
        if isNumeric:
            df_summary.loc[df_summary['Name']==name, 'Minimum'] = df[name].min()
            df_summary.loc[df_summary['Name']==name, 'Maximum'] = df[name].max()
    return df_summary


def eval_train_generator(dataframe, target, split_number, upper_bound=None):
    # upper_bound by default is set to none, which means all data will be sampled
    # this value can be configured to use for down sampling in imbalanced, skewed data

    true_record = dataframe[dataframe[target]==1]
    false_record = dataframe[dataframe[target]==0]

    false_record = shuffle(false_record)
    true_record = shuffle(true_record)

    false_eval = false_record.iloc[0:split_number]
    true_eval = true_record.iloc[0:split_number]
    
    if upper_bound:
        false_train = false_record.iloc[split_number+1:upper_bound]
        true_train = true_record.iloc[split_number+1:upper_bound]
    else:
        false_train = false_record.iloc[split_number+1:]
        true_train = true_record.iloc[split_number+1:]

    eval_dataset = pd.concat([true_eval, false_eval])
    eval_dataset = shuffle(eval_dataset)

    train_dataset =  pd.concat([true_train, false_train])
    train_dataset = shuffle(train_dataset)

    print(eval_dataset.shape)
    print(train_dataset.shape)

    eval_dataset.to_csv("data store/evaluation_dataset.csv", encoding='utf-8')
    train_dataset.to_csv("data store/training_dataset.csv", encoding='utf-8')
    print("Completed")
    return True