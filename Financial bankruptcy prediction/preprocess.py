import pandas as pd
from sklearn.preprocessing import StandardScaler

# Pre process step begins:
class preprocessor():

    def __init__(self, df):
        self.dataframe = df
    
    def getValue(self):
        return self.dataframe

    def setValue(self, df):
        self.dataframe = df

    def feature_dropper(self, features=[]):
        df = self.dataframe.drop(features, axis=1)
        self.setValue(df)
        
    def null_imputer(self, columns=[], mode="mean"):
        if mode == "mean":
            self.dataframe[columns].apply(lambda x: x.fillna(x.mean()), axis=1)
            df = self.dataframe
        self.setValue(df)


    def null_dropper(self, columns=[]):
        df = self.dataframe.dropna(subset=columns)
        self.setValue(df)

    def standardize(self, standard, columns=[]):

        df_copy = self.dataframe.copy()
        df_copy = df_copy[columns]
        standard_df = standard.fit_transform(df_copy)
        self.dataframe[columns] = standard_df


def feature_preprocessor(train_data_object, features_dropna, features_impute, features_todrop, target):

    print("before processing: ", train_data_object.getValue().shape)

    train_data_object.null_imputer(columns=features_impute)
    train_data_object.null_dropper(columns=features_dropna)
    train_data_object.feature_dropper(features=features_todrop)

    obj = train_data_object.getValue()
    
    feature_list = obj.drop([target], axis=1).columns
    standardizer = StandardScaler()
    train_data_object.standardize(standardizer, columns=feature_list)
        
    return train_data_object.getValue()




