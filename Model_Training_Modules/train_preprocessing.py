import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import numpy as np
import scipy.stats as st
from sklearn.model_selection import train_test_split
import pickle as pkl
import feature_engine.outliers as feo
import category_encoders as ce
import feature_engine.encoding as fee
from datetime import datetime

class train_Preprocessor:
    def __init__(self, file_object, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of Preprocessor class
            Output: None
        '''
        self.file_object = file_object
        self.result_dir = result_dir
        self.log_writer = App_Logger()

    def extract_compiled_data(self, path):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start reading compiled data from database")
        self.path = path
        try:
            data = pd.read_csv(path)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception(f"Fail to read compiled data from database with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish reading compiled data from database")
        return data
    
    def data_cleaning(self,data):
        '''
            Method Name: data_cleaning
            Description: This method performs initial data cleaning on a given pandas dataframe, while removing certain
            data anomalies.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        cleaned_data = data.copy()
        cleaned_data = cleaned_data[cleaned_data['Sales at Cost']>=0]
        cleaned_data[['NSU','GST Value','MRP','SP']] = cleaned_data[['NSU','GST Value','MRP','SP']].applymap(lambda x: abs(x))
        cleaned_data['NSV'] = np.round(cleaned_data['NSU'] * cleaned_data['SP'],2)
        cleaned_data['NSV-GST'] = cleaned_data['NSV'] - cleaned_data['GST Value']
        cleaned_data['Gross Sales'] = np.round(cleaned_data['NSU'] * cleaned_data['MRP'],2)
        cleaned_data['DIS'] = cleaned_data['MRP'] - cleaned_data['SP']
        cleaned_data['DIS%'] = np.round(cleaned_data['DIS']/cleaned_data['MRP']*100,2)
        cleaned_data = cleaned_data[cleaned_data['SP']<=cleaned_data['MRP']]
        cleaned_data = cleaned_data[(cleaned_data['Sales at Cost']/cleaned_data['NSU'])<10000]
        return cleaned_data

    def remove_irrelevant_columns(self, data, column):
        '''
            Method Name: remove_irrelevant_columns
            Description: This method removes columns from a pandas dataframe, which are not relevant for analysis.
            Output: A pandas DataFrame after removing the specified columns. In addition, columns that are removed will be 
            stored in a separate csv file labeled "Columns_Removed.csv"
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start removing irrelevant columns from the dataset")
        try:
            data = data.drop(column, axis=1)
            result = pd.concat([pd.Series(column, name='Columns_Removed'), pd.Series(["Irrelevant column"]*len(column), name='Reason')], axis=1)
            result.to_csv(self.result_dir+'Columns_Drop_from_Original.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
            raise Exception(f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish removing irrelevant columns from the dataset")
        return data

    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will 
            be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception(f"Fail to remove duplicated rows with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    
    def features_and_labels(self,data,column):
        '''
            Method Name: features_and_labels
            Description: This method splits a pandas dataframe into two pandas objects, consist of features and target labels.
            Output: Two pandas/series objects consist of features and labels separately.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start separating the data into features and labels")
        try:
            X = data.drop(column, axis=1)
            y = data[column]/data['NSU']
            y = y.rename('Cost_per_Unit')
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to separate features and labels with the following error: {e}")
            raise Exception(f"Fail to separate features and labels with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish separating the data into features and labels")
        return X, y
    
    def check_gaussian(self, X_train):
        '''
            Method Name: check_gaussian
            Description: This method classifies columns into gaussian and non-gaussian columns based on anderson test.
            Output: Two list of columns for gaussian and non-gaussian variables.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start categorizing columns into gaussian vs non-gaussian distribution")
        gaussian_columns = []
        non_gaussian_columns = []
        try:
            for column in X_train.columns:
                result = st.anderson(X_train[column])
                if result[0] > result[1][2]:
                    non_gaussian_columns.append(column)
                    self.log_writer.log(self.file_object, f"{column} column is identified as non-gaussian")
                else:
                    gaussian_columns.append(column)
                    self.log_writer.log(self.file_object, f"{column} column is identified as gaussian")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to categorize columns into gaussian vs non-gaussian distribution with the following error: {e}")
            raise Exception(f"Fail to categorize columns into gaussian vs non-gaussian distribution with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish categorizing columns into gaussian vs non-gaussian distribution")
        return gaussian_columns, non_gaussian_columns

    def iqr_lower_upper_bound(self, X_train, column):
        '''
            Method Name: iqr_lower_upper_bound
            Description: This method computes lower bound and upper bound of outliers based on interquartile range (IQR) method
            Output: Two floating values that consist of lower bound and upper bound of outlier points for non-gaussian variables
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f"Start computing lower and upper bound of outliers for {column} column")
        try:
            Q1 = X_train[column].quantile(0.25)
            Q3 = X_train[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
            raise Exception(f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish computing lower and upper bound of outliers for {column} column")
        return lower_bound, upper_bound

    def gaussian_lower_upper_bound(self, X_train, column):
        '''
            Method Name: gaussian_lower_upper_bound
            Description: This method computes lower bound and upper bound of outliers based on gaussian method
            Output: Two floating values that consist of lower bound and upper bound of outlier points for gaussian variables
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f"Start computing lower and upper bound of outliers for {column} column")
        try:
            lower_bound = np.mean(X_train[column]) - 3 * np.std(X_train[column])
            upper_bound = np.mean(X_train[column]) + 3 * np.std(X_train[column])
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
            raise Exception(f"Fail to compute lower and upper bound of outliers for {column} column with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish computing lower and upper bound of outliers for {column} column")
        return lower_bound, upper_bound

    def check_outliers(self, X_train_num, column_list, type):
        '''
            Method Name: check_outliers
            Description: This method computes number and proportion of outliers for every variable based on type of variable
            (gaussian vs non-gaussian) that is categorized from check_gaussian function.
            Output: No output returned. Instead, the results that contains number and proportion of outliers for every variable
            are stored in a csv file named as "Outliers_Info.csv" 
            (One csv file for gaussian variables and another csv file for non gaussian variables)
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f"Start checking outliers for the following columns: {column_list}") 
        outlier_num = []
        outlier_prop = []
        try:
            for column in column_list:
                if type == 'non-gaussian':
                    lower_bound, upper_bound = self.iqr_lower_upper_bound(X_train_num, column)
                elif type == 'gaussian':
                    lower_bound, upper_bound = self.gaussian_lower_upper_bound(X_train_num, column)
                outlier_num.append(len(X_train_num[(X_train_num[column] < lower_bound) | (X_train_num[column] > upper_bound)]))
                outlier_prop.append(np.round(len(X_train_num[(X_train_num[column] < lower_bound) | (X_train_num[column] > upper_bound)])/len(X_train_num),4))
            results = pd.concat([pd.Series(X_train_num[column_list].columns, name='Variable'),pd.Series(outlier_num,name='Number_Outliers'),pd.Series(outlier_prop, name='Prop_Outliers')],axis=1)
            if type == 'non-gaussian':
                results.to_csv(self.result_dir + 'Outliers_Info_Non_Gaussian.csv', index=False)
            elif type == 'gaussian':
                results.to_csv(self.result_dir + 'Outliers_Info_Gaussian.csv', index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to check outliers for the following columns: {column_list} with the following error: {e}")
            raise Exception(f"Fail to check outliers for the following columns: {column_list} with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish checking outliers for the following columns: {column_list}")

    def outlier_capping(self, method, fold, X_train, X_test, y_train, y_test, column_list):
        '''
            Method Name: outlier_capping
            Description: This method caps identified outliers based on given method (either IQR or gaussian)
            Output: Returns four pandas dataframe for features and labels after removing outliers
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f"Start removing outliers for the following columns: {column_list}")
        try:
            trimmer = feo.Winsorizer(capping_method=method, tail='both', fold=fold,variables=column_list)
            train_data = pd.DataFrame(trimmer.fit_transform(pd.concat([X_train, y_train],axis=1)))
            test_data = pd.DataFrame(trimmer.transform(pd.concat([X_test, y_test],axis=1)))
            X_train_cap_outliers = train_data.iloc[:,:-1]
            X_test_cap_outliers = test_data.iloc[:,:-1]
            y_train_cap_outliers = train_data.iloc[:,-1]
            y_test_cap_outliers = test_data.iloc[:,-1]
            pkl.dump(trimmer, open(self.result_dir + f'{method}_outliercapping.pkl','wb'))
        except Exception as e:
            self.log_writer.log(self.file_object, f"Fail to cap outliers for the following columns: {column_list} with the following error: {e}")
            raise Exception(f"Fail to cap outliers for the following columns: {column_list} with the following error: {e}")
        self.log_writer.log(self.file_object, f"Finish capping outliers for the following columns: {column_list}") 
        return X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers, y_test_cap_outliers

    def category_encoding(self, X_train, X_test,y_train):
        '''
            Method Name: category_encooding
            Description: This method performs various categorical data encoding methods based on the given dataset.
            Output: Returns two pandas dataframe for encoded features, while storing encoder objects used in pickle file format.
            On Failure: Logging error and raise exception
        '''
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        count_encoder = ce.CountEncoder(cols = ['Brand'])
        X_train_encoded['Brand'] = count_encoder.fit_transform(X_train['Brand'])
        X_test_encoded['Brand'] = count_encoder.transform(X_test['Brand'])
        pkl.dump(count_encoder, open(self.result_dir + 'CountEncoder.pkl','wb'))
        catboost_encoder = ce.cat_boost.CatBoostEncoder(cols=['MC'])
        X_train_encoded['MC'] = catboost_encoder.fit_transform(X_train_encoded['MC'],y_train)
        X_test_encoded['MC'] = catboost_encoder.transform(X_test_encoded['MC'])
        pkl.dump(catboost_encoder, open(self.result_dir + 'CatBoostEncoder.pkl','wb'))
        X_train_encoded['Month'] = X_train_encoded['Fdate'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").month)
        X_train_encoded['Year'] = X_train_encoded['Fdate'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").year).map({2017: 1, 2018: 2, 2019: 3})
        X_test_encoded['Month'] = X_test_encoded['Fdate'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").month)
        X_test_encoded['Year'] = X_test_encoded['Fdate'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").year).map({2017: 1, 2018: 2, 2019: 3})
        onehot_encoder = fee.OneHotEncoder(variables=['ZONE'],drop_last=True)
        X_train_encoded = onehot_encoder.fit_transform(X_train_encoded)
        X_test_encoded = onehot_encoder.transform(X_test_encoded)
        X_train_encoded = X_train_encoded.drop(['Fdate'],axis=1)
        X_test_encoded = X_test_encoded.drop(['Fdate'],axis=1)
        return X_train_encoded, X_test_encoded
    
    def data_preprocessing(self, start_path, end_path_for_transform, col_remove, target_col):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: Eight pandas dataframe (features and labels for with vs without outliers scenario), 
            where all the data preprocessing tasks are performed.
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.start_path = start_path
        self.end_path_for_transform = end_path_for_transform
        self.col_remove = col_remove
        self.target_col = target_col
        data = self.extract_compiled_data(self.start_path)
        cleaned_data = self.data_cleaning(data)
        cleaned_data = self.remove_irrelevant_columns(cleaned_data, self.col_remove)
        cleaned_data = self.remove_duplicated_rows(cleaned_data)
        X, y = self.features_and_labels(cleaned_data, self.target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_train_num = X_train.select_dtypes(include=['float64','int64'])
        gaussian_columns, non_gaussian_columns = self.check_gaussian(X_train_num)
        pd.Series(gaussian_columns, name='Variable').to_csv(self.result_dir+'Gaussian_columns.csv',index=False)
        pd.Series(non_gaussian_columns, name='Variable').to_csv(self.result_dir+'Non_gaussian_columns.csv',index=False)
        self.check_outliers(X_train_num, non_gaussian_columns, 'non-gaussian')
        X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers, y_test_cap_outliers = self.outlier_capping('iqr', 1.5, X_train, X_test, y_train, y_test, ['GST Value','SP','MRP'])
        X_train_cap_outliers['NSV'] = X_train_cap_outliers['NSV-GST'] + X_train_cap_outliers['GST Value']
        X_train_cap_outliers['NSU'] = X_train_cap_outliers['NSV']/X_train_cap_outliers['SP']
        X_train_cap_outliers['Gross Sales'] = X_train_cap_outliers['NSU'] * X_train_cap_outliers['MRP']
        X_train_cap_outliers['DIS'] = X_train_cap_outliers['MRP'] - X_train_cap_outliers['SP']
        X_train_cap_outliers['DIS%'] = np.round(X_train_cap_outliers['DIS']/X_train_cap_outliers['MRP']*100,2)
        X_test_cap_outliers['NSV'] = X_test_cap_outliers['NSV-GST'] + X_test_cap_outliers['GST Value']
        X_test_cap_outliers['NSU'] = X_test_cap_outliers['NSV']/X_test_cap_outliers['SP']
        X_test_cap_outliers['Gross Sales'] = X_test_cap_outliers['NSU'] * X_test_cap_outliers['MRP']
        X_test_cap_outliers['DIS'] = X_test_cap_outliers['MRP'] - X_test_cap_outliers['SP']
        X_test_cap_outliers['DIS%'] = np.round(X_test_cap_outliers['DIS']/X_test_cap_outliers['MRP']*100,2)
        y_train_cap_outliers = y_train_cap_outliers*X_train['NSU']/X_train_cap_outliers['NSU']
        y_test_cap_outliers = y_test_cap_outliers*X_test['NSU']/X_test_cap_outliers['NSU']
        X_train, X_test = self.category_encoding(X_train, X_test, y_train)
        X_train.to_csv(self.result_dir+'X_train.csv',index=False)
        X_test.to_csv(self.result_dir+'X_test.csv',index=False)
        y_train.to_csv(self.result_dir+'y_train.csv',index=False)
        y_test.to_csv(self.result_dir+'y_test.csv',index=False)
        X_train_cap_outliers, X_test_cap_outliers = self.category_encoding(X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers)
        X_train_cap_outliers.to_csv(self.result_dir+'X_train_cap_outliers.csv',index=False)
        X_test_cap_outliers.to_csv(self.result_dir+'X_test_cap_outliers.csv',index=False)
        y_train_cap_outliers.rename('Cost_per_Unit').to_csv(self.result_dir+'y_train_cap_outliers.csv',index=False)
        y_test_cap_outliers.rename('Cost_per_Unit').to_csv(self.result_dir+'y_test_cap_outliers.csv',index=False)
        self.log_writer.log(self.file_object, 'End of data preprocessing')
        return X_train, X_test, y_train, y_test, X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers, y_test_cap_outliers

