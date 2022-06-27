import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Application_Logger.logger import App_Logger
import pickle as pkl
import ast
from kneed import KneeLocator
import optuna
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence as olsi
random_state=42

class model_trainer:
    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_trainer class
            Output: None
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()
    
    def setting_attributes(trial, cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error):
        '''
            Method Name: setting_attributes
            Description: This method sets attributes related to model metric performance for a given trial
            Output: None
        '''
        trial.set_user_attr("train_rmse", cv_results['train_rmse'].mean())
        trial.set_user_attr("train_mae", cv_results['train_mae'].mean())
        trial.set_user_attr("val_rmse", cv_results['test_rmse'].mean())
        trial.set_user_attr("val_mae", cv_results['test_mae'].mean())
        trial.set_user_attr("train_val_rmse", train_val_root_mean_squared_error)
        trial.set_user_attr("test_rmse", test_root_mean_squared_error)
        trial.set_user_attr("train_val_mae", train_val_mean_absolute_error)
        trial.set_user_attr("test_mae", test_mean_absolute_error)

    def huber_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: huber_objective
            Description: This method sets the objective function for HuberRegressor model by setting various 
            hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error of 
            given model on validation set from using 5 fold cross validation
        '''
        epsilon = trial.suggest_float('epsilon',1,2)
        alpha = trial.suggest_float('alpha',0.0001,0.1)
        reg = HuberRegressor(max_iter=10000, epsilon=epsilon, alpha=alpha)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae
    
    def lasso_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: lasso_objective
            Description: This method sets the objective function for Lasso model by setting various hyperparameters 
            for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        alpha = trial.suggest_float('alpha',0.1,1)
        reg = Lasso(max_iter=10000, alpha=alpha)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def dt_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: dt_objective
            Description: This method sets the objective function for DecisionTreeRegressor model by setting 
            various hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        reg = DecisionTreeRegressor(random_state=random_state)
        path = reg.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[-int(len(path.ccp_alphas)*0.0025):-1])
        reg = DecisionTreeRegressor(random_state=random_state, ccp_alpha=ccp_alpha,max_features='sqrt',max_depth=10)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def rf_objective(trial,X_train_data,y_train_data, X_test_data, y_test_data):
        '''
            Method Name: rf_objective
            Description: This method sets the objective function for RandomForestRegressor model by setting 
            various hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        reg = DecisionTreeRegressor(random_state=random_state)
        path = reg.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[-int(len(path.ccp_alphas)*0.0025):-1])
        reg = RandomForestRegressor(random_state=random_state, ccp_alpha = ccp_alpha, max_features='sqrt',n_jobs=-1,max_depth=10)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def et_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: et_objective
            Description: This method sets the objective function for ExtraTreesRegressor model by setting 
            various hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        reg = DecisionTreeRegressor(random_state=random_state)
        path = reg.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[-int(len(path.ccp_alphas)*0.0025):-1])
        reg = ExtraTreesRegressor(random_state=random_state, ccp_alpha=ccp_alpha,max_features='sqrt',n_jobs=-1,max_depth=10)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def adaboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: adaboost_objective
            Description: This method sets the objective function for AdaBoostRegressor model by setting various 
            hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        learning_rate = trial.suggest_float('learning_rate',0.001,2)
        loss = trial.suggest_categorical('loss',['linear','square','exponential'])
        reg = AdaBoostRegressor(learning_rate=learning_rate, loss=loss, random_state=random_state)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def gradientboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: gradientboost_objective
            Description: This method sets the objective function for GradientBoostRegressor model by setting 
            various hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        reg = DecisionTreeRegressor(random_state=random_state)
        path = reg.cost_complexity_pruning_path(X_train_data, y_train_data)
        ccp_alpha = trial.suggest_categorical('ccp_alpha',path.ccp_alphas[-int(len(path.ccp_alphas)*0.0025):-1])
        reg = GradientBoostingRegressor(random_state=random_state, ccp_alpha=ccp_alpha,max_depth=10,max_features='sqrt')        
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae

    def xgboost_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: xgboost_objective
            Description: This method sets the objective function for XGBoostRegressor model by setting 
            various hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        booster = trial.suggest_categorical('booster',['gbtree','gblinear'])
        eta = trial.suggest_float('eta',0,1)
        gamma = trial.suggest_float('gamma',1,50)
        min_child_weight = trial.suggest_float('min_child_weight',1,50)
        max_delta_step = trial.suggest_int('max_delta_step',1,10)
        colsample_bytree = trial.suggest_float('colsample_bytree',0,1)
        colsample_bylevel = trial.suggest_float('colsample_bylevel',0,1)
        colsample_bynode = trial.suggest_float('colsample_bynode',0,1)
        lambdas = trial.suggest_float('lambda',1,2)
        alpha = trial.suggest_float('alpha',0,1)
        subsample = trial.suggest_float('subsample',0.5,1)
        reg = XGBRegressor(eval_metric='rmse', verbosity=0,max_depth=10, booster=booster, eta=eta, gamma=gamma, 
        min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample, 
        colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode, 
        lambdas=lambdas, alpha=alpha, random_state=random_state)
        cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error = model_trainer.regression_metrics(reg,X_train_data,y_train_data,X_test_data, y_test_data)
        rmse, mae = cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()
        model_trainer.setting_attributes(trial,cv_results,train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error)
        return rmse, mae        

    def ann_objective(trial,X_train_data,y_train_data,X_test_data,y_test_data):
        '''
            Method Name: ann_objective
            Description: This method sets the objective function for Sequential model by setting various 
            hyperparameters for different Optuna trials.
            Output: Two floating point values that represents root mean squared error and mean absolute error 
            of given model on validation set from using 5 fold cross validation
        '''
        cont_model = Sequential()
        hp_units = trial.suggest_int('units', 32, 640, 32)
        activation_options = trial.suggest_categorical('activation',['relu','selu','elu'])
        hidden_kernels = trial.suggest_categorical('hidden_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal'])
        output_kernels = trial.suggest_categorical('output_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal'])
        batch_norm = trial.suggest_categorical('batch_normalization',[True,False])
        dropout = trial.suggest_categorical('dropout',[True,False])
        dropout_rate = trial.suggest_float('dropoutrate',0.1,0.5,step=0.1)
        cont_model.add(Dense(units = hp_units, input_dim=X_train_data.shape[1], activation=activation_options, kernel_initializer = hidden_kernels))
        if batch_norm == True:
            cont_model.add(layers.BatchNormalization())
        if dropout == True:
            cont_model.add(layers.Dropout(rate=dropout_rate))
        for i in range(trial.suggest_int('extra_hidden_layers', 0, 2)):
            cont_model.add(layers.Dense(units=trial.suggest_int('units_' + str(i+1), 32, 640, 32), 
            activation=activation_options, kernel_initializer = hidden_kernels))
            if batch_norm == True:
                cont_model.add(layers.BatchNormalization())
            if dropout == True:
                cont_model.add(layers.Dropout(rate=dropout_rate))
        cont_model.add(Dense(1, activation="linear", kernel_initializer = output_kernels))
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_data, y_train_data, test_size=0.25, random_state=random_state)
        val_model = tf.keras.models.clone_model(cont_model)
        val_model.compile(loss='mse', optimizer= "adam", metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        val_model.fit(np.array(X_train_sub), np.array(y_train_sub), epochs=5,batch_size=512,verbose=1)
        train_root_mean_squared_error = mean_squared_error(np.array(y_train_sub),val_model.predict(np.array(X_train_sub)),squared=False)
        val_root_mean_squared_error = mean_squared_error(np.array(y_val_sub),val_model.predict(np.array(X_val_sub)),squared=False)
        train_mean_absolute_error = mean_absolute_error(np.array(y_train_sub),val_model.predict(np.array(X_train_sub)))
        val_mean_absolute_error = mean_absolute_error(np.array(y_val_sub),val_model.predict(np.array(X_val_sub)))
        trial.set_user_attr("train_rmse", train_root_mean_squared_error)
        trial.set_user_attr("train_mae", train_mean_absolute_error)
        trial.set_user_attr("val_rmse", val_root_mean_squared_error)
        trial.set_user_attr("val_mae", val_mean_absolute_error)
        test_model = tf.keras.models.clone_model(cont_model)
        test_model.compile(loss='mse', optimizer= "adam", metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        test_model.fit(np.array(X_train_data), np.array(y_train_data), epochs=100,batch_size=512,verbose=1,validation_data=(np.array(X_test_data), np.array(y_test_data)),callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=10)])
        train_val_root_mean_squared_error = mean_squared_error(np.array(y_train_data),test_model.predict(np.array(X_train_data)),squared=False)
        test_root_mean_squared_error = mean_squared_error(np.array(y_test_data),test_model.predict(np.array(X_test_data)),squared=False)
        train_val_mean_absolute_error = mean_absolute_error(np.array(y_train_data),test_model.predict(np.array(X_train_data)))
        test_mean_absolute_error = mean_absolute_error(np.array(y_test_data),test_model.predict(np.array(X_test_data)))
        trial.set_user_attr("train_val_rmse", train_val_root_mean_squared_error)
        trial.set_user_attr("test_rmse", test_root_mean_squared_error)
        trial.set_user_attr("train_val_mae", train_val_mean_absolute_error)
        trial.set_user_attr("test_mae", test_mean_absolute_error)
        return val_root_mean_squared_error, val_mean_absolute_error

    def regression_metrics(reg,X_train_data,y_train_data, X_test_data, y_test_data):
        '''
            Method Name: regression_metrics
            Description: This method sets the objective function for Sequential model by setting various 
            hyperparameters for different Optuna trials.
            Output: Result from 5-fold cross validation in dictionary format and four different floating point values 
            that represents root mean squared error and mean absolute error of given model on train-val set and test set.
        '''
        cv_results = cross_validate(reg, X_train_data, y_train_data, cv=5, return_train_score=True,
        scoring={"rmse": make_scorer(mean_squared_error, squared=False), "mae": make_scorer(mean_absolute_error)})
        reg.fit(np.array(X_train_data),np.array(y_train_data))
        train_val_root_mean_squared_error = mean_squared_error(np.array(y_train_data),reg.predict(np.array(X_train_data)),squared=False)
        test_root_mean_squared_error = mean_squared_error(np.array(y_test_data),reg.predict(np.array(X_test_data)),squared=False)
        train_val_mean_absolute_error = mean_absolute_error(np.array(y_train_data),reg.predict(np.array(X_train_data)))
        test_mean_absolute_error = mean_absolute_error(np.array(y_test_data),reg.predict(np.array(X_test_data)))
        return cv_results, train_val_root_mean_squared_error, test_root_mean_squared_error, train_val_mean_absolute_error, test_mean_absolute_error

    def initialize_model_training(self,folderpath,filepath):
        '''
            Method Name: initialize_model_training
            Description: This method initializes the model training process by creating a new CSV file 
            for storing model results, while setting the list of objective functions with its corresponding model 
            objects for model training.
            Output: Two list objects that represent objective functions with its corresponding initial model objects.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, 'Start initializing objectives required for model training')
        self.filepath = filepath
        objectives = [model_trainer.huber_objective, model_trainer.lasso_objective, model_trainer.dt_objective, 
        model_trainer.rf_objective, model_trainer.et_objective,model_trainer.adaboost_objective, 
        model_trainer.gradientboost_objective, model_trainer.xgboost_objective, model_trainer.ann_objective]
        selectors = [HuberRegressor(max_iter=10000),Lasso(max_iter=10000),
        DecisionTreeRegressor(max_features='sqrt',max_depth=10), RandomForestRegressor(max_features='sqrt',n_jobs=-1,max_depth=10),
        ExtraTreesRegressor(max_features='sqrt',n_jobs=-1,max_depth=10),AdaBoostRegressor(),
        GradientBoostingRegressor(max_depth=10,max_features='sqrt'),XGBRegressor(max_depth=10),Sequential()]
        try:
            results = pd.concat([pd.Series(name='column_list'), pd.Series(name='num_features'), pd.Series(name='model_name'), 
            pd.Series(name='best_params'), pd.Series(name='clustering_indicator'),
            pd.Series(name='train_rmse'), pd.Series(name='val_rmse'), pd.Series(name='train_mae'),pd.Series(name='val_mae'), 
            pd.Series(name='train_val_rmse'), pd.Series(name='test_rmse'), pd.Series(name='train_val_mae'),
            pd.Series(name='test_mae')], axis=1)
            results.to_csv(folderpath+filepath, mode='w',index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to create initial csv file of results from model training with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, 'Finish initializing objectives required for model training')
        return objectives, selectors

    def fit_scaled_data(self, data, scaler_type):
        '''
            Method Name: fit_scaled_data
            Description: This method compute the necessary values required to be used for data scaling based 
            on the type of scaler. (i.e. mean and standard deviation for Standard Scaler or min and max for MinMaxScaler 
            or median and quantiles for RobustScaler)
            Output: Fitted scaler object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start fitting scaler on dataset')
        try:
            scaler_type = scaler_type.fit(data)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fitting scaler on dataset failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish fitting scaler on dataset')
        return scaler_type

    def transform_scaled_data(self, data, scaler_type):
        '''
            Method Name: transform_scaled_data
            Description: This method centers and scale the data based on a given fitted scaler object.
            Output: A pandas dataframe of scaled data
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start transforming scaler on dataset')
        try:
            data_scaled = pd.DataFrame(scaler_type.transform(data), columns=data.columns)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Transforming scaler on dataset failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish transforming scaler on dataset')
        return data_scaled

    def scale_vs_non_scale_data(self,reg,X_train_scaled,X_train,y_train_scaled,y_train):
        '''
            Method Name: scale_vs_non_scale_data
            Description: This method selects between scaled vs non-scaled data based on the type of model used for training.
            Output: Two pandas dataframe of feature and label data.
            On Failure: Logging error and raise exception
        '''
        if type(reg).__name__ in ['HuberRegressor','Lasso','Sequential']:
            X_train_data = X_train_scaled
            y_train_data = y_train_scaled
        else:
            X_train_data = X_train
            y_train_data = y_train
        X_train_data = X_train_data.reset_index(drop=True)
        y_train_data = y_train_data.reset_index(drop=True)
        return X_train_data, y_train_data

    def optuna_optimizer(self, obj, n_trials):
        '''
            Method Name: optuna_optimizer
            Description: This method creates a new Optuna study object and optimizes the given objective function.
            Output: List of frozen trial objects from optimizing a given objective function.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start performing optuna hyperparameter tuning for {obj.__name__} model')
        try:
            study = optuna.create_study(directions=['minimize','minimize'])
            study.optimize(obj, n_trials=n_trials, n_jobs=-1)
            trials = study.best_trials
        except Exception as e:
            self.log_writer.log(self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing optuna hyperparameter tuning for {obj.__name__} model')
        return trials
    
    def train_per_model(self,obj, reg, trial_size,col_list, n_features,X_train,y_train,X_test,y_test,num_features, col_selected, model_name, best_params, clustering_yes_no, train_rmse,val_rmse,train_mae,val_mae,train_val_rmse,test_rmse,train_val_mae,test_mae, clustering):
        '''
            Method Name: train_per_model
            Description: This method stores the following types of results required in respective list objects by 
            batches before storing these results in CSV file using store_tuning_results function.
            1. Number of features
            2. Column names selected
            3. Model name
            4. Model parameters
            5. Clustering (Yes vs No)
            6. Train Root Mean Squared Error (RMSE)
            7. Validation RMSE
            8. Train Mean Absolute Error (MAE)
            9. Validation MAE
            10. Train-Validation RMSE
            11. Test RMSE
            12. Train-Validation MAE
            13. Test MAE
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start model training on {type(reg).__name__} for {n_features} features with {clustering} clustering')
        try:
            func = lambda trial: obj(trial, np.array(X_train), np.array(y_train),np.array(X_test),np.array(y_test))
            func.__name__ = type(reg).__name__
            if func.__name__ == 'Sequential':
                model = tf.keras.models.clone_model(reg)
            else:
                model = clone(reg)
            trials = self.optuna_optimizer(func,trial_size)
            for trial in trials:
                num_features.append(n_features)
                col_selected.append(col_list)
                model_name.append(type(model).__name__)
                if func.__name__ == 'Sequential':
                    best_params.append(trial.params)
                else:
                    best_params.append(model.set_params(**trial.params).get_params())
                clustering_yes_no.append(clustering)
                train_rmse.append(trial.user_attrs['train_rmse'])
                val_rmse.append(trial.user_attrs['val_rmse'])
                train_mae.append(trial.user_attrs['train_mae'])
                val_mae.append(trial.user_attrs['val_mae'])
                train_val_rmse.append(trial.user_attrs['train_val_rmse'])
                test_rmse.append(trial.user_attrs['test_rmse'])                
                train_val_mae.append(trial.user_attrs['train_val_mae'])
                test_mae.append(trial.user_attrs['test_mae']) 
                self.log_writer.log(self.file_object, f"Results for {type(model).__name__} with {n_features} features and {clustering} clustering saved for trial {trial.number}")
        except Exception as e:
            self.log_writer.log(self.file_object, f'Model training on {type(reg).__name__} for {n_features} features with {clustering} clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish model training on {type(reg).__name__} for {n_features} features with {clustering} clustering')

    def store_tuning_results(self,col_selected,num_features,model_name,best_params,clustering_yes_no, train_rmse,val_rmse,train_mae,val_mae,train_val_rmse,test_rmse,train_val_mae,test_mae,folderpath,filepath,n_features):
        '''
            Method Name: store_tuning_results
            Description: This method stores the following list objects in CSV file for model evaluation:
            1. Number of features
            2. Column names selected
            3. Model name
            4. Model parameters
            5. Clustering (Yes vs No)
            6. Train Root Mean Squared Error (RMSE)
            7. Validation RMSE
            8. Train Mean Absolute Error (MAE)
            9. Validation MAE
            10. Train-Validation RMSE
            11. Test RMSE
            12. Train-Validation MAE
            13. Test MAE
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start appending results from model training for {n_features} features')
        try:
            results = pd.concat([pd.Series(col_selected, name='column_list'), pd.Series(num_features, name='num_features'), 
                                pd.Series(model_name, name='model_name'), pd.Series(best_params, name='best_params'),
                                pd.Series(clustering_yes_no, name='clustering_indicator'),
                                pd.Series(train_rmse, name='train_rmse'), pd.Series(val_rmse, name='val_rmse'), 
                                pd.Series(train_mae, name='train_mae'), pd.Series(val_mae, name='val_mae'),
                                pd.Series(train_val_rmse, name='train_val_rmse'),pd.Series(test_rmse, name='test_rmse'),
                                pd.Series(train_val_mae, name='train_val_mae'),pd.Series(test_mae, name='test_mae')], axis=1)
            results.to_csv(folderpath+filepath, mode='a',header=False, index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Appending results from model training for {n_features} features failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish appending results from model training for {n_features} features')

    def best_model(self, folderpath, filepath, bestresultpath, threshold):
        '''
            Method Name: best_model
            Description: This method identifies the best model to use for model deployment based on RMSE and MAE 
            performance metrics with a pre-defined threshold of difference allowed between model performance 
            on training set and test set.
            Output: A pandas dataframe that contains information about the best model identified from model evaluation.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start determining best configuration to use for saving models')
        try:
            results = pd.read_csv(folderpath+filepath)
            results = results[(np.abs(results['train_val_rmse'] - results['test_rmse'])/results['train_val_rmse'] < threshold) & (np.abs(results['train_val_mae'] - results['test_mae'])/results['train_val_mae'] < threshold)]
            final_models = results[(results['test_rmse'] == results['test_rmse'].min()) & (results['test_mae'] == results['test_mae'].min())].sort_values(by=['num_features','clustering_indicator'])
            # If no model performs best for both metrics, then pick the model with the lowest root mean squared error
            if len(final_models) == 0:
                final_models = results[(results['test_rmse'] == results['test_rmse'].min())].sort_values(by=['num_features','clustering_indicator'])
            pd.DataFrame(final_models, columns = final_models.columns).to_csv(folderpath+bestresultpath,index=False)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Fail to determine best configuration to use for saving models with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish determining best configuration to use for saving models')
        return final_models

    def k_means_clustering(self, data, start_cluster, end_cluster):
        '''
            Method Name: k_means_clustering
            Description: This method performs K-means clustering on given data, while identifying the most suitable number
            of clusters to use for the given dataset.
            Output: KneeLocator object and KMeans object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start deriving best number of clusters from k-means clustering')
        wcss=[]
        try:
            for i in range (start_cluster,end_cluster+1):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=random_state)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            kneeloc = KneeLocator(range(start_cluster,end_cluster+1), wcss, curve='convex', direction='decreasing')
            kmeans=KMeans(n_clusters=kneeloc.knee,init='k-means++',random_state=random_state)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Deriving best number of clusters from k-means clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish deriving best number of clusters from k-means clustering')
        return kneeloc, kmeans

    def add_cluster_number_to_data(self,train_scaled_data, train_data, test_scaled_data, test_data, final=False):
        '''
            Method Name: add_cluster_number_to_data
            Description: This method performs K-means clustering on given scaled data, while identifying the most suitable number
            of clusters to use for the given dataset and adding the cluster number to the existing scaled and unscaled data.
            Output: Four pandas dataframe consist of training and testing data for both scaled and unscaled data. Note that a pickle object will 
            be created if the best model identified requires K-means clustering.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start performing data clustering')
        try:
            kneeloc, kmeans = self.k_means_clustering(train_scaled_data, 1, 10)
            train_scaled_data['cluster'] = kmeans.fit_predict(train_scaled_data)
            train_data['cluster'] = train_scaled_data['cluster']
            test_scaled_data['cluster'] = kmeans.predict(test_scaled_data)
            test_data['cluster'] = test_scaled_data['cluster']
            train_scaled_data.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)
            test_scaled_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            if final == True:
                pkl.dump(kmeans, open('Saved_Models/kmeans_model.pkl', 'wb'))
        except Exception as e:
            self.log_writer.log(self.file_object, f'Performing data clustering failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing data clustering that contains {kneeloc.knee} clusters')
        return train_data, test_data, train_scaled_data, test_scaled_data

    def data_scaling_train_test(self, folderpath, train_data, test_data):
        '''
            Method Name: data_scaling_train_test
            Description: This method performs feature scaling on training and testing data using Robust Scaler method
            Output: Two pandas dataframe consist of training and testing scaled data.
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start performing scaling data on train and test set')
        try:
            scaler = self.fit_scaled_data(train_data, RobustScaler())
            train_data_scaled = self.transform_scaled_data(train_data, scaler)
            test_data_scaled = self.transform_scaled_data(test_data, scaler)
            pkl.dump(scaler, open(folderpath+'RobustScaler.pkl', 'wb'))
        except Exception as e:
            self.log_writer.log(self.file_object, f'Data scaling on training and test set failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish performing scaling data on train and test set')
        return train_data_scaled, test_data_scaled

    def learning_curve_plot(self,folderpath, train_size=None, train_score_m=None, test_score_m=None, history=None, Sequential=False):
        '''
            Method Name: learning_curve_plot
            Description: This method plots learning curve of a given model and saves the figure plot in a given folder path.
            Note that Sequential model involves using a slightly different learning curve, where number of epochs is used.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start plotting learning curve')
        try:
            if Sequential == False:
                fig1, ax1 = plt.subplots()
                ax1.plot(train_size, train_score_m, 'o-', color="b")
                ax1.plot(train_size, test_score_m, 'o-', color="r")
                ax1.legend(('Training score', 'Test score'), loc='best')
                ax1.set_xlabel("Training Samples")
                ax1.set_ylabel("RMSE")
                ax1.set_title("Learning Curve Analysis (CV=5)")
                ax1.grid()
                ax1.annotate(np.round(train_score_m[-1],4),(train_size[-1]-20,train_score_m[-1]+0.015))
                ax1.annotate(np.round(test_score_m[-1],4),(train_size[-1]-20,test_score_m[-1]-0.015))
                plt.savefig(folderpath+'Learning_Curve_Analysis.png')
            elif Sequential == True:
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = range(1, len(loss) + 1)
                plt.plot(epochs, loss, 'r', label='Training loss')
                plt.plot(epochs, val_loss, 'b', label='Validation loss')
                plt.title('Training and Validation Loss over Number of Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(folderpath+'Learning_Curve_Analysis.png')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Plotting learning curve failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish plotting learning curve')

    def feature_importance_plot(self, folderpath, model, X, y):
        '''
            Method Name: feature_importance_plot
            Description: This method plots feature importances of a given model and saves the figure plot in a given folder path.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start plotting feature importance plot')
        try:
            if type(model).__name__ in ['DecisionTreeRegressor','RandomForestRegressor','ExtraTreesRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor']:
                result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                forest_importances = pd.Series(result.importances_mean, index=X.columns)
                fig, ax = plt.subplots()
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.savefig(folderpath+'Feature_Importance.png')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Plotting feature importance plot failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish plotting feature importance plot')

    def leverage_plot(self, folderpath, final_result, X_sub, y_sub):
        '''
            Method Name: leverage_plot
            Description: This method plots leverage plot of a given model and saves the figure plot in a given folder path.
            Note that data points that are identified as highly influenced are also saved in two different
            CSV files for features and labels respectively.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start plotting leverage plot')
        try:
            df = pd.concat([X_sub, y_sub],axis=1)
            df = df.rename(columns = {'NSV-GST':'Net_NSV','Gross Sales':'Gross_Sales','DIS%':'DIS_percent'})
            columns_sub = final_result['column_list'].values[0].replace("'","").replace("NSV-GST","Net_NSV").replace("Gross Sales","Gross_Sales").replace("DIS%","DIS_percent").strip("][").split(', ')[:final_result['num_features'].values[0]]
            model =smf.ols(formula ='Cost_per_Unit ~ ' + ' + '.join(columns_sub), data=df)
            results = model.fit()
            cook_dist = olsi(results).cooks_distance[0]
            influential_points = cook_dist[cook_dist > 3*np.mean(cook_dist)]
            influential_inputs = X_sub.iloc[influential_points.index]
            influential_outputs = y_sub.iloc[influential_points.index]
            influential_inputs.to_csv(folderpath+'influential_inputs.csv')
            influential_outputs.to_csv(folderpath+'influential_outputs.csv')
            fig, ax = plt.subplots(figsize=(12,8))
            fig = sm.graphics.influence_plot(results, ax = ax)
            plt.savefig(folderpath+'Leverage_Plot.png')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Plotting leverage plot failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish plotting leverage plot')

    def train_overall_model(self, X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, folderpath):
        '''
            Method Name: train_overall_model
            Description: This method trains the model on the entire dataset using the best model identified from model evaluation for model deployment.
            Output: Trained model object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f'Start training and saving the {name_model} model')
        try:
            X_sub = pd.concat([X_train_sub, X_test_sub]).reset_index(drop=True)
            y_sub = pd.concat([y_train_data, y_test_data]).reset_index(drop=True)
            self.leverage_plot(folderpath, final_result, X_sub, y_sub)
            if type(model).__name__ != 'Sequential':
                overall_model = clone(model)
                overall_model.set_params(**eval(final_result['best_params'].values[0].replace("\'missing\': nan,","").replace("'", "\"")))
                train_size, train_score, test_score = learning_curve(estimator=overall_model, X=X_sub, y=y_sub, cv=5, scoring='neg_root_mean_squared_error')
                train_score_m = np.mean(np.abs(train_score), axis=1)
                test_score_m = np.mean(np.abs(test_score), axis=1)
                self.learning_curve_plot(folderpath, train_size, train_score_m, test_score_m)
                overall_model.fit(np.array(X_sub),np.array(y_sub))
                self.feature_importance_plot(folderpath, overall_model, X_sub, y_sub)
            else:
                overall_model = tf.keras.models.clone_model(model)
                hp_units = ast.literal_eval(final_result['best_params'].values[0])['units']
                activation_options = ast.literal_eval(final_result['best_params'].values[0])['activation']
                hidden_kernels = ast.literal_eval(final_result['best_params'].values[0])['hidden_initializer']
                output_kernels = ast.literal_eval(final_result['best_params'].values[0])['output_initializer']
                batch_norm = ast.literal_eval(final_result['best_params'].values[0])['batch_normalization']
                dropout = ast.literal_eval(final_result['best_params'].values[0])['dropout']
                dropout_rate = ast.literal_eval(final_result['best_params'].values[0])['dropoutrate']
                overall_model.add(Dense(units = hp_units, input_dim=X_sub.shape[1], activation=activation_options, kernel_initializer = hidden_kernels))
                if batch_norm == True:
                    overall_model.add(layers.BatchNormalization())
                if dropout == True:
                    overall_model.add(layers.Dropout(rate=dropout_rate))
                for i in range(ast.literal_eval(final_result['best_params'].values[0])['extra_hidden_layers']):
                    overall_model.add(layers.Dense(units=ast.literal_eval(final_result['best_params'].values[0])['units_' + str(i+1)], 
                    activation=activation_options, kernel_initializer = hidden_kernels))
                    if batch_norm == True:
                        overall_model.add(layers.BatchNormalization())
                    if dropout == True:
                        overall_model.add(layers.Dropout(rate=dropout_rate))
                overall_model.add(Dense(1, activation="linear", kernel_initializer = output_kernels))
                overall_model.compile(loss='mse', optimizer= "adam", metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
                result = overall_model.fit(np.array(X_train_sub), np.array(y_train_data), epochs=100,batch_size=512,verbose=1,validation_data=(np.array(X_test_sub), np.array(y_test_data)),callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=10)])
                self.learning_curve_plot(folderpath, train_size=None, train_score_m=None, test_score_m=None, history = result, Sequential = True)
                overall_model.fit(np.array(X_sub), np.array(y_sub), epochs=100,batch_size=512,verbose=1)
        except Exception as e:
            self.log_writer.log(self.file_object, f'Training and saving the {name_model} model failed with the following error: {e}')
            raise Exception()
        self.log_writer.log(self.file_object, f'Finish training and saving the {name_model} model')
        return overall_model

    def train_model_and_hyperparameter_tuning(self, train_input, test_input, train_output, test_output, train_input_cap_outliers, test_input_cap_outliers, train_output_cap_outliers, test_output_cap_outliers, folderpath, filepath, bestresultpath, threshold):
        '''
            Method Name: train_model_and_hyperparameter_tuning
            Description: This method performs all the model training and hyperparameter tuning tasks for the data.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, 'Start model training and hyperparameter tuning')
        self.train_input = train_input
        self.test_input = test_input
        self.train_output = train_output
        self.test_output = test_output
        self.train_input_cap_outliers = train_input_cap_outliers
        self.test_input_cap_outliers = test_input_cap_outliers
        self.train_output_cap_outliers = train_output_cap_outliers
        self.test_output_cap_outliers = test_output_cap_outliers
        self.folderpath = folderpath
        self.filepath = filepath
        self.bestresultpath = bestresultpath
        self.threshold = threshold
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        objectives, selectors = self.initialize_model_training(self.folderpath,self.filepath)
        X_train_scaled_cap_outliers, X_test_scaled_cap_outliers = self.data_scaling_train_test(self.folderpath, self.train_input_cap_outliers, self.test_input_cap_outliers)
        for n_features in range(1,min(21,len(X_train_scaled_cap_outliers.columns)+1)):
            for obj, reg in zip(objectives, selectors):
                num_features, col_selected, model_name, best_params, clustering_yes_no  = [], [], [], [], []
                train_rmse, val_rmse, train_mae, val_mae, train_val_rmse, test_rmse, train_val_mae, test_mae  = [], [], [], [], [], [], [], []
                X_train_data, y_train_data = self.scale_vs_non_scale_data(reg,X_train_scaled_cap_outliers,self.train_input,self.train_output_cap_outliers,self.train_output)
                X_test_data, y_test_data = self.scale_vs_non_scale_data(reg,X_test_scaled_cap_outliers,self.test_input,self.test_output_cap_outliers,self.test_output)                       
                transformer = SelectKBest(f_regression, k=n_features)
                X_train_sub = pd.DataFrame(transformer.fit_transform(X_train_data, y_train_data), columns=transformer.get_feature_names_out())
                X_test_sub = pd.DataFrame(transformer.transform(X_test_data), columns=transformer.get_feature_names_out())
                for clustering in ['yes', 'no']:
                    col_list_with_outliers = list(transformer.get_feature_names_out())
                    if clustering == 'yes':
                        X_train_sub_temp = X_train_sub.copy()
                        X_test_sub_temp = X_test_sub.copy()
                        X_train_scaled_sub = pd.DataFrame(transformer.transform(X_train_scaled_cap_outliers), columns = col_list_with_outliers)
                        X_test_scaled_sub = pd.DataFrame(transformer.transform(X_test_scaled_cap_outliers), columns = col_list_with_outliers)
                        X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub_temp, X_test_scaled_sub, X_test_sub_temp)
                        X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(reg,X_train_scaled_cluster_sub,X_train_cluster_sub,self.train_output_cap_outliers,y_train_data)
                        X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(reg,X_test_scaled_cluster_sub,X_test_cluster_sub,self.test_output_cap_outliers,y_test_data) 
                        col_list_with_outliers.extend(pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True).columns.tolist())
                        X_train_cluster_data = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
                        X_test_cluster_data = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
                        self.train_per_model(obj, reg, 50, col_list_with_outliers, n_features,X_train_cluster_data,y_train_data,X_test_cluster_data,
                        y_test_data,num_features, col_selected, model_name, best_params, clustering_yes_no, train_rmse, val_rmse, 
                        train_mae, val_mae, train_val_rmse, test_rmse, train_val_mae, test_mae, clustering)
                    else:
                        self.train_per_model(obj, reg, 50, col_list_with_outliers, n_features,X_train_sub,y_train_data,X_test_sub,
                        y_test_data,num_features, col_selected, model_name, best_params, clustering_yes_no, train_rmse, 
                        val_rmse, train_mae, val_mae, train_val_rmse, test_rmse, train_val_mae, test_mae, clustering)
                self.store_tuning_results(col_selected,num_features,model_name,best_params, clustering_yes_no, train_rmse, val_rmse, 
                train_mae, val_mae, train_val_rmse, test_rmse, train_val_mae, test_mae, self.folderpath, self.filepath,n_features)
        
        final_result = self.best_model(self.folderpath, self.filepath, self.bestresultpath, self.threshold)
        name_model = final_result['model_name'].values[0]
        num_features = final_result['num_features'].values[0]
        clustering = final_result['clustering_indicator'].values[0]
        model_dict = {'HuberRegressor':HuberRegressor(),'Lasso':Lasso(),'DecisionTreeRegressor':DecisionTreeRegressor(),
        'RandomForestRegressor':RandomForestRegressor(),'ExtraTreesRegressor':ExtraTreesRegressor(),
        'AdaBoostRegressor':AdaBoostRegressor(),'GradientBoostingRegressor':GradientBoostingRegressor(),
        'XGBRegressor':XGBRegressor(),'Sequential':Sequential()}
        model = model_dict[name_model]
        X_train_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_scaled_cap_outliers,self.train_input,self.train_output_cap_outliers,self.train_output)
        X_test_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled_cap_outliers,self.test_input,self.test_output_cap_outliers,self.test_output)
        columns = final_result['column_list'].values[0].replace("'","").strip("][").split(', ')[:num_features]
        X_train_scaled_sub = X_train_scaled_cap_outliers[columns]
        X_test_scaled_sub = X_test_scaled_cap_outliers[columns]
        X_train_sub = X_train_data[columns]
        X_test_sub = X_test_data[columns]

        if clustering == 'yes':
            X_train_cluster_sub, X_test_cluster_sub, X_train_scaled_cluster_sub, X_test_scaled_cluster_sub = self.add_cluster_number_to_data(X_train_scaled_sub, X_train_sub, X_test_scaled_sub, X_test_sub, final=True)
            X_train_cluster_data, y_train_data = self.scale_vs_non_scale_data(model,X_train_scaled_cluster_sub,X_train_cluster_sub,self.train_output_cap_outliers,y_train_data)
            X_test_cluster_data, y_test_data = self.scale_vs_non_scale_data(model,X_test_scaled_cluster_sub,X_test_cluster_sub,self.test_output_cap_outliers,y_test_data) 
            X_train_sub = pd.concat([X_train_cluster_data.iloc[:,:-1], pd.get_dummies(X_train_cluster_data['cluster'], drop_first=True)],axis=1)
            X_test_sub = pd.concat([X_test_cluster_data.iloc[:,:-1], pd.get_dummies(X_test_cluster_data['cluster'], drop_first=True)],axis=1)
        trained_model = self.train_overall_model(X_train_sub, X_test_sub, y_train_data, y_test_data, model, final_result, name_model, self.folderpath)
        if type(model).__name__ != 'Sequential':
            pkl.dump(trained_model,open('Saved_Models/'+name_model+'_'+clustering+'_clustering.pkl','wb'))
        else:
            trained_model.save('Saved_Models/'+name_model+'_'+clustering+'_clustering.h5')
        self.log_writer.log(self.file_object, 'Finish model training and hyperparameter tuning')