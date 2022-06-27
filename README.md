# Wafer Status Classification Project

**Background**
---

<img src="https://www.semiconductorforu.com/wp-content/uploads/2021/02/silicon-wafer.jpg">

In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor used for the fabrication of integrated circuits. Monitoring working conditions of these wafers present its challenges of having additional resources required for manual monitoring with insights and decisions that need to be made quickly for replacing wafers that are not in good working conndition when required. Using IIOT (Industrial Internet of Things) helps to overcome this challenge through a collection of real-time data from multiple sensors. 

Thus, the main goal of this project is to design a machine learning model that predicts whether a wafer is in a good working condition or not based on inputs from 590 different sensors for every wafer. The quality of wafer sensors can be classified into two different categories: 0 for "good wafer" and 1 for "bad wafer".

Dataset is provided in .csv format by client under <b>Training_Batch_Files</b> folder for model training, while dataset under <b>Prediction_Batch_Files</b> folder will be used for predicting quality of wafer sensors.

In addition, schema of datasets for training and prediction is provided in .json format by the client for storing seperate csv files into a single MySQL database.

**Code and Resources Used**
---
- **Python Version** : 3.9.10
- **Packages** : feature-engine, imbalanced-learn, kneed, matplotlib, mysql-connector-python, numpy, optuna, pandas, scikit-learn, scipy, seaborn, streamlit, tensorflow, keras
- **Dataset source** : Education materials from OneNeuron platform
- **Streamlit documentation** : https://spotipy.readthedocs.io/en/2.16.0/
- **Optuna documentation** : https://spotipy.readthedocs.io/en/2.16.0/
- **Tensorflow documentation**: https://www.tensorflow.org/api_docs

**CRISP-DM Methodology**
---
For any given Machine Learning projects, CRISP-DM (Cross Industry Standard Practice for Data Mining) methodology is the most commonly adapted methodology used.
The following diagram below represents a simple summary of the CRISP-DM methodology for this project:

<img src="https://www.datascience-pm.com/wp-content/uploads/2018/09/crisp-dm-wikicommons.jpg" width="450" height="400">

Note that an alternative version of this methodology, known as CRISP-ML(Q) (Cross Industry Standard Practice for Machine Learning and Quality Assurance) can also be used in this project. However, the model monitoring aspect is not used in this project, which can be considered for future use.

**Project Architecture Summary**
---
The following diagram below summarizes the structure for this project:

![image](https://user-images.githubusercontent.com/34255556/164873790-34d8826f-2acc-43c9-9d7c-6aafd2e2b355.png)

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> and <b>Prediction_Logs</b> folders. Any bad quality data identified for model training and model prediction will be archived accordingly in <b>Archive_Training_Data</b> and <b>Archive_Prediction_Data</b> folders.

**Project Folder Structure**
---
The following points below summarizes the use of every file/folder available for this project:
1. Application_Logger: Helper module for logging model training and prediction process
2. Archive_Prediction_Data: Stores bad quality prediction csv files that have been used previously for model prediction
3. Archive_Training_Data: Stores bad quality training csv files that have been used previously for model training
4. Bad_Prediction_Data: Temporary folder for identifying bad quality prediction csv files
5. Bad_Training_Data: Temporary folder for identifying bad quality prediction csv files
6. Good_Prediction_Data: Temporary folder for identifying good quality prediction csv files
7. Good_Training_Data: Temporary folder for identifying good quality training csv files
8. Intermediate_Pred_Results: Stores results from model prediction
9. Intermediate_Train_Results: Stores additional information from model training process
10. Model_Prediction_Modules: Helper modules for model prediction
11. Model_Training_Modules: Helper modules for model training
12. Prediction_Batch_Files: Stores csv batch files to be used for model prediction
13. Prediction_Data_FromDB: Stores compiled data from SQL database for model prediction
14. Prediction_Logs: Stores logging information from model prediction for future debugging and maintenance
15. Saved_Models: Stores best models identified from model training process for model prediction
16. Training_Batch_Files: Stores csv batch files to be used for model training
17. Training_Data_FromDB: Stores compiled data from SQL database for model training
18. Training_Logs: Stores logging information from model training for future debugging and maintenance
19. Dockerfile: Additional file for Docker model deployment
20. main.py: Main file for program execution
21. Procfile: Additional file for Heroku model deployment
22. README.md: Details summary of project for presentation
23. requirements.txt: List of Python packages to install for model deployment
24. schema_prediction.json: JSON file that contains database schema for model prediction
25. schema_training.json: JSON file that contains database schema for model training
26. setup.sh : Additional file for Heroku model deployment

The following sections below explains the three main approaches that can be used for model deployment in this project:
1. <b>Cloud Platform (Heroku)</b>
2. <b>Local system</b>
3. <b>Docker</b>

**Project Instructions (Heroku)**
---
<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- Procfile
- requirements.txt
- setup.sh

<b>Note that deploying models onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Delete files stored inside Training_Logs and Prediction_Logs folder, while creating a dummy.txt file inside both folders respectively. This is to ensure both directories exist when the model is deployed into Heroku.
<img src="https://user-images.githubusercontent.com/34255556/160224012-4f861309-1e7a-40ad-b466-dbdc8e22f20e.png" width="600" height="80">

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. Go to "Resources" tab and search for ClearDB MySQL in the add-ons search bar.
<img src="https://user-images.githubusercontent.com/34255556/160224064-35295bf6-3170-447a-8eae-47c6721cf8f0.png" width="600" height="200">

5. Select the ClearDB MySQL add-on and select the relevant pricing plan. (Note that I select Punch plan, which currently cost about $9.99 per month to increase storage capacity for this project.)

6. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
  logins = {"host": <host_name>, 
            "user": <user_name>, 
            "password": <password>, 
            "dbname": <default_Heroku_database_name>}
```
- For security reasons, this file needs to be stored in private. I've also included a video reference link below for clear instructions on setup ClearDB MySQL for Heroku.
  
[![Deploy MySQL Database into Heroku](https://i.ytimg.com/vi/Zcg71lxW-Yo/hqdefault.jpg)](https://www.youtube.com/watch?v=Zcg71lxW-Yo&ab_channel=CodeJava)

7. Inside your new app, deploy the code into the app by either linking your github repository or manually deploy it using Heroku CLI (Instructions are available and self-explanatory when selecting Heroku CLI option).
<img src="https://user-images.githubusercontent.com/34255556/160223941-2aacc3ca-4ab5-4996-be46-f2d553933dd5.png" width="600" height="300">

8. After successful model deployment, open the app and you will see the following interface designed using Streamlit:
<img src="https://user-images.githubusercontent.com/34255556/174513887-d2614ff2-a1d4-4ed5-886a-4ea44045bcc1.png" width="600" height="300">

9. From the image above, click on Data Preparation first for initializing data ingestion into MySQL, followed by Preprocessing & Model_Training for training models, followed by Data Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/174514771-8510f66c-784c-4c32-9565-d5f42ce6373d.png" width="600" height="350">
<img src="https://user-images.githubusercontent.com/34255556/174514833-1fc904b3-faae-408d-9f06-c31293413d3b.png" width="600" height="350">
  
<b>Important Note: Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.</b>

**Project Instructions (Local Environment)**
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.

3. Open MySQL in your local machine system and create a new database name of your choice with the following syntax: <b>CREATE DATABASE db_name;</b>
- Note that you will need to install MySQL if not available in your local system: https://dev.mysql.com/downloads/windows/installer/8.0.html
  
4. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <new_local_database_name>} 
```
- For security reasons, this file needs to be stored in private.
  
5. Open anaconda prompt and create a new environment with the following syntax: <b>conda create -n myenv python=3.9.10 anaconda</b>
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

6. After creating a new anaconda environment, activate the environment using the following command: <b>conda activate myenv</b>

7. Go to the local directory where your downloaded file is located and run the following command to install all the python libraries : <b>pip install -r requirements.txt</b>

8. After installing all the required Python libraries, run the following command on your project directory: <b>streamlit run main.py</b>

If you encounter the TomlDecodeError, ensure that the <b>config.toml</b> file is removed from the directory where Streamlit is installed to prevent TomlDecodeError. The following link explains more details about the error that you might encounter: https://stackoverflow.com/questions/59811357/how-to-solve-toml-decoder-tomldecodeerror-key-group-not-on-a-line-by-itself-l

9. A new browser will open after successfully running the streamlit app with the following interface:
<img src="https://user-images.githubusercontent.com/34255556/174513887-d2614ff2-a1d4-4ed5-886a-4ea44045bcc1.png" width="600" height="300">

10. From the image above, click on Data Preparation first for initializing data ingestion into MySQL, followed by Preprocessing & Model_Training for training models, followed by Data Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/174514771-8510f66c-784c-4c32-9565-d5f42ce6373d.png" width="600" height="350">
<img src="https://user-images.githubusercontent.com/34255556/174514833-1fc904b3-faae-408d-9f06-c31293413d3b.png" width="600" height="350">

**Project Instructions (Docker)**
---
A suitable alternative for deploying this project is to use Docker, which allows easy deployment on other running instances. 
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

Docker Desktop needs to be installed into your local system, before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/160223555-388eb56f-7ca5-4d40-88ca-ad80b1638599.png" width="600" height="200">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.  
  
3. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private.
  
4. Create a file named Dockerfile with the following commands:
<img src="https://user-images.githubusercontent.com/34255556/160229685-c268b253-02f2-42f3-912a-189930a997f4.png">

5. Build a new docker image on the project directory with the following command: <b>docker build -t api-name .</b>

6. Run the docker image on the project directory with the following command: <b>docker run -p 8501:8501 api-name</b>
<img src="https://user-images.githubusercontent.com/34255556/160229611-1e20ef06-dba2-4b0c-8735-2ac44fc1d38f.png" width="600" height="100">

- Note that port 8501 is required to run streamlit on Docker.

7. Open a new browser with the following link shown on the image above, which will open after successfully running the streamlit app with the following interface:
<img src="https://user-images.githubusercontent.com/34255556/174513887-d2614ff2-a1d4-4ed5-886a-4ea44045bcc1.png" width="600" height="300">

8. From the image above, click on Data Preparation first for initializing data ingestion into MySQL, followed by Preprocessing & Model_Training for training models, followed by Data Prediction for model prediction. (Note that zip files will be available for download after successfully completing either model training or model prediction, which contains the results from running the process)
<img src="https://user-images.githubusercontent.com/34255556/174514771-8510f66c-784c-4c32-9565-d5f42ce6373d.png" width="600" height="350">
<img src="https://user-images.githubusercontent.com/34255556/174514833-1fc904b3-faae-408d-9f06-c31293413d3b.png" width="600" height="350">

**Project Findings**
---

### 1. Overall confusion matrix

The following image below summarizes the confusion matrix from the final model trained for predicting status of wafers:

![image](https://user-images.githubusercontent.com/34255556/175194305-afb07f5c-6c66-433f-ad95-6869bc327998.png)

From the image above, the classification model performs better for status of wafers in bad condition (1) with less false negatives (10 samples), as compared to false positives (173 samples). Given that the model evaluation criteria emphasize the costly impact of having false negatives, the current classification model is optimized to improve recall score, instead of precision score.

---
### 2. Best classification model configuration

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Support Vector Classifier

  - <b>Best model hyperparameters</b>: {'C': 0.02364010981012719, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}

  - <b>Number of features selected</b>: 27

  - <b>List of features selected</b>: ['Sensor56', 'Sensor76', 'Sensor101', 'Sensor104', 'Sensor112', 'Sensor122', 'Sensor126', 'Sensor130', 'Sensor131', 'Sensor184', 'Sensor246', 'Sensor269', 'Sensor292', 'Sensor320', 'Sensor324', 'Sensor337', 'Sensor346', 'Sensor384', 'Sensor407', 'Sensor419', 'Sensor424', 'Sensor434', 'Sensor456', 'Sensor511', 'Sensor518', 'Sensor541', 'Sensor583', 1, 2, 3] (Note: 1,2 and 3 indicates cluster number)
  
  - <b>Clustering</b>: Yes

Note that the results above may differ for every instance of project implementation.

---
### 3. Summary of model evaluation metrics from best classification model

The following information below summarizes the evaluation metrics from the best model identified in this project: 

  - <b>Matthews Correlation Coefficient (Train-val set)</b>: 0.2576
  - <b>Matthews Correlation Coefficient (Test set)</b>: 0.2541
  
  - <b>Recall score (Train-val set)</b>: 0.7626
  - <b>Recall score (Test set)</b>: 0.7609

Note that the results above may differ for every instance of project implementation.

---
### 4. Learning Curve Analysis
  
![image](https://user-images.githubusercontent.com/34255556/175194088-598f252a-ff27-41ba-9234-6bdbedefa0a3.png)

From the diagram above, the gap between train and test recall scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
Although the gap between both scores are approximately 13%, both scores have not yet stabilize over a number of training samples. This indicates that more data can still be collected to ensure less risk of overfitting.

---
### 5. Summary from feature-engineering/data cleaning process
Note that all intermediate results from this stage are stored in Intermediate_Train_Results folder for reference.

Prior to model training, the following steps have been taken for feature engineering/data cleaning with the following outcomes:

#### i. Removal of irrelevant features
From a total of 591 features, 131 features have been removed with the following breakdown along with its justification:

| Justification                | No. of features|
| :---------------------------:|:--------------:|
| Label ID                     | 1              |
| More than 80% missing values | 4              |
| Constant variance            | 126            |

For more details of which features have been removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

#### ii. Gaussian vs Non-gaussian Variables
From the remaining 460 features, 90 features are identified to follow gaussian distribution and remaining 370 features are identified to follow non-gaussian distribution. All features are identified to follow either gaussian or non-gaussian distribution using Anderson test from Statsmodels package.

For more details of which features are gaussian or non-gaussian, refer to the following CSV files: <b>Gaussian_columns.csv, Non_gaussian_columns.csv</b>

#### iii. Proportion of Missing Values
Out of 460 features, 435 features are identified to have missing values with different proportions. All 435 features are identified to have strong to very strong correlation of missingness with other features (data missing not completely at random), thus using simple imputation methods like mean or median imputation is less suitable. Instead, <b>iterative imputation is applied across all features with missing values</b>.

For more details of proportion of missing values for all features, refer to the following CSV file: <b>Missing_Values_Info.csv</b>

Additional CSV files like <b>Imputation_Methods.csv</b> and <b>Missing_Values_Records.csv</b> have also been included for reference.

#### iv. Summary of Outliers Identified
Out of 460 features, 440 features are identified to have outliers (13 gaussian features and 427 non gaussian features) with different proportion of outliers ranging from 0.15% to 27.06%. Although these outliers have been identified using statistical methods like mean and standard deviation for gaussian variables and median and interquartile range (IQR) for non-gaussian variables, removing such outliers will require further investigation and proper justification from a business perspective.

Thus, an alternative method to handle these outliers is to capping outliers at boundary values using mean and st. deviation for gaussian variables and IQR for non-gaussian variables. Additional experiments can also be done to identify impact of outlier handling methods on model performance.

For more details of proportion of outliers for all features, refer to the following CSV files: <b>Outliers_Info_Gaussian.csv, Outliers_Info_Non_Gaussian.csv</b>

#### v. Gaussian transformation on non-gaussian variables
In Machine Learning, several machine learning models like logistic regression and gaussian naive bayes tends to perform best when data follows the assumption of normal distribution. Out of 370 non-gaussian features, 65 features can be transformed into gaussian distribution with the following breakdown:

| Gaussian transformation| No. of features|
| :---------------------:|:--------------:|
| Yeo-johnson            | 22             |
| Reciprocal             | 1              |
| Square root            | 24             |
| Logarithmic            | 18             |

Note that anderson test is used to identify whether a given gaussian transformation technique successfully converts a non-gaussian feature to a gaussian feature.

For more detailed breakdown of gaussian transformation method used for various features, refer to the following CSV file: Best_Transformation_Non_Gaussian.csv

#### vi. Feature scaling methods used
Feature scaling is only essential in some Machine Learning models like Logistic Regression, SVC, KNN and Neural networks for faster convergence and to prevent misinterpretation of one feature significantly more important than other features.
  
**Legality**
---
This is a personal project made for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
