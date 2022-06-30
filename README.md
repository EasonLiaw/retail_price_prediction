# Retail Price Prediction Project

**Background**
---

<img src="https://datacrops.com/wp-content/uploads/2017/08/How-Retail-Pricing-Intelligence-can-assist-with-Price-Optimization-792350.jpg">

One of the current challenges faced by most retailers is to optimize prices with a balancing act between maximizing profits and maximizing sales. While traditional rule-based methods have been used by retailers to manage price optimization, these methods require manual analysis of customer and market data to ensure prices are aligned with current market conditions. Inevitably, the overall expansion of digitization of retail industry in recent years due to the global pandemic has resulted in a massive increase of sales-related data, such that traditional rule-based methods result in difficulty of continuous monitoring. 

For this project, the main goal is to deploy initial regression models that help to predict optimum retail price for various products, while also improve brand and sales of various retail products in the long term. With the help of Machine Learning, retailers will be able to utilize the full potential of their data by effectively setting prices that maximizes their profits without discouraging customers from purchasing their products.

Dataset is provided in .csv format by client under <b>Training_Batch_Files</b> folder for model training. In addition, schema of datasets for training and prediction is provided in .json format by the client for storing seperate csv files into a single PostgreSQL database.

For model prediction, a web API is used (created using StreamLit) for user input and the results generated from model prediction along with user inputs are stored directly into a table in PostgreSQL database.

**Code and Resources Used**
---
- **Python Version** : 3.9.7
- **Packages** : feature-engine, kneed, matplotlib, psycopg2, numpy, optuna, pandas, scikit-learn, scipy, seaborn, streamlit, tensorflow, keras, category_encoders
- **Dataset source** : Dummy data from 360DIGITMG
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

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> and <b>Prediction_Logs</b> folders.

**Project Folder Structure**
---
The following points below summarizes the use of every file/folder available for this project:
1. Application_Logger: Helper module for logging model training and prediction process
2. Good_Train_Data: Stores csv file after initial data preprocessing for data ingestion into PostgreSQL database
3. Intermediate_Train_Results: Stores additional information from model training process
4. Model_Training_Modules: Helper modules for model training
5. Saved_Models: Stores best models identified from model training process for model prediction
6. Training_Batch_Files: Stores csv files to be used for model training
7. Training_Data_FromDB: Stores compiled data from PostgreSQL database for model training
8. Training_Logs: Stores logging information from model training for future debugging and maintenance
9. Dockerfile: Additional file for Docker model deployment
10. main.py: Main file for program execution
11. README.md: Details summary of project for presentation
12. requirements.txt: List of Python packages to install for model deployment
13. schema_prediction.json: JSON file that contains database schema for model prediction
14. schema_training.json: JSON file that contains database schema for model training

The following sections below explains the three main approaches that can be used for model deployment in this project:
1. <b>Google Cloud Platform (GCP)</b>
2. <b>Local system</b>
3. <b>Docker</b>

**Project Instructions (GCP)**
---
<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- Procfile
- requirements.txt
- setup.sh

<b>Note that deploying models onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.
<img src="https://user-images.githubusercontent.com/34255556/176189931-0fb31aab-d5ab-4326-bd62-7c5327f709fa.png" width="600" height="300">

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

![image](https://user-images.githubusercontent.com/34255556/176185292-cf3cb500-1055-4c49-95fb-71c4df1249db.png)

9. From the image above, click on Model Training for initializing data ingestion into PostgreSQL, further data cleaning and training models, followed by Initialize Prediction Database for initialize a new table for model prediction in PostgreSQL database.

![image](https://user-images.githubusercontent.com/34255556/176185669-4a418c16-4806-4f06-95c6-abf5643b5012.png)
![image](https://user-images.githubusercontent.com/34255556/176185692-5b352acb-2e88-4b34-87f3-6f5cc5c08822.png)

10. For model prediction, users can input the following information onto the interface for retail price prediction:
- Brand: List of brand values
- MC: List of category of material values
- NSU (Net Sales Unit): Floating value from 0
- GST: Floating value from 0
- MRP (Maximum Retail Price): Floating value from 0
- SP (Selling Price): Floating value from 0
- Zone: [NORTH, SOUTH, EAST, WEST]
- Month: 1 to 12
- Year: 2017 onwards

Screenshot below shows an example of the web API with the output generated from model prediction:
<img src="https://user-images.githubusercontent.com/34255556/176658808-50927709-f568-4cdf-b061-1450082c7ea0.png" width="650" height="500">

**Project Instructions (Local Environment)**
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/176189931-0fb31aab-d5ab-4326-bd62-7c5327f709fa.png" width="600" height="300">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.

3. Open pgAdmin 4 in your local machine system and create a new database name of your choice with the following syntax: <b>CREATE DATABASE db_name;</b>
- Note that you will need to install PostgreSQL first if not available in your local system: https://www.postgresql.org/download/
- After installing PostgreSQL, download pgAdmin4 for a better GUI when interacting with PostgreSQL database: https://www.pgadmin.org/download/
  
4. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <new_local_database_name>} 
```
- For security reasons, this file needs to be stored in private.
  
5. Open anaconda prompt and create a new environment with the following syntax: <b>conda create -n myenv python=3.9.7 anaconda</b>
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

6. After creating a new anaconda environment, activate the environment using the following command: <b>conda activate myenv</b>

7. Go to the local directory where your downloaded file is located and run the following command to install all the python libraries : <b>pip install -r requirements.txt</b>

8. After installing all the required Python libraries, run the following command on your project directory: <b>streamlit run main.py</b>

If you encounter the TomlDecodeError, ensure that the <b>config.toml</b> file is removed from the directory where Streamlit is installed to prevent TomlDecodeError. The following link explains more details about the error that you might encounter: https://stackoverflow.com/questions/59811357/how-to-solve-toml-decoder-tomldecodeerror-key-group-not-on-a-line-by-itself-l

9. A new browser will open after successfully running the streamlit app with the following interface:

![image](https://user-images.githubusercontent.com/34255556/176185292-cf3cb500-1055-4c49-95fb-71c4df1249db.png)

10. From the image above, click on Model Training for initializing data ingestion into PostgreSQL, further data cleaning and training models, followed by Initialize Prediction Database for initialize a new table for model prediction in PostgreSQL database.

![image](https://user-images.githubusercontent.com/34255556/176185669-4a418c16-4806-4f06-95c6-abf5643b5012.png)
![image](https://user-images.githubusercontent.com/34255556/176185692-5b352acb-2e88-4b34-87f3-6f5cc5c08822.png)

11. For model prediction, users can input the following information onto the interface for retail price prediction:
- Brand: List of brand values
- MC: List of category of material values
- NSU (Net Sales Unit): Floating value from 0
- GST: Floating value from 0
- MRP (Maximum Retail Price): Floating value from 0
- SP (Selling Price): Floating value from 0
- Zone: [NORTH, SOUTH, EAST, WEST]
- Month: 1 to 12
- Year: 2017 onwards

Screenshot below shows an example of the web API with the output generated from model prediction:
<img src="https://user-images.githubusercontent.com/34255556/176658808-50927709-f568-4cdf-b061-1450082c7ea0.png" width="650" height="500">

**Project Instructions (Docker)**
---
A suitable alternative for deploying this project is to use Docker, which allows easy deployment on other running instances. 
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

Docker Desktop needs to be installed into your local system, before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/176189931-0fb31aab-d5ab-4326-bd62-7c5327f709fa.png" width="600" height="300">

2. Empty the files stored inside Training_Logs and Prediction_Logs folder.

3. Open pgAdmin 4 in your local machine system and create a new database name of your choice with the following syntax: <b>CREATE DATABASE db_name;</b>
- Note that you will need to install PostgreSQL first if not available in your local system: https://www.postgresql.org/download/
- After installing PostgreSQL, download pgAdmin4 for a better GUI when interacting with PostgreSQL database: https://www.pgadmin.org/download/
  
4. Add an additional Python file named as DBConnectionSetup.py that contains the following Python code structure: 
```
logins = {"host": <host_name>, 
          "user": <user_name>, 
          "password": <password>, 
          "dbname": <default_database_name>} 
```
- For security reasons, this file needs to be stored in private. <b>Note that setting host name to "localhost" will result in connection error when connecting Docker to external PostgreSQL database. Instead, local ip adddress needs to be specified as the host name.</b>
  
5. Create a file named Dockerfile with the following commands:
<img src="https://user-images.githubusercontent.com/34255556/160229685-c268b253-02f2-42f3-912a-189930a997f4.png">

6. Build a new docker image on the project directory with the following command: <b>docker build -t api-name .</b>

7. Run the docker image on the project directory with the following command: <b>docker run -p 8501:8501 api-name</b>
<img src="https://user-images.githubusercontent.com/34255556/160229611-1e20ef06-dba2-4b0c-8735-2ac44fc1d38f.png" width="600" height="100">

- Note that port 8501 is required to run streamlit on Docker.

8. A new browser will open after successfully running the streamlit app with the following interface:

![image](https://user-images.githubusercontent.com/34255556/176185292-cf3cb500-1055-4c49-95fb-71c4df1249db.png)

9. From the image above, click on Model Training for initializing data ingestion into PostgreSQL, further data cleaning and training models, followed by Initialize Prediction Database for initialize a new table for model prediction in PostgreSQL database.

![image](https://user-images.githubusercontent.com/34255556/176185669-4a418c16-4806-4f06-95c6-abf5643b5012.png)
![image](https://user-images.githubusercontent.com/34255556/176185692-5b352acb-2e88-4b34-87f3-6f5cc5c08822.png)

10. For model prediction, users can input the following information onto the interface for retail price prediction:
- Brand: List of brand values
- MC: List of category of material values
- NSU (Net Sales Unit): Floating value from 0
- GST: Floating value from 0
- MRP (Maximum Retail Price): Floating value from 0
- SP (Selling Price): Floating value from 0
- Zone: [NORTH, SOUTH, EAST, WEST]
- Month: 1 to 12
- Year: 2017 onwards

Screenshot below shows an example of the web API with the output generated from model prediction:
<img src="https://user-images.githubusercontent.com/34255556/176658808-50927709-f568-4cdf-b061-1450082c7ea0.png" width="650" height="500">

**Project Findings**
---

### 1. Best regression model configuration

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Extra Trees Regressor

  - <b>Best model hyperparameters</b>: {'bootstrap': False, 'ccp_alpha': 6.3279240080591865, 'criterion': 'squared_error', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

  - <b>Number of features selected</b>: 4

  - <b>List of features selected</b>: ['MC', 'MRP', 'SP', 'DIS']
  
  - <b>Clustering</b>: No

Note that the results above may differ for every instance of project implementation.

---
### 2. Summary of model evaluation metrics from best regression model

The following information below summarizes the evaluation metrics from the best model identified in this project: 

  - <b>RMSE (Train-val set)</b>: 44.7699
  - <b>RMSE (Test set)</b>: 45.8646
  
  - <b>MAE (Train-val set)</b>: 16.7682
  - <b>MAE (Test set)</b>: 17.7559

Note that the results above may differ for every instance of project implementation.

---
### 3. Learning Curve Analysis

<img src="https://user-images.githubusercontent.com/34255556/176183310-c0d47587-6716-4f56-8f97-9c82a8c0fc21.png" width="500" height="350">

From the diagram above, the gap between train and test RMSE scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
The gap between both scores are approximately 9.95%, which doesn't provide indication of model overfitting or underfitting. In addition, both train and test RMSE scores have stabilize over larger number of samples (i.e. more than 30000), which indicates that collecting more samples may not further improve model performance.

---
### 4. Leverage Plot Analysis

<img src="https://user-images.githubusercontent.com/34255556/176183900-04e252c8-a9a6-4668-9fe2-73ac4c742cbc.png" width="500" height="350">

From the diagram above, there are many records that are identified as having high leverage or large residuals. Using Cook’s distance (value is more than 3 times the average of all values), this confirms that there are 279 records as being highly influential, which requires further investigation on the data for better understanding whether these data anomalies should be treated.

---
### 5. Feature Importance Analysis

<img src="https://user-images.githubusercontent.com/34255556/176184221-c95734ee-3676-41c2-a207-5a7b42ea3541.png" width="500" height="350">

From the diagram above, it is very clear that variables like MRP and SP are the two most important features that determines the cost per unit of a given retail item, given both features have a very high feature importance score (>0.4). Meanwhile, other features like MC and DIS provide much lower importance score (<0.05).

---
### 6. Summary from feature-engineering/data cleaning process
Note that all intermediate results from this stage are stored in Intermediate_Train_Results folder for reference.

Prior to model training, the following steps have been taken for feature engineering/data cleaning with the following outcomes:

#### i. Data Anomalies
From the original dataset given, the following data anomalies have been identified that needs to be addressed:
- <b>23 records of negative cost of goods sold (Sales at Cost variable)</b> identified. Such records are excluded from the model training process. Further investigation will be required to validate those records before including those as part of the analysis.

- <b>Negative values for MRP, SP, NSU and GST Value features</b> have been identified. Constraints have been set for those variables to be strictly greater than zero. Other variables that depend on those features will need to be recalculated before model training.

- <b>195 records identified to have selling price (SP) greater than maximum retail price (MRP)</b>. Such records are excluded from the model training process, since it is illegal for retailers in India to set retail prices beyond the maximum possible retail price (MRP).

#### ii. Removal of irrelevant features
From a total of 19 features, 6 features have been removed with the following breakdown along with its justification:

| Justification                | No. of features|
| :---------------------------:|:--------------:|
| Label ID                     | 1              |
| Irrelevant columns           | 5              |

For more details of which features have been removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

#### iii. Gaussian vs Non-gaussian Variables
From the remaining 9 continuous features, all continuous features are identified to follow non-gaussian distribution. All features are identified to follow either gaussian or non-gaussian distribution using Anderson test from Statsmodels package.

For more details of which features are gaussian or non-gaussian, refer to the following CSV files: <b>Gaussian_columns.csv, Non_gaussian_columns.csv</b>

#### iv. Summary of Outliers Identified
All 9 continuous features are identified to have outliers with different proportion of outliers ranging from 11.8% to 14.1%. Although these outliers have been identified using statistical methods like median and interquartile range (IQR) for non-gaussian variables, removing such outliers will require further investigation and proper justification from a business perspective.

Thus, an alternative method to handle these outliers is to capping outliers at boundary values using IQR for non-gaussian variables. However, simply capping outliers at boundary values for all non-gaussian features is not possible for this dataset, since there are other features that internally depends on other features. Upon closer inspection, these following features can be used for capping outliers without resulting in too many features having large proportion of outliers:  <b>['GST Value','SP','MRP']</b>

After capping outliers of those features, remainning features will require recalculation to ensure formula consistency between all features.

For more details of proportion of outliers for all features, refer to the following CSV file: <b>Outliers_Info_Non_Gaussian.csv</b>

#### v. Categorical data encoding
This dataset has the following categorical features that requires data encoding using the following methods:

| Feature Name| Encoding Method   | Justification of method                   |
| :----------:|:-----------------:|:-----------------------------------------:|
| Brand       | Count Encoding    | High cardinality with no duplicated counts|
| MC          | CatBoost Encoding | High cardinality with duplicated counts   |
| ZONE        | One Hot Encoding  | Low cardinality                           |

In addition, Fdate variable has been split into Month and Year column.

#### vi. Feature scaling methods used
Feature scaling is only essential in some Machine Learning models like Huber Regression, Lasso and Neural networks for faster convergence and to prevent misinterpretation of one feature significantly more important than other features. In this project, Robust Scaler is used for feature scaling, since this method is more robust to outliers (outliers are only capped in this project).
  
**Legality**
---
This is a personal project made for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
