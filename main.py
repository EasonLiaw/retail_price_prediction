from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import streamlit as st
import pandas as pd
import pickle as pkl
from keras.models import load_model
import numpy as np
import psycopg2
import DBConnectionSetup as login
import json
from Application_Logger.logger import App_Logger
import datetime
import pandas as pd

def main():
    log_writer = App_Logger()
    host = login.logins['host']
    user = login.logins['user']
    password = login.logins['password']
    dbname = login.logins['dbname']
    tbname = 'prediction_price'
    fileobject = open('Prediction_Logs/Prediction_Log.txt', 'a+')
    st.title("Retail Price Optimizer")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Retail Price Optimizer App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Pipeline for model training process
    if st.button("Model Training"):
        with open("Training_Logs/Training_Main_Log.txt", 'a+') as file:
            trainvalidator = rawtraindatavalidation('retailprice_training', file, "Good_Train_Data/")
            trainvalidator.initial_data_preparation('schema_training.json',"Training_Batch_Files/","Good_Train_Data/",'Training_Data_FromDB/Training_Data.csv') 
        with open("Training_Logs/Training_Preprocessing_Log.txt", 'a+') as file:
            preprocessor = train_Preprocessor(file, 'Intermediate_Train_Results/')
            X_train, X_test, y_train, y_test, X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers, y_test_cap_outliers = preprocessor.data_preprocessing('Training_Data_FromDB/Training_Data.csv', 'Columns_Drop_from_Original.csv',['UID','NAME','(NSV-GST)-SALES AT COST','MARGIN%','Gross RGM(P-L)','Gross Margin %(Q/P*100)'],'Sales at Cost')
        with open("Training_Logs/Training_Model_Log.txt", 'a+') as file:
            trainer = model_trainer(file)
            trainer.train_model_and_hyperparameter_tuning(X_train, X_test, y_train, y_test, X_train_cap_outliers, X_test_cap_outliers, y_train_cap_outliers, y_test_cap_outliers,'Intermediate_Train_Results/','Model_results_by_num_features.csv','Best_Model_Results.csv', 0.10)
        st.success('Model training process is complete')

    # Process for creating database for model prediction
    if st.button("Initialize Prediction Database"):
        log_writer.log(fileobject, f"Start creating new table({tbname}) in SQL database ({dbname})")
        try:
            with open('schema_prediction.json', 'r') as f:
                schema = json.load(f)
            initialize_predictionDB(schema, host, user, password, dbname, tbname, log_writer, fileobject)
        except Exception as e:
            log_writer.log(fileobject, f"Error in creating new table({tbname}) in SQL database ({dbname} with the following error: {e})")
            raise Exception()
        log_writer.log(fileobject, f"Finish creating new table({tbname}) in SQL database ({dbname})")
        st.success('Database for price prediction has been initialized')

    # User input form for price prediction
    with st.expander("Model Prediction"):
        brand = st.selectbox('Select Brand of Product',['GOLDEN HARVEST PREMIUM','KARMIQ',"SANGI'S KITCHEN",'EKTAA','GOLDEN HARVEST PRIME',
        'GOLDEN HARVEST','COP LOOSE','DESI ATTA','MOTHER EARTH','COP UNBRANDED','NATURES CHOICE','FRESH & PURE','KOSH',
        'PREMIUM HARVEST','NILGIRIS','BUSTAAN','AGRI PURE','SHUBHRA','FARMERS PRIDE','SUNKIST','GOLDEN HARVEST DAILY','FB SIS'])

        mc = st.selectbox('Select Material Category of Product',['Urad Dal', 'Raisin', 'Chilli/Mirch Powder', 'Sabudana', 'Poha/Avalakki/Chirwa', 'Toor Dal', 'Non Veg - Curry Masa',
        'Chilli/Mirch Whole', 'Kabuli Chana Whole', 'Lokwan Wheat', 'Other Whole Spices', 'Mustard/Rai Whole', 'Watana Whole','North Indian Mixes', 'Fenugreek/Methi Whol', 
        'Millets', 'Double Boiled Rice', 'Rice Based Flour', 'Cumin/Jeera Whole','South Indian Mixes', 'Health & Other Flour', 'Salt', 'Other Whole Pulses', 
        'Multigrain Flour', 'Jaggery', 'Red Chana Whole','Masoor Dal', 'Fennel/Saunf Whole', 'Health Rice', 'Moong Chilka', 'Turmeric/Haldi Powde', 'Cashew', 
        'Sona Masoori Raw Ric','Dried Fig', 'Cinnamon/Dalchini Wh', 'Seeds & Super food', 'Rava', 'Single Boiled Rice', 'Olive & Exotic Oil', 'Urad Whole',
        'Cardamom/Elaichi Who', 'Garam Masala Powder', 'Herbs & Seasoning', 'Sunflower Oil', 'Sesame/Til Whole', 'Standard Rice','Veg - Curry/Sabji Ma', 'Other Raw Rice', 
        'Basic Sugar', 'Daliya', 'Rajma Whole', 'Mustard Oil', 'Dates', 'Basmati Broken Rice','Other Dried Fruits', 'Chawli Whole', 'Kurmura/Murmura/Lahi', 'Moong Dal', 
        'Coriander/Dhania Who', 'Soya Chunks', 'Chana Dal','Coriander/Dhania Pow', 'Popular Rice', 'Premium Rice', 'Kolam Raw Rice', 'Mixed Dal', 'Masoor Whole', 
        'Maida', 'Fragrant Raw Rice','Panch Phoron Whole', 'Pista', 'Basmati Head Grain R', 'Flavoured Pista', 'Basmati Dubar Rice', 'Tamarind/Imli Whole',
        'Asafoetida/Hing Powd', 'Moong Whole', 'Flavoured Cashew', 'Poppy Seed/Khus Khus', 'Bay Leaf/Tej Patta W', 'Wheat Flour','Flavoured Almond', 'Dry Mango/Amchur Pow', 
        'Urad Chilka', 'Dry Fruit Gift Pack', 'Dried Fenugreek/Kasu','Mixed Dry Fruits', 'Dal/Sambhar Masala', 'Besan', 'Snackmix Masala', 'Ricebran Oil', 'Desi Ghee', 
        'Saffron/Kesar Whole', 'Futana', 'Makhana Whole', 'Pepper/Kali Mirch Po', 'Sharbati Wheat', 'Carrom/Ajwain Whole', 'Upwas Flour', 'Almond', 'Basmati Tibar Rice', 
        'RICE RECIPE READY', 'Pepper/Kali Mirch Wh', 'Walnut','Other Split Pulses', 'Groundnut', 'Clove/Laung Whole', 'DAL RECIPE READY', 'Til Oil', 'Steam Rice', 
        'Economy Rice', 'Soyabean Oil', 'Cumin/Jeera Powder', 'Cow Ghee', 'Health Oil', 'Other Flavoured Nuts', 'Supreme Rice', 'Other Powdered Spice', 'Ponny Raw Rice', 
        'Dried Coconut/Kopra', 'Garam Masala Whole', 'Value Added Sugar', 'Parmal Raw Rice', 'Groundnut Oil', 'Drinks & Beverage Ma', 'Cottonseed Oil', 'Vanaspati', 'Palm Oil'])
        nsu = st.number_input('Enter Net Sales Unit',min_value=0.00)
        gst = st.number_input('Enter GST Value',min_value=0.00)
        mrp = st.number_input('Enter maximum possible retail price',min_value=0.01)
        sp = st.number_input('Enter selling price',min_value=0.00)
        countencoder = pkl.load(open('Intermediate_Train_Results/CountEncoder.pkl','rb'))
        brand_value = countencoder.transform(pd.Series([brand],name='Brand'))
        catboostencoder = pkl.load(open('Intermediate_Train_Results/CatBoostEncoder.pkl','rb'))
        mc_value = catboostencoder.transform(pd.Series([mc],name='MC'))
        zone = st.selectbox('Select zone of product',('NORTH','SOUTH','EAST','WEST'))
        zone_east_value = 1 if zone=='EAST' else 0
        zone_north_value = 1 if zone=='NORTH' else 0
        zone_south_value = 1 if zone=='SOUTH' else 0
        month=st.selectbox('Select month of transaction',(1,2,3,4,5,6,7,8,9,10,11,12))
        year=st.selectbox('Select year of transaction',(2017,2018,2019,2020,2021,2022))
        year_value = 1 if year == 2017 else 2 if year == 2018 else 3 if year == 2019 else 4 if year == 2020 else 5 if year == 2021 else 6 if year == 2022 else 0
        date = str(datetime.date(year,month,1))
        input_values = np.array([[brand_value.values[0][0], mc_value.values[0][0], nsu, nsu*sp,gst,nsu*sp-gst,nsu*mrp,mrp,sp,mrp-sp,(mrp-sp)/mrp*100,month,year_value,zone_east_value,zone_north_value,zone_south_value]])
        if st.button('predict'):
            result = pd.read_csv('Intermediate_Train_Results/Best_Model_Results.csv')
            num_features = result['num_features'].values[0]
            col_list = result['column_list'].values[0].replace("'","").strip("][").split(', ')[:num_features]
            regressor = pkl.load(open('Saved_Models/ExtraTreesRegressor_no_clustering.pkl','rb'))
            inputs = pd.DataFrame(input_values, columns = ['Brand','MC','NSU','NSV','GST_Value','Net_NSV','Gross_Sales','MRP','SP','DIS','DIS_percent','Month','Year','ZONE_EAST','ZONE_NORTH','ZONE_SOUTH'])
            inputform = np.array(inputs.loc[:,col_list])
            predicted_value = regressor.predict(inputform)
            conn = psycopg2.connect(host=host,user=user,password=password,database = dbname)
            mycursor = conn.cursor()
            brand_input = str(brand).replace("'","''")
            try:
                mycursor.execute(f"INSERT INTO {tbname} VALUES ('{datetime.datetime.now()}','{zone}', '{brand_input}', '{mc}', '{date}', {nsu}, {nsu*sp},{gst},{nsu*sp-gst},{nsu*mrp},{mrp},{sp},{mrp-sp},{(mrp-sp)/mrp*100},{np.round(predicted_value[0],2)})")
                conn.commit()
            except Exception as e:
                log_writer.log(fileobject, f'Record could not be inserted into database with the following error: {e}')
                conn.rollback()
                raise Exception()
            log_writer.log(fileobject, f"Record with predicted cost of goods sold of {np.round(predicted_value[0],2)} is inserted into SQL database")
            st.write('Predicted cost of goods sold per unit is $',str(np.round(predicted_value[0],2)))

def initialize_predictionDB(schema, host, user, password, dbname, tbname, log_writer, fileobject):
    """
        Method Name: initialize_predictionDB
        Description: This method creates a new database and table in PostgreSQL database based on a given schema object.
        Output: None
        On Failure: Logging error and raise exception
    """
    try:
        conn = psycopg2.connect(host=host,user=user,password=password,database=dbname)
        mycursor = conn.cursor()
        for name, type in zip(schema['ColName'].keys(),schema['ColName'].values()):
            mycursor.execute(f"""SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{tbname}'""")
            if mycursor.fetchone()[0] == 1:
                try:
                    mycursor.execute(f"ALTER TABLE {tbname} ADD \"{name}\" {type}")
                    log_writer.log(fileobject, f"Column {name} added into {tbname} table")
                except:
                    log_writer.log(fileobject, f"Column {name} already exists in {tbname} table")
            else:
                mycursor.execute(f"CREATE TABLE {tbname} (\"{name}\" {type})")
                log_writer.log(fileobject, f"{tbname} table created with column {name}")
            conn.commit()
    except ConnectionError:
        raise ConnectionError()
    except:
        raise Exception()
    conn.close()

if __name__=='__main__':
    main()