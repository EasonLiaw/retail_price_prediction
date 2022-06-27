import os, json
import pandas as pd
import psycopg2
import csv
from Application_Logger.logger import App_Logger
import DBConnectionSetup as login
import shutil

class DBOperations:
    def __init__(self, tablename, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of DBOperations class
            Output: None
        '''
        self.tablename = tablename
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.host = login.logins['host']
        self.user = login.logins['user']
        self.password = login.logins['password']
        self.dbname = login.logins['dbname']

    def newDB(self, schema):
        '''
            Method Name: newDB
            Description: This method creates a new database and table in PostgreSQL database based on a given schema object.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, f"Start creating new table({self.tablename}) in SQL database ({self.dbname})")
        self.schema = schema
        try:
            conn = psycopg2.connect(host=self.host,user=self.user,password=self.password,database=self.dbname)
            mycursor = conn.cursor()
            for name, type in zip(self.schema['ColName'].keys(),self.schema['ColName'].values()):
                mycursor.execute(f"""SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.tablename}'""")
                if mycursor.fetchone()[0] == 1:
                    try:
                        mycursor.execute(f"ALTER TABLE {self.tablename} ADD \"{name}\" {type}")
                        self.log_writer.log(self.file_object, f"Column {name} added into {self.tablename} table")
                    except:
                        self.log_writer.log(self.file_object, f"Column {name} already exists in {self.tablename} table")
                else:
                    mycursor.execute(f"CREATE TABLE {self.tablename} (\"{name}\" {type})")
                    self.log_writer.log(self.file_object, f"{self.tablename} table created with column {name}")
                conn.commit()
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception(f"The following error occured when connecting to SQL database: {e}")
        conn.close()
        self.log_writer.log(self.file_object, f"Finish creating new table({self.tablename}) in SQL database ({self.dbname})")
    
    def data_insert(self, gooddir):
        '''
            Method Name: data_insert
            Description: This method inserts data from existing csv file into PostgreSQL database
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start inserting new good training data into SQL database")
        self.gooddir = gooddir
        try:
            conn = psycopg2.connect(host=self.host,user=self.user,password=self.password,database = self.dbname)
            mycursor = conn.cursor()
            with open(self.gooddir+'price.csv', "r") as f:
                next(f)
                filename = csv.reader(f)
                for line in enumerate(filename):
                    try:
                        mycursor.execute(f"INSERT INTO {self.tablename} VALUES ({','.join(line[1])})")
                        conn.commit()
                    except Exception as e:
                        self.log_writer.log(self.file_object, f'Row {line[0]} could not be inserted into database for price.csv file')
                        conn.rollback()
                self.log_writer.log(self.file_object, f"Price.csv file added into database")
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception(f"The following error occured when connecting to SQL database: {e}")
        conn.close()
        self.log_writer.log(self.file_object, "Finish inserting new good training data into SQL database")

    def compile_data_from_DB(self,compiledir):
        '''
            Method Name: compile_data_from_DB
            Description: This method compiles data from PostgreSQL table into csv file for further data preprocessing.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start writing compiled good training data into a new CSV file")
        self.compiledir = compiledir
        try:
            conn = psycopg2.connect(host=self.host,user=self.user,password=self.password,database = self.dbname)
            data = pd.read_sql(f'''SELECT DISTINCT * FROM {self.tablename};''', conn)
            data.to_csv(self.compiledir, index=False)
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to SQL database")
            raise Exception("Error connecting to SQL database")
        except Exception as e:
            self.log_writer.log(self.file_object, f"The following error occured when connecting to SQL database: {e}")
            raise Exception(f"The following error occured when connecting to SQL database: {e}")
        conn.close()
        self.log_writer.log(self.file_object, "Finish writing compiled good training data into a new CSV file")

class rawtraindatavalidation(DBOperations):
    def __init__(self, tablename, file_object, gooddir):
        '''
            Method Name: __init__
            Description: This method initializes instance of rawtraindatavalidation class, while inheriting methods from DBOperations class
            Output: None
        '''
        super().__init__(tablename, file_object)
        self.gooddir = gooddir
        self.log_writer = App_Logger()

    def load_train_schema(self, filename):
        '''
            Method Name: load_train_schema
            Description: This method loads the schema of the training data from a given JSON file for creating tables in PostgreSQL database.
            Output: JSON object
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start loading train schema")
        self.filename = filename
        try:
            with open(filename, 'r') as f:
                schema = json.load(f)
        except Exception as e:
            self.log_writer.log(self.file_object, f"Training schema fail to load with the following error: {e}")
            raise Exception(f"Training schema fail to load with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish loading train schema")
        return schema
    
    def copy_data_from_csv(self):
        '''
            Method Name: copy_data_from_csv
            Description: This method copies data from CSV file to a different folder for initial data preprocessing before inserting data into PostgreSQL.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start copying data from CSV file")
        try:
            shutil.copyfile(self.batchfilepath+"/price.csv", self.goodfilepath+"/price.csv")
        except Exception as e:
            self.log_writer.log(self.file_object, f"Data from CSV file could not be copied into database with the following error: {e}")
            raise Exception(f"Data from CSV file could not be copied into database with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish copying data from CSV file")
    
    def blank_with_null_replacement(self):
        '''
            Method Name: blank_with_null_replacement
            Description: This method replaces blanks with null values in a given CSV file.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start replacing missing values with null keyword")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(self.gooddir+file)
                filename = filename.fillna('null')
                filename.to_csv(self.gooddir+file, index=False)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Replacing missing values with null keyword for file {file} fail with the following error: {e}")
                raise Exception(f"Replacing missing values with null keyword for file {file} fail with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish replacing missing values with null keyword")

    def characters_single_quotes(self):
        '''
            Method Name: characters_single_quotes
            Description: This method adds single quotes to all string related data types in a given CSV file. Not adding single quotes to string data types will result in error when inserting data into PostgreSQL table.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(self.file_object, "Start handling single quotes on characters")
        for file in os.listdir(self.gooddir[:-1]):
            try:
                filename = pd.read_csv(self.gooddir+file)
                char_df = filename.select_dtypes('object')
                for column in char_df.columns:
                    filename[column] = filename[column].map(lambda x: x.replace("'","''"))
                    filename[column] = filename[column].map(lambda x: f'\'{x}\'')
                filename.to_csv(self.gooddir+file, index=False)
            except Exception as e:
                self.log_writer.log(self.file_object, f"Handling single quotes on characters for file {file} fail with the following error: {e}")
                raise Exception(f"Handling single quotes on characters for file {file} fail with the following error: {e}")
        self.log_writer.log(self.file_object, "Finish handling single quotes on characters")
    
    def initial_data_preparation(self, schemapath, batchfilepath, goodfilepath, finalfilepath):
        '''
            Method Name: initial_data_preparation
            Description: This method performs all the preparation tasks for the data to be ingested into PostgreSQL database.
            Output: None
        '''
        self.log_writer.log(self.file_object, "Start initial data preparation")
        self.schemapath = schemapath
        self.batchfilepath = batchfilepath
        self.goodfilepath = goodfilepath
        self.finalfilepath = finalfilepath
        schema = self.load_train_schema(self.schemapath)
        self.copy_data_from_csv()
        self.characters_single_quotes()
        self.blank_with_null_replacement()
        self.newDB(schema)
        self.data_insert(self.goodfilepath)
        self.compile_data_from_DB(self.finalfilepath)
        self.log_writer.log(self.file_object, "Finish initial data preparation")
