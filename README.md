# Disaster Response Pipeline Project

# Installation
The required libraries including: pandas, sqlalchemy, re, pickle, nltk and sklearn. "punck","words","stopwords" and "wordnet"

# Project Motivation
This is the my fifth project in Udacity Data Science nanedegree.
In this project,disaster data from Figure Eight will be analyzed and used to build a model for an API that classifies disaster messages. The utlimate goal is to build a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

# File Descriptions
There are five foleders:

1. data folder
	- disaster_categories.csv: dataset including all the categories values
	- disaster_messages.csv: dataset including all the messages strings
	- process_data.py: ETL pipeline scripts to read, clean, and save data into as sql database
	- DisasterResponse.db: the output data file from the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
	- train_classifier.py: machine learning pipeline scripts to train and export a model as classifier
	- classifier.pkl: the trained classifier from the machine learning pipeline
3. app
	- run.py: Flask file to run the web application
4. templates contains html file for the web applicatin
5. notebook: contains the jupter notebook strips as a reference

# Results
An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database. 
Then a machine learning pipepline was built to train a randomforest model to performs multi-output classification.
At last, a Flask app was created to show data visualization and classify the message that user enters on the web page.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
