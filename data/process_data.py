import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""
	load the raw data file. One contains the message text strings, one contains the categories of each message
	input: messages_filepath - the filepath that stored the messages file
		   categories_filepath - the filepath that stored the categories file
	output: df - a dataframe that contains both information from messages data fie and categories data file
	"""
	messages = pd.read_csv(messages_filepath)	#load message dataframe
	categories = pd.read_csv(categories_filepath)  # load categories dataframe
	df = messages.merge(categories, on='id', how='inner')  # combine two dataframe
	return df


def clean_data(df):
	"""
	a function to clean the data, split the original categroies column into individual columns for categories,remove dulication
	input: df - a dataframe that contains both information from messages data fie and categories data file
	output: df - a cleaned dataframe
	"""
	categories = df['categories'].str.split(';', expand=True)  # split the values in categories column
	row = categories.iloc[0]
	get_colnames=lambda row0:row0[0:-2]  #use the value before -# as column name
	category_colnames = row.apply(get_colnames)
	categories.columns = category_colnames
	for column in categories:
        # set each value to be the last character of the string
		categories[column] = categories[column].str[-1]    
        # convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])
	df.drop("categories",axis=1,inplace=True)  #drop the original categoires column in df
	df =pd.concat([df, categories], axis=1)    # concat the new encoded categories columns with orignal df
	df.drop_duplicates(subset=["original","message"],inplace=True)   # drop duplicated message in original or message columns
	return df


def save_data(df, database_filename):
	"""
	store the cleaned dataframe to an sql file
	input: df - the cleaned dataframe
		   database_filename - the filepath where the data will be stored
	output: none
	"""
	engine = create_engine('sqlite:///'+database_filename)
	df.to_sql("DisaterResponse", engine, index=False)
	pass  


def main():
	"""
	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	"""
	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)
        
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
        
		print('Cleaned data saved to database!')
    
	else:
		print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
	main()