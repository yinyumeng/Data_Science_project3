# load packages
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
englishwords=set(nltk.corpus.words.words())
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
import pickle


def load_data(database_filepath):
	"""
	load the sql data and extract the input value and target values
	
	input: database_filepath - the filepath that contains the cleaned sql datafile
	output: X - the training data set, input values
			Y - the training data set, target values
			category_names - the name of each target, this will be used to display the evluation result of each target
	"""
	engine = create_engine('sqlite:///' + database_filepath) #load sql file
	df=pd.read_sql_table('DisaterResponse', engine) #extract sql file
	X = df["message"] #use the message column as input data
	Y = df.iloc[:,4:] #use the columns after the forth one as target data
	category_names = list(Y.columns)  #assign the category names
	return X,Y,category_names


def tokenize(text):
	"""
	tonkenize a text string, convert it to lower case, remove punctutation and root individual words
	
	input: text - a text string
	output: words - tokenized text string. a list that contains several individual words strings
	"""
	text = text.lower()  # convert the text to lower case
	text = re.sub(r"[^a-zA-Z0-9]", " ", text)  #remove punctutation
	words = word_tokenize(text)  #tokenize text
	words=[x for x in words if x in englishwords and x not in stopwords.words("english")] #remove stop words and non-english words(just in case)
	wordnet_lemmatizer=WordNetLemmatizer()
	words = [wordnet_lemmatizer.lemmatize(w, pos='v') for w in words]  # root verb words
	words = [wordnet_lemmatizer.lemmatize(w, pos='n') for w in words]  # root noun words
	words = [wordnet_lemmatizer.lemmatize(w, pos='a') for w in words]  # root adj words
	words = [wordnet_lemmatizer.lemmatize(w, pos='v') for w in words]  # root adv words
	return words
    


def build_model():
	"""
	create a ML pipeine that contains countvectorizer, tfidftransformer and randomforestclassifier
	input: None
	Output: ML pipeline
	"""
	pipeline = Pipeline([
						('vect', CountVectorizer(tokenizer=tokenize)),  # create a countvectorizer using the tokenize function
						('tfidf', TfidfTransformer()), # create a tfidf transformer
						('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200,min_samples_split=6,criterion='gini'))), # create a randomforesetclassifier
						])
    
	return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
	"""
	evaluate the modeling results and display the accuracy score of each target category
	input: model - the model for evaluation
		   X_test - input values of testing data
		   Y_test - target values of testing data
		   category_names - a list that contains the names of each target category
	output: none
	"""
	y_pred=model.predict(X_test)
	col_names=Y_test.columns
	for i in range(len(col_names)):
		precision,recall,fscore,support=score(Y_test.iloc[:, i].values,y_pred[:, i],average='weighted')
		#print("Category:", col_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
		print("Category:", col_names[i],"\n")
		print('Precision of %s: %.2f \n' %(col_names[i], precision))
		print('Recall of %s: %.2f \n' %(col_names[i], recall))
		print('F1score of %s: %.2f \n' %(col_names[i], fscore))
		print('Accuracy of %s: %.2f \n\n' %(col_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))
	pass


def save_model(model, model_filepath):
	"""
	save the trained model using pickle
	input: model - the trained model
		   model_filepath - the filepath to store the model
	"""
	pickle.dump(model, open(model_filepath, "wb"))
	pass


def main():
	"""
	python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
	"""
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  #splite the training and testing data
        
		print('Building model...')
		model = build_model()
        
		print('Training model...')
		model.fit(X_train, Y_train)
        
		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test, category_names)

		print('Saving model...\n    MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

		print('Trained model saved!')

	else:
		print('Please provide the filepath of the disaster messages database '\
				'as the first argument and the filepath of the pickle file to '\
				'save the model to as the second argument. \n\nExample: python '\
				'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
	main()