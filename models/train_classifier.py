import sys
import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
import pickle

def load_data(database_filepath):
    """
    Loads the preprocessed data from the SQL database
    
    Args:
    database_filepath (str): SQL file name
    
    Returns:
    X (numpy array): the disaster messages (features)
    Y (numpy array): the disaster categories (targets)
    category_names (list): the disaster category names
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)
    
    
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = (df.columns[4:]).tolist()
    return X,Y,category_names
  


def tokenize(text):
    """
    Tokenises the text data
    
    Args:
    text (str): Disaster response messages as text
    
    Returns:
    tokens (list): Processed text after normalising, tokenising and lammatising
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Normalise text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenise text
    words=word_tokenize(text)
    
    # Lemmatise and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return tokens


def build_model():
    """
    Builds an AdaBoost classifier model via GridSearchCV
    
    Returns:
    Trained model
    """
    #Build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    #Hyperparams to be tuned
    parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators' : [50,100,150],
    }
    
    #Create model
    model = GridSearchCV(pipeline, param_grid=parameters,verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints a report of the model's performance on the test data
    
    Args:
    model: model trained in build_model() function
    X_test (numpy array): test features
    Y_test (numpy array): test target
    category_names (list): disaster category names
    
    Returns: 
    Printed report of model performance
    """
    Y_pred = model.predict(X_test)

    #print(classification_report(Y_test.values, Y_pred, target_names=category_names))
    print("----Classification Report per Category:\n")
    
    for i in range(len(category_names)):
        #actual=Y_test[:, i]
        #predicted=Y_pred[:, i]
    
        print("Label:", category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves model to a pickle file
    
    Args:
    model: the model trained in the build_model() function
    model_filepath (str): location of the model when it's saved
    
    Returns:
    None
    """
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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