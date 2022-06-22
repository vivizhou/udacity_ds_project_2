import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql("disaster_messages", engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")
    
    #Normalization 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # Anything that isn't A through Z or 0 through 9 will be replaced by a space

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        #clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    return(clean_tokens)


def build_model():
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__min_samples_split': [2, 3, 4],
    'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3, n_jobs=4, verbose = 2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(36):
        print('Feature',i, ':', category_names[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        print("_______________________________________________________")
        accuracy = (Y_test.values == Y_pred).mean()
    print('Accuracy: {:.3f}'.format(accuracy))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
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