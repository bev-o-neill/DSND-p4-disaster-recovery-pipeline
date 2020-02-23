# Disaster Response Pipeline Project

The aim of this project is to classify disaster response messages through machine learning techniques.

### File Descriptions
* Data 
  * process_data.py: reads in the data, cleans and stores it in a SQL database.
  * disaster_categories.csv and disaster_messages.csv (raw datasets).
  * DisasterResponse.db: created database from transformed and cleaned data.

* Models
  * train_classifier.py: this program loads the data, transforms it using natural language processing, run an AdaBoost Classifier model using GridSearchCV and trains it
  * classifier.pkl: the pickled model file

* App
  * run.py: this program runs a Flask app that allows the user to visualise the results.
  * templates: folder containing the html templates



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements

The raw input datasets were supplied by Figure Eight.
