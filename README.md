# Automated-Content-Detector-ACM-
##Introduction####

This readme file explains the major components of the project.

##Python Notebooks

There are two main jupior note book files for binary and multi-label classifiction models. This jupitor notebook contains the core implemnetation of classification models. 

Binary Classifier model: F21MP_AbusiveDetection_BinaryClassifierFinal.ipynb
Multi label Classifier Model: F21MP_MultiLabelClassification_Final.ipynb

* Base Model with TF-IDF for comparison : Base_TF-IDF_Model.ipynb

Explonatory data analysis performed on different notebooks file saved uner "Notebook versions" folder.


##Data Files

--Binary classification data set: 
  Train data set:OffensiveBinaryDataset.csv
  Test dataset:test.csv

-- Multi label classification
   Train data: train.csv
   Test data: multiLabelTest.csv

-- Native Slang dictionary for text preprocessing
   slangDict.csv


## Pre-trained embedding vector files
   GoogleNews-vectors-negative300.bin :- word2vec file.
   glove.6B.200d.txt :- Glove file


## Application file for model deployment
  Name of the python file: app.py 
  For starting the server, we need to run the app.py file on the working directory from python/ anaconda environment. Once the application starts an ip   address will be displayed on the command prompt. This ip address (eg: http://127.0.0.1:5000/) can be used in the browser to use the web interface of the application.

## Saved model artifacts:
  In order to run the application we need the following model artifacts.
  ---Containerized files
  SavedBinaryAbussiveDetectionModel_V.01 :- This folder contains the saved conatiner file for Binary classification model which is loaded in the app.py   file.
  SavedMultiLabelAbussiveDetectionModel_V.01 :- This file is a saved version of multi label model file which is hosted in the app.py.
  
  --- Pickeled Tokenizer class.
  tokenizer.pickle :- for binary tokenizer
  tokenizer_multiLabel.pickle :- multi label tokenizer

## Configuration file
   config.yml :- this file is to configure the threshold value of probabalities for the multi label classification.

## Web development files for Application UI
   All the HTML, Java script, CSS and D3.js files are stored in the 'templates' folder in the paraent directory.

## Twitter scrapped data file and its prediction
   scraped_tweets: this file conatins the tweets extracted through user interface for the given hashtag for a specified period of time.
   tiwtter_scrapping_binary_prediction :- This file contains the tweets extracted and its predicted labels.

  
  

