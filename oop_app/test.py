import pandas as pd
import numpy as np

# Neural Network as a class
from sklearn.model_selection import train_test_split

from analysis.ai.bagging_regression import BaggingReg
from analysis.ai.rf_classification import AiRFClassifier
from analysis.ai.rf_regression import RFRegression
from data_preparation.other_functions import CleanData



seed = 10
np.random.seed(seed)

df=pd.read_csv('C:/Users/ccndu/Downloads/Real estate valuation data set.csv')
dfx=df.drop(columns=['Y house price of unit area', 'No']) # inputs
dfy=df['Y house price of unit area']
print(df)
x_train, x_test, y_train, y_test = train_test_split(dfx,dfy, train_size=0.1)
aiModel= BaggingReg(x_train, x_test, y_train, y_test)
# Set Classifer of RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
# Define Settings for Regressor

#Criteria for the Bagging Regressor Model:
no_of_estimators = 10 # integer value: number of trees in the forest
max_features = 1 # The number of features to draw from X to train each base estimator
max_samples = 1 # The number of samples to draw from X to train each base estimator
random_state = 10 # To randomly resample the original dataset
oob_score = True # Use out-of-bag samples to estimate the generalization error.

regressionmodel = BaggingReg(x_train, x_test, y_train, y_test) # creation of the object
regressionmodel.build_model(no_of_estimators, max_features, max_samples, random_state, oob_score)

score = regressionmodel.get_test_score()
print(score)