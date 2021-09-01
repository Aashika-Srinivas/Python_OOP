# Import libraries
import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix

from analysis.analysis_interface import AnalysisInterface
from response import Response


class NeuralClassifier(AnalysisInterface):

    # Constructor method
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Additional property to be used in some methods
        self.estimator = None

    # Destructor
    def __del__(self):
        print('The Keras Classifier Neural Network has been deleted')

    # Create Model
    def build_model(self, opt, fun, init, epo, batch, nn):
        self.opt = opt
        self.fun = fun
        self.init = init
        self.epo = epo
        self.batch = batch
        self.nn = nn

        # Try - Except statement the building of the model fails
        try:
            # when X_train (input) is 1 a direct assingment is needed, otherwise there is a problem here inputs= X_train.shape[1]
            if len(self.X_train.shape) == 1:
                inputs = 1
            else:
                inputs = self.X_train.shape[1]

            # Number of categories (outputs)
            outputs = len(Counter(self.y_test).keys())

            # Define the keras base model
            def baseline_model():
                kerasmodel = Sequential()
                kerasmodel.add(Dense(self.nn[0], input_dim=inputs, kernel_initializer=self.init, activation='relu'))
                kerasmodel.add(Dense(self.nn[1], kernel_initializer=self.init, activation='relu'))
                kerasmodel.add(Dense(outputs, kernel_initializer=self.init, activation=self.fun))

                # Compile model
                kerasmodel.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
                return kerasmodel

            # Build and fit the model
            self.estimator = KerasClassifier(build_fn=baseline_model, epochs=self.epo, batch_size=self.batch, verbose=0)
            self.estimator.fit(self.X_train, self.y_train)

        except:
            return Response.failure('Error building model')

        return Response.success(self.estimator)
    
    # Get accuracy score using train data
    def get_train_score(self):
        score = self.estimator.score(self.X_train, self.y_train)
        return score

    # Get accuracy score using test data
    def get_test_score(self):
        score = self.estimator.score(self.X_test, self.y_test)
        return score

    # Predict values from X_test
    def predict_y_from_x_test(self):
        y_predict = self.estimator.predict(self.X_test)
        return y_predict

    # Predict values from new X_data
    def predict_y(self, X_Data):
        y_predict = self.estimator.predict(X_Data)
        return y_predict
    
    # Plot heatmap -> confusion matrix
    def get_plot(self):
        # Confusion Matrix Labels
        y_predict = self.predict_y_from_x_test()
        y = np.array(y_predict)
        label = np.unique(y_predict)
        cmatrix = confusion_matrix(self.y_test.values, y_predict)
        plt.figure(figsize=(10, 10))
        fig = sn.heatmap(cmatrix, annot=True, fmt="d",linewidths=0.5, xticklabels=label, yticklabels=label)
        plt.xlabel('Predicted')
        plt.ylabel('Original')
        plt.title('Neural Network Confusion Matrix')
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        return figure_name
