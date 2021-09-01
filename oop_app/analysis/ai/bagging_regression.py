# Importing necessary libraries

import random
import time

import matplotlib.pyplot as plt
import pydot
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Definition of Random Forest Regression Class with functions
from sklearn.tree import export_graphviz

from analysis.analysis_interface import AnalysisInterface
from response import Response


class BaggingReg(AnalysisInterface):

    # Dunder method init
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Dunder method del
    def __del__(self):
        print('The Scikit Learn Bagging Regressor has been deleted')

    # Initialize and Train Regression Model
    def build_model(self, no_of_estimators, max_features, max_samples, random_state, oob_score):

        try:
            regmodel = BaggingRegressor(n_estimators=no_of_estimators, max_features=max_features, max_samples=max_samples,
                                        random_state=random_state, oob_score=oob_score)
            regmodel.fit(self.X_train, self.y_train)

            self.regmodel = regmodel

        except:
            return Response.failure('Error building model')

        return Response.success(regmodel)


    def get_train_score(self):
        score = self.regmodel.score(self.X_train, self.y_train)
        return score

    def get_test_score(self):
        score = self.regmodel.score(self.X_test, self.y_test)
        return score

    def get_score_rmse_test(self):
        score_rmse_test = mean_squared_error(self.y_test, self.predict_y_from_x_test(), squared=False)
        return score_rmse_test

    def get_score_mse_test(self):
        score_mse_test = mean_squared_error(self.y_test, self.predict_y_from_x_test())
        return score_mse_test

    def get_score_mae_test(self):
        score_mae_test = mean_absolute_error(self.y_test, self.predict_y_from_x_test())
        return score_mae_test

    def predict_y_from_x_test(self):
        y_predict = self.regmodel.predict(self.X_test)
        return y_predict

    def predict_y(self, X_Data):
        y_predict = self.regmodel.predict(X_Data)
        return y_predict

    # Plot the tree graph(s)
    def get_tree_graph(self):
        tree_plots = []
        for tree_in_forest in self.regmodel.estimators_:
            plt.figure(figsize=(20, 20))
            tree.plot_tree(tree_in_forest, feature_names=self.X_train.columns, filled=True)
            figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
            plt.savefig(figure_name)
            plt.close()
            tree_plots.append(figure_name)

        return tree_plots

    # Plot the price difference between the predicted and the real prices of the training set
    def get_oob_prediction_graph(self):
        # Set the style
        plt.style.use('fivethirtyeight')
        # list of x locations for plotting
        prediction = list(self.regmodel.oob_prediction_)
        difference = self.y_train - prediction
        x_values = list(range(len(difference)))
        # Make a bar chart
        plt.bar(x_values, difference, orientation='vertical')
        # Axis labels and titlerence')
        plt.xlabel('Number')
        plt.title('Difference Predicted/Real')
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name

    # Get the oob_score
    def get_oob_score(self):
        score = self.regmodel.oob_score_

        return score
