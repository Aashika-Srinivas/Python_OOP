# The libraries are imported
import random
import time

import matplotlib.pyplot as plt
import numpy as np
# added
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from analysis.analysis_interface import AnalysisInterface
from response import Response

"""
This class performs classification using bagging classier algorithm
"""


class BgnClassifier(AnalysisInterface):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self, seed, n_est, max_feat, max_samp):
        """
        This function builds the bagging model
        :param seed:
        :param n_est:
        :param max_feat:
        :param max_samp:
        :return:
        """

        try:
            base_classifier = DecisionTreeClassifier()  # Decision Tree classifier is used as base classifier

            # The bagging classifier is developed
            bg_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=n_est,
                                              max_features=max_feat,
                                              max_samples=max_samp,
                                              random_state=seed)
            bg_classifier.fit(self.X_train, self.y_train)

            self.bg_classifier = bg_classifier

        except:
            return Response.failure('Error building model')

        return Response.success(bg_classifier)

    def get_train_score(self):
        """
        This function gets the train score of the model
        :return:
        """
        score = self.bg_classifier.score(self.X_train, self.y_train)
        return score

    def get_test_score(self):
        """
        This function gets the score based on test data
        :return:
        """
        score = self.bg_classifier.score(self.X_test, self.y_test)
        return score

    def predict_y_from_x_test(self):
        """
        This function is used to predict the y values based on x values
        :return:
        """
        y_predict = self.bg_classifier.predict(self.X_test)
        return y_predict

    def predict_y(self, X_Data):
        """
        This function is used to predict any y value given x
        :param X_Data:
        :return:
        """
        y_predict = self.bg_classifier.predict(X_Data)
        return y_predict

    # added
    def get_plot(self):
        """
        This function is used to plot a confusion matrix of the model
        :return:
        """

        y_predict = self.predict_y_from_x_test()
        y = np.array(y_predict)
        label = np.unique(y_predict)

        cmatrix = confusion_matrix(self.y_test.values, y_predict)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cmatrix, annot=True, linewidths=0.5, xticklabels=label, yticklabels=label)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('RFC Confusion Matrix')
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)

        return figure_name

    def get_tree_plot(self):
        # This function is used to plot the tree of the result.
        tree_plots = []
        for tree_in_forrest in self.bg_classifier.estimators_:
            plt.figure(figsize=(20, 20))
            tree.plot_tree(tree_in_forrest, feature_names=self.X_train.columns, filled=True)
            figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
            plt.savefig(figure_name)
            plt.close()
            tree_plots.append(figure_name)
        return tree_plots
