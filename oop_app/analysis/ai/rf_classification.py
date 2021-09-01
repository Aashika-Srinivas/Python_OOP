# Author: Quentin Lyons
# Program: Random Forrest Classifier
# Date: 09/04/2021
# References
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# The following represents a class of RandomForrest implementation
#
from analysis.analysis_interface import AnalysisInterface
from response import Response

''' This class defines the blue print of AI method - Random Forrest'''


class AiRFClassifier(AnalysisInterface):
    """Documentation for Class artificial
    The class can handle multiple AI methods

    More Details
    """

    def __init__(self, X_train, X_test, y_train, y_test):

        """The constructor
        Initialise Model data
        """
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test
        self.__classifier = None
        self.__predictionValues = None
        self.importances = list()

    def __del__(self):
        print('AI Model has been deleted')

        # setter methods

    def build_model(self, n_est, crit, minSampleSplit, mdepth, mfeature, r_state):
        """
        Sets the classifier to the type of Random Forrest Classifier

        More Details
        """
        try:
            self.__classifier = RandomForestClassifier(n_estimators=n_est, criterion=crit,
                                                       min_samples_split=minSampleSplit,
                                                       max_depth=mdepth, max_features=mfeature, random_state=r_state)

            self.__classifier.fit(X=self.__X_train, y=self.__y_train, sample_weight=None)
            # Obtain the importance features of the split
            self.importances = self.__classifier.feature_importances_

        except:
            return Response.failure('Error building model')

        return Response.success(self.__classifier)

        # Get methods

    def get_train_score(self):
        score = self.__classifier.score(self.__X_train, self.__y_train)
        return score

    def get_test_score(self):
        score = self.__classifier.score(self.__X_test, self.__y_test)
        return score

    def predict_y_from_x_test(self):
        y_predict = self.__classifier.predict(self.__X_test)
        return y_predict

    def predict_y(self, X_Data):
        y_predict = self.__classifier.predict(X_Data)
        return y_predict

    # Get methods
    def get_classifier(self):
        """
        Get classifier type
        More Details
        """
        return self.__classifier

    def get_confusionPlot(self):
        """
        Get model Parameters
        More Details
        """
        # print(Ydata)
        Ydata = self.predict_y_from_x_test()

        y = np.array(Ydata)
        label = np.unique(y)
        confuseMatrix = confusion_matrix(self.__y_test, Ydata)
        plt.figure(figsize=(10, 10))
        sns.heatmap(confuseMatrix, annot=True, linewidths=0.5, xticklabels=label, yticklabels=label)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('RFC Confusion Matrix')
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()

        return figure_name

    def get_tree_graph(self):
        """
        Gets the tree Model plots
        More Details
        """
        # Cycle through the trees and build model
        # Reference: https://mljar.com/blog/visualize-tree-from-random-forest/
        tree_plots = []
        for tree_in_forrest in self.get_classifier().estimators_:
            plt.figure(figsize=(20, 20))
            tree.plot_tree(tree_in_forrest, feature_names=self.__X_train.columns, filled=True)
            figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
            plt.savefig(figure_name)
            plt.close()
            tree_plots.append(figure_name)
        return tree_plots

    def get_visual_plot(self):
        """
        Gets the Models Features of importance
        More Details
        """
        # Set up plot parameters
        plt.figure(figsize=(20, 20))
        width = 0.2
        # Get the feature importances
        importances = self.importances
        # Set the location for Bar plot
        xValues = list(range(len(importances)))
        plt.bar(xValues, importances, width, color='r')
        # Set the plot labels
        plt.xticks(xValues, self.__X_test.columns)
        plt.xlabel('Features', fontsize=16)
        plt.ylabel('Importance', fontsize=16)
        plt.title('Features Of Importance', fontsize=20)
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name