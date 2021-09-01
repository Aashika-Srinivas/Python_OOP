# Importing necessary libraries
import random
import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from IPython.display import Image, display
import pydot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Random Forest as a class
from analysis.analysis_interface import AnalysisInterface
from response import Response

class RFRegression(AnalysisInterface):

    # Constructor method
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Destructor method
    def __del__(self):
        print('The Scikit Learn Random Forest Regressor has been deleted')

    # Initialize and Train Regression Model
    def build_model(self, no_of_estimators, measurement_criterion, max_depth, min_samples_split,
                    min_samples_leaf, min_weight_fraction_leaf):
        self.no_of_estimators = no_of_estimators
        self.measurement_criterion = measurement_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf

        #Try-Except Statement to identify if model was not built correctly
        try:
            reg_model = RandomForestRegressor(n_estimators=no_of_estimators, criterion=measurement_criterion,
                                              max_depth=max_depth, min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              min_weight_fraction_leaf=min_weight_fraction_leaf)
            reg_model.fit(self.X_train, self.y_train)
            self.reg_model = reg_model
        except:
            return Response.failure('Error building model')
        return Response.success(reg_model)

    #Get R2 score using train data
    def get_train_score(self):
        score = self.reg_model.score(self.X_train, self.y_train)
        return score

    # Get R2 score using test data
    def get_test_score(self):
        score = self.reg_model.score(self.X_test, self.y_test)
        return score

    # Get RMSE score using test data
    def get_score_rmse_test(self):
        score_rmse_test = mean_squared_error(self.y_test, self.predict_y_from_x_test(), squared=False)
        return score_rmse_test

    # Get MSE score using test data
    def get_score_mse_test(self):
        score_mse_test = mean_squared_error(self.y_test, self.predict_y_from_x_test())
        return score_mse_test

    # Get MAE score using test data
    def get_score_mae_test(self):
        score_mae_test = mean_absolute_error(self.y_test, self.predict_y_from_x_test())
        return score_mae_test

    # Predict values from X_test
    def predict_y_from_x_test(self):
        y_predict = self.reg_model.predict(self.X_test)
        return y_predict

    #Predict values from (new) X_Data    
    def predict_y(self, X_Data):
        y_predict = self.reg_model.predict(X_Data)
        return y_predict

    # Plot the tree graph(s)
    def get_tree_graph(self):
        tree_plots = []
        for tree_in_forest in self.reg_model.estimators_:
            plt.figure(figsize=(20, 20))
            tree.plot_tree(tree_in_forest, feature_names=self.X_train.columns, filled=True)
            figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
            plt.savefig(figure_name)
            plt.close()
            tree_plots.append(figure_name)
        return tree_plots

    # Plot the importances graph
    def get_importances_graph(self):
        # list of x locations for plotting
        importances = list(self.reg_model.feature_importances_)
        x_values = list(range(len(importances)))
        # Make a bar chart
        plt.bar(x_values, importances, orientation='vertical')
        # Tick labels for x axis
        feature_list = list(self.X_train.columns)
        plt.xticks(x_values, feature_list, rotation='vertical')
        # Axis labels and title
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title('Variable Importance')
        plt.tight_layout()
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name
