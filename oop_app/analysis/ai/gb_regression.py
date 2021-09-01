# Importing necessary libraries

import random
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Random Forest as a class
from analysis.analysis_interface import AnalysisInterface
from response import Response

class GradientBoostingReg(AnalysisInterface):

    # Constructor method
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Destructor method
    def __del__(self):
        print('The Scikit Learn Gradient Boosting Regressor has been deleted')

    # Initialize and Train Regression Model
    def build_model(self, loss, learning_rate, subsample, no_of_estimators,
                    measurement_criterion, max_depth, min_samples_split, min_samples_leaf,
                    min_weight_fraction_leaf):
        self.no_of_estimators = no_of_estimators
        self.measurement_criterion = measurement_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.loss = loss
        self.learning_rate = learning_rate
        self.subsample = subsample

        #Try-Except Statement to identify if model was not built correctly
        try:
            boost_reg_model = GradientBoostingRegressor(n_estimators=no_of_estimators, criterion=measurement_criterion,
                                                        max_depth=max_depth, min_samples_split=min_samples_split,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf)
            boost_reg_model.fit(self.X_train, self.y_train)
            self.boost_reg_model = boost_reg_model

        except:
            return Response.failure('Error building model')

        return Response.success(self.boost_reg_model)

    #Get R2 score using train data
    def get_train_score(self):
        score = self.boost_reg_model.score(self.X_train, self.y_train)
        return score

    # Get R2 score using test data
    def get_test_score(self):
        score = self.boost_reg_model.score(self.X_test, self.y_test)
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
        y_predict = self.boost_reg_model.predict(self.X_test)
        return y_predict

    #Predict values from (new) X_Data  
    def predict_y(self, X_Data):
        y_predict = self.boost_reg_model.predict(X_Data)
        return y_predict

    # Build 3 graphs that represent the gradient boosting regression results
    def get_deviance_graph(self):
        # 1st graph:
        test_score = np.zeros((self.no_of_estimators), dtype=np.float64)
        for i, y_pred in enumerate(self.boost_reg_model.staged_predict(self.X_test)):
            test_score[i] = self.boost_reg_model.loss_(self.y_test, y_pred)

        plt.title('Deviance', loc='center')
        plt.plot(np.arange(self.no_of_estimators) + 1, self.boost_reg_model.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(self.no_of_estimators) + 1, test_score, 'r-', label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.tight_layout()

        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name

    def get_feature_importance_graph(self):
        feature_list = list(self.X_train.columns)

        # 2nd graph: Feature Importance (MDI)
        feature_importance = self.boost_reg_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_list)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (MDI)', loc='center')
        plt.tight_layout()

        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name

        # Predict new results

    def get_permutation_graph(self):
        feature_list = list(self.X_train.columns)

        # 3rd graph: Permutation Importance (test set)
        result = permutation_importance(self.boost_reg_model, self.X_test, self.y_test, n_repeats=10, random_state=42,
                                        n_jobs=2)
        sorted_idx2 = result.importances_mean.argsort()
        plt.boxplot(result.importances[sorted_idx2].T, vert=False, labels=np.array(feature_list)[sorted_idx2])
        plt.title('Permutation Importance (test set)', loc='center')
        plt.xlabel('Permutation Importance')
        plt.tight_layout()

        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()
        return figure_name
    
