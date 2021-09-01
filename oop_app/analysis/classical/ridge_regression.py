# importing linear models
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

from analysis.analysis_interface import AnalysisInterface
from response import Response


class RidgeRegression(AnalysisInterface):

    # Initialise
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Predicting Y column using Linear regression
    def build_model(self, tune_alpha=False, ridge_alpha=None):
        try:
            if tune_alpha:
                param_grid = {'alpha': sp_rand()}
                # create and fit a ridge regression model, testing random alpha values
                model = Ridge()
                rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
                rsearch.fit(self.X_train, self.y_train)
                # summarize the results of the random parameter search
                tuned_alpha = rsearch.best_estimator_.alpha
                # now we test lasso regression.
                self.ridge_reg = Ridge(alpha=tuned_alpha, random_state=42)
            else:
                if ridge_alpha is None:
                    self.ridge_reg = Ridge(random_state=42)
                else:
                    self.ridge_reg = Ridge(alpha=ridge_alpha, random_state=42)

            # fitting the data on the model
            self.ridge_reg.fit(self.X_train, self.y_train)
        except:
            return Response.failure('Error building model')
        return Response.success(self.ridge_reg)

    def get_train_score(self):
        y_predict = self.predict_y(self.X_train)
        r2score = r2_score(self.y_train, y_predict)
        return r2score

    def get_test_score(self):
        y_predict = self.predict_y(self.X_test)
        r2score = r2_score(self.y_test, y_predict)
        return r2score

    def predict_y_from_x_test(self):
        y_predict = self.ridge_reg.predict(self.X_test)
        return y_predict

    def predict_y(self, X_Data):
        y_predict = self.ridge_reg.predict(X_Data)
        return y_predict

    # Root mean squared(rms) error calculation
    def get_rmse(self, y_predict):
        rmse = np.sqrt(mean_squared_error(self.y_test, y_predict))
        return rmse

    def get_score_mse_test(self):
        score_mse_test = mean_squared_error(self.y_test, self.predict_y_from_x_test())
        return score_mse_test

    def get_score_mae_test(self):
        score_mae_test = mean_absolute_error(self.y_test, self.predict_y_from_x_test())
        return score_mae_test

    def get_r2score(self, y_predict):
        # score-root mean square error
        r2score = r2_score(self.y_test, y_predict)
        return r2score

    def get_plot(self, y_predict):
        plt.xlabel('Index')
        plt.ylabel('Y values')

        plt.plot(self.y_test.values, color='black', label='Actual Y')
        plt.plot(y_predict, color='blue', label='Predicted Y', linewidth=1)

        plt.legend(loc='lower right')
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.close()

        return figure_name
