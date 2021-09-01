# importing linear models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import numpy as np

class LinRegression:

    #Initialise
    def __init__ (self,X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    #Predicting Y column using Linear regression
    def get_predict_y(self):
        lin_reg = LinearRegression()

        #fitting the data on the model
        lin_reg.fit(self.X_train, self.y_train)

        #predicted output
        y_predict = lin_reg.predict(self.X_test)
        return y_predict

    #Root mean squared(rms) error calculation
    def get_rms(self, y_predict):
        self.y_predict = y_predict
        rmse=np.sqrt(mean_squared_error(self.y_test,self.y_predict))
        result = dict()
        return rmse

    def get_r2score(self, y_predict):
        self.y_predict = y_predict
        #score-root mean square error
        r2score=r2_score(self.y_test,self.y_predict)
        return r2score

    def get_r2score(self, y_predict):
        plt.scatter(x, y)
        plt.plot(x, y_predict)

        plt.xlabel("")
        plt.ylabel(self.y_column)

        plt.scatter(x, y, color='black')
        plt.plot(x, y_new, color='blue', linewidth=2)
