# Classification
# ensemble models
# linear models
# Non-linear models
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# for hyperparameter optimization
from analysis.analysis_interface import AnalysisInterface


# random Number generation
# for splitting data into training and test
from response import Response


class KnnClassifier(AnalysisInterface):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        try:
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.model.fit(self.X_train, self.y_train)
        except:
            return Response.failure('Error building model')
        return Response.success(self.model)

    def get_train_score(self):
        score = self.model.score(self.X_train, self.y_train)
        return score

    def get_test_score(self):
        score = self.model.score(self.X_test, self.y_test)
        return score

    def predict_y_from_x_test(self):
        y_predict = self.model.predict(self.X_test)
        return y_predict

    def predict_y(self, X_Data):
        y_predict = self.model.predict(X_Data)
        return y_predict

    def result_df(self):
        return pd.DataFrame({'Actual': self.y_test, 'Predicted': self.predict_y_from_x_test()})
