from abc import ABC, abstractmethod

'''
An interface to ensure that all the analysis class implement all the expected methods
'''


class AnalysisInterface(ABC):

    @abstractmethod
    def __init__(self, X_train, X_test, y_train, y_test):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_train_score(self):
        pass

    @abstractmethod
    def get_test_score(self):
        pass

    @abstractmethod
    def predict_y_from_x_test(self):
        pass

    @abstractmethod
    def predict_y(self, X_Data):
        pass
