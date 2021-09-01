from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
# To graph the results
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Neural Network as a class
from analysis.analysis_interface import AnalysisInterface
from response import Response


class NNRegressor(AnalysisInterface):

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
        print('The Keras Regressor Neural Network has been deleted')

    # Create Model with the design parameters as inputs
    def build_model(self, opt, loss, fun, init, epo, batch, nn):
        self.opt = opt
        self.loss = loss
        self.fun = fun
        self.init = init
        self.epo = epo
        self.batch = batch
        self.nn = nn

        # Try - Except statement the building of the model fails
        try:
            # when X_train (input) is 1 a direct assignment is needed, otherwise there is a problem here inputs=
            # X_train.shape[1]
            if len(self.X_train.shape) == 1:
                inputs = 1
            else:
                inputs = self.X_train.shape[1]

            # Define the keras base model
            def baseline_model():
                keras_model = Sequential()
                keras_model.add(Dense(self.nn[0], input_dim=inputs, kernel_initializer=self.init, activation=self.fun))
                keras_model.add(Dense(self.nn[1], kernel_initializer=self.init, activation=self.fun))
                keras_model.add(Dense(1, kernel_initializer=self.init))

                # Compile  model
                keras_model.compile(loss=self.loss, optimizer=self.opt, metrics=['mse', 'mae'])
                return keras_model

            # Build and fit the model
            self.estimator = KerasRegressor(build_fn=baseline_model, epochs=self.epo, batch_size=self.batch, verbose=0)
            self.estimator.fit(self.X_train, self.y_train)

        except:
            return Response.failure('Error building model')

        return Response.success(self.estimator)

    # Predict values from X_test
    def predict_y_from_x_test(self):
        y_predict = self.estimator.predict(self.X_test)
        return y_predict

    # Predict values from new X_data
    def predict_y(self, X_Data):
        y_predict = self.estimator.predict(X_Data)
        return y_predict
    
    # Get R2 score using train data
    def get_train_score(self):
        score = r2_score(self.y_train, self.estimator.predict(self.X_train))
        return score

    # Get R2 score using test data
    def get_test_score(self):
        score = r2_score(self.y_test, self.predict_y_from_x_test())
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

    # Plot original output vs predicted output (using test data)
    def get_plot(self):
        plt.plot(self.y_test.values, label='y original')
        plt.plot(self.predict_y_from_x_test(), label='y predicted')
        plt.legend()
        figure_name = 'plots/figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        plt.savefig(figure_name)
        plt.show()
        plt.close()
        return figure_name
