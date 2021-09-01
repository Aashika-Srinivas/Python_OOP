from tkinter import *
from tkinter import messagebox, simpledialog, filedialog

import pandas as pd
from PIL import ImageTk, Image

import app_constants
from analysis.ai.bagging_classifier import BgnClassifier
from analysis.ai.bagging_regression import BaggingReg
from analysis.ai.gb_classification import GradientBoostingClassification
from analysis.ai.gb_regression import GradientBoostingReg
from analysis.ai.keras_nn_regression import NNRegressor
from analysis.ai.nn_classification import NeuralClassifier
from analysis.ai.rf_classification import AiRFClassifier
from analysis.ai.rf_regression import RFRegression
from analysis.classical.dt_classifier import DtClassifier
from analysis.classical.dtr_regression import DTRRegression
from analysis.classical.knn_classifier import KnnClassifier
from analysis.classical.knn_regression import KNNRegression
from analysis.classical.lasso_regression import LassoRegression
from analysis.classical.linear_regression import LinRegression
from analysis.classical.lr_classifier import LrClassifier
from analysis.classical.nb_classifier import NbClassifier
from analysis.classical.rf_classifier import RfClassifier
from analysis.classical.ridge_regression import RidgeRegression
from analysis.classical.svm_classifier import SvmClassifier
from data_preparation.other_functions import CleanData
from gui_components.dialogs.multiple_plot_box import MultiplePlotBox
from gui_components.dialogs.option_dialog import OptionDialog
from gui_components.dialogs.predict_box import PredictBox
from gui_components.dialogs.text_scrollbar import TextScrollCombo
from gui_components.dialogs.zoom_box import ZoomBox
from gui_components.widgets.table_view import TableView

"""
This function is used to perform the actual analysis
"""


class AnalyseFrame(Frame):
    def __init__(self, parent, data_frame, x_columns, y_column, master=None):
        super().__init__(master)

        self.parent = parent

        self._df = data_frame
        self.x_columns = x_columns
        self.y_column = y_column
        self.analysis_type = parent.get_analysis_type()

        self.main_frame = Frame(self, pady=3)
        self.main_frame.grid(row=0, sticky='ew')

        self.top_frame1 = Frame(self.main_frame, pady=3)
        self.top_frame1.grid(row=0)

        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)

        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        if self.analysis_type == 'Classification':
            title = 'Perform Classification Analysis'
            algorithms = ['Classical | Logistic Regression Classification',
                          'Classical | K-Nearest Neighbours Classification',
                          'Classical | State Vector Machine Classification',
                          'Classical | Naive Bayes Classification',
                          'Classical | Decision Tree Classification',
                          'Classical | Random Forest Classification',
                          'AI | Random Forrest Classifier',
                          'AI | Gradient Boosting Classifier',
                          'AI | Bagging Classifier',
                          'AI | Neural Network Classifier',
                          'Compare Classification Algorithms'
                          ]
        else:
            title = 'Perform Regression Analysis'
            algorithms = ['Classical | Linear Regression', 'Classical | Lasso Regression',
                          'Classical | Ridge Regression', 'Classical | KNN Regression',
                          'Classical | Decision Tree Regression',
                          'AI | Keras Neural Network Regressor',
                          'AI | Random Forrest Regressor',
                          'AI | Gradient Boosting Regression',
                          'AI | Bagging Regressor',
                          'Compare Regression Algorithms'
                          ]

        title_label = Label(self.top_frame1, text=title,
                            fg='blue', font=('Arial', 16))
        x_col_names = 'Independent (x) columns: '
        for index, col in enumerate(self.x_columns, start=1):
            if index == 1:
                x_col_names = x_col_names + col
            elif index == len(self.x_columns):
                x_col_names = x_col_names + ' and ' + col + '.'
            else:
                x_col_names = x_col_names + ', ' + col

        x_col_label = Label(self.top_frame1, text=x_col_names, fg='blue')
        y_col_label = Label(self.top_frame1, text='Dependent (y) column: ' + self.y_column, fg='blue')

        analyse_button = Button(self.top_frame1, background='green', fg='white', relief='raised',
                                text='Analyse',
                                command=self.analyse)

        percentages = []
        for num in range(5, 105, 5):
            percentages.append(num)

        training_label = Label(self.top_frame1, text='Percentage of data for training:')

        self.training_percent = StringVar(self.top_frame1)

        self.training_percent.set(percentages[1])
        training_percent_option_menu = OptionMenu(self.top_frame1, self.training_percent, *percentages)

        algorithm_label = Label(self.top_frame1, text='Select Algorithm:')
        self.algorithm_name = StringVar(self)
        algorithm_options = algorithms
        self.algorithm_name.set(algorithm_options[0])
        algorithm_option_menu = OptionMenu(self.top_frame1, self.algorithm_name, *algorithm_options)

        random_label = Label(self.top_frame1, text='Select how you want to split:')
        self.random_option_name = StringVar(self)
        random_options = ['Split by random', 'Split sequential']
        self.random_option_name.set(random_options[0])

        random_option_menu = OptionMenu(self.top_frame1, self.random_option_name, *random_options)

        self.use_default_option = StringVar(self)
        use_default_options = ['Change Parameters', 'Use Defaults']
        self.use_default_option.set(use_default_options[0])

        use_default_option_menu = OptionMenu(self.top_frame1, self.use_default_option, *use_default_options)

        view_button = Button(self.top_frame1, background='blue', fg='white', relief='raised',
                             text='View DF',
                             command=self.view)

        title_label.grid(row=0, column=0, columnspan=10)
        x_col_label.grid(row=1, column=0, columnspan=10)
        y_col_label.grid(row=2, column=0, columnspan=10)
        view_button.grid(row=3, column=0, columnspan=10)
        algorithm_label.grid(row=4, column=0, padx=(5, 2))
        algorithm_option_menu.grid(row=4, column=1, padx=(2, 5))
        training_label.grid(row=4, column=2, padx=(5, 2))
        training_percent_option_menu.grid(row=4, column=4, padx=(2, 5))
        random_label.grid(row=4, column=5, padx=(5, 2))
        random_option_menu.grid(row=4, column=6, padx=(2, 5))
        use_default_option_menu.grid(row=4, column=7, padx=(5, 2))
        analyse_button.grid(row=5, column=0, columnspan=10)

    def analyse(self):
        """
        This function is called when the analyse button is clicked
        :return:
        """
        self.center_frame.grid_forget()
        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)
        train_percent = int(self.training_percent.get())
        if self.random_option_name.get() == 'Split by random':
            if self.use_default_option.get() == 'Use Defaults':
                result = CleanData.split_data(self._df[self.x_columns], self._df[self.y_column], train_percent, True,
                                              app_constants.DEFAULT_RANDOM_STATE)
            else:
                seed = simpledialog.askinteger('Input', 'Enter slitting seed (random state). Enter 0 to use default',
                                               parent=self.parent)
                if seed is None:
                    return False
                if seed == 0:
                    result = CleanData.split_data(self._df[self.x_columns], self._df[self.y_column], train_percent,
                                                  True)
                else:
                    result = CleanData.split_data(self._df[self.x_columns], self._df[self.y_column], train_percent,
                                                  True,
                                                  seed)
        else:
            result = CleanData.split_data(self._df[self.x_columns], self._df[self.y_column], train_percent)

        if not result['success']:
            return False

        result_data = result['data']
        self.x_train = result_data['x_train']
        self.x_test = result_data['x_test']
        self.y_train = result_data['y_train']
        self.y_test = result_data['y_test']

        # Classical Algorithms
        # Regression
        if self.algorithm_name.get() == 'Classical | Linear Regression':
            self.analysis = LinRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            train_r2_score = self.analysis.get_train_score()
            test_r2_score = self.analysis.get_test_score()
            rmse = self.analysis.get_rmse(self.y_predict_from_test)
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            figure = self.analysis.get_plot(self.y_predict_from_test)

            text = 'Linear Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(
                test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)
            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Lasso Regression':
            self.analysis = LassoRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                alpha = app_constants.LASSO_ALPHA
                response = self.analysis.build_model(False, alpha)
            else:
                dlg = OptionDialog(self.parent, 'Select One', 'Lasso alpha',
                                   ['Enter value', 'Use default', 'Use tuned value'])

                if dlg.result is None:
                    return False
                alpha = None
                if dlg.result == 'Enter value':
                    alpha = simpledialog.askfloat('Input', 'Enter Alpha',
                                                  parent=self.parent)
                    if alpha is None:
                        return False
                    response = self.analysis.build_model(False, alpha)
                elif dlg.result == 'Use tuned value':
                    response = self.analysis.build_model(True)
                else:
                    response = self.analysis.build_model()

                if alpha is None:
                    alpha = dlg.result

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            train_r2_score = self.analysis.get_train_score()
            test_r2_score = self.analysis.get_test_score()
            rmse = self.analysis.get_rmse(self.y_predict_from_test)
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            figure = self.analysis.get_plot(self.y_predict_from_test)

            text = 'Lasso Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(
                test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + \
                   '\n\n................\n' + '\nLasso alpha: ' + str(alpha)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Ridge Regression':
            self.analysis = RidgeRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                alpha = app_constants.RIDGE_ALPHA
                response = self.analysis.build_model(False, alpha)
            else:
                dlg = OptionDialog(self.parent, 'Select One', 'Ridge alpha',
                                   ['Enter value', 'Use default', 'Use tuned value'])

                if dlg.result is None:
                    return False
                alpha = None
                if dlg.result == 'Enter value':
                    alpha = simpledialog.askfloat('Input', 'Enter alpha',
                                                  parent=self.parent)
                    if alpha is None:
                        return False
                    response = self.analysis.build_model(False, alpha)
                elif dlg.result == 'Use tuned value':
                    response = self.analysis.build_model(True)
                else:
                    response = self.analysis.build_model()

                if alpha is None:
                    alpha = dlg.result

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            train_r2_score = self.analysis.get_train_score()
            test_r2_score = self.analysis.get_test_score()
            rmse = self.analysis.get_rmse(self.y_predict_from_test)
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            figure = self.analysis.get_plot(self.y_predict_from_test)

            text = 'Ridge Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(
                test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + '\n\n................\n' + \
                   '\nRidge alpha: ' + str(alpha)           

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | KNN Regression':
            self.analysis = KNNRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                tuned_alpha = False
                response = self.analysis.build_model(tuned_alpha)
            else:
                dlg = OptionDialog(self.parent, 'Select One', 'Tune alpha', ['Yes', 'No'])

                if dlg.result is None:
                    return False

                if dlg.result == 'Yes':
                    response = self.analysis.build_model(True)
                else:
                    response = self.analysis.build_model()

                tuned_alpha = dlg.result

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            train_r2_score = self.analysis.get_train_score()
            test_r2_score = self.analysis.get_test_score()
            rmse = self.analysis.get_rmse(self.y_predict_from_test)
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            figure = self.analysis.get_plot(self.y_predict_from_test)

            text = 'KNN Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(
                test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + '\n\n................\n' + \
                   '\nTune alpha?: ' + str(tuned_alpha)          

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Decision Tree Regression':
            self.analysis = DTRRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                tuned_alpha = False
                response = self.analysis.build_model(tuned_alpha)
            else:
                dlg = OptionDialog(self.parent, 'Select One', 'Tune alpha', ['Yes', 'No'])
                if dlg.result is None:
                    return False

                if dlg.result == 'Yes':
                    response = self.analysis.build_model(True)
                else:
                    response = self.analysis.build_model()

                tuned_alpha = dlg.result

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            train_r2_score = self.analysis.get_train_score()
            test_r2_score = self.analysis.get_test_score()
            rmse = self.analysis.get_rmse(self.y_predict_from_test)
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            figure = self.analysis.get_plot(self.y_predict_from_test)

            text = 'Decision Tree Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(
                test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + '\n\n................\n' + \
                   '\nTune alpha?: ' + str(tuned_alpha)
          

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))

        # Classification
        elif self.algorithm_name.get() == 'Classical | Logistic Regression Classification':
            self.analysis = LrClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'Logistic Regression Classification: \ntrain_score = ' + str(train_score) + ' \ntest_score = ' + str(
                test_score)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))

        elif self.algorithm_name.get() == 'Classical | K-Nearest Neighbours Classification':
            self.analysis = KnnClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'K-Nearest Neighbours Classification: \ntrain_score = ' + str(
                train_score) + ' \ntest_score = ' + str(test_score)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | State Vector Machine Classification':
            self.analysis = SvmClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'State Vector Machine Classification: \ntrain_score = ' + str(
                train_score) + ' \ntest_score = ' + str(test_score)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Naive Bayes Classification':
            self.analysis = NbClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'Naive Bayes Classification: \ntrain_score = ' + str(train_score) + ' \ntest_score = ' + str(
                test_score)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Decision Tree Classification':
            self.analysis = DtClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'Decision Tree Classification: \ntrain_score = ' + str(train_score) + ' \ntest_score = ' + str(
                test_score)            

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'Classical | Random Forest Classification':
            self.analysis = RfClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = self.analysis.build_model()
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            result_df = self.analysis.result_df()

            text = 'Random Forest Classification: \ntrain_score = ' + str(train_score) + ' \ntest_score = ' + str(
                test_score)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)

            df_col = list(result_df.columns.values)
            df_first_10_row = list(result_df.head(10).values)
            table_frame = Frame(self.center_frame)
            TableView(table_frame, df_col, df_first_10_row)
            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            table_frame.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))

        # AI Models
        # Regression
        elif self.algorithm_name.get() == 'AI | Keras Neural Network Regressor':
            self.analysis = NNRegressor(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                opt = app_constants.NNR_DEFAULT_OPT
                loss = app_constants.NNR_DEFAULT_LOSS
                act_fun = app_constants.NNR_DEFAULT_FUN
                init_fun = app_constants.NNR_DEFAULT_INIT
                epo = app_constants.NNR_DEFAULT_EPO
                batch = app_constants.NNR_DEFAULT_BATCH
                nn = app_constants.NNR_DEFAULT_NN
            else:
                opts = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop']
                losses = ['MAE', 'MSE', 'MAPE']
                act_funs = ['relu', 'softplus', 'selu', 'elu']
                init_funs = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal',
                             'he_uniform']
                epochs = [100, 200, 300]
                batches = [1, 2, 5]
                dlg = OptionDialog(self.parent, 'Select One', 'Select optimizer to compile the model', opts)

                if dlg.result is None:
                    return False

                opt = dlg.result
                dlg = OptionDialog(self.parent, 'Select One', 'Select objective function (losses)', losses)

                if dlg.result is None:
                    return False

                loss = dlg.result
                dlg = OptionDialog(self.parent, 'Select One', 'Select activation function', act_funs)

                if dlg.result is None:
                    return False

                act_fun = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select initialization function', init_funs)

                if dlg.result is None:
                    return False

                init_fun = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select epoch', epochs)

                if dlg.result is None:
                    return False

                epo = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select batch size', batches)

                if dlg.result is None:
                    return False

                batch = dlg.result

                nn1 = simpledialog.askinteger('Input', 'Enter number of hidden units in layer 1 (value from 8 to 16)',
                                              parent=self.parent)
                if nn1 is None or nn1 not in range(8, 16):
                    return False

                nn2 = simpledialog.askinteger('Input', 'Enter number of hidden units in layer 2 (value from 4 to 10)',
                                              parent=self.parent)
                if nn2 is None or nn2 not in range(4, 8):
                    return False

                nn = [nn1, nn2]

            response = self.analysis.build_model(opt, loss, act_fun, init_fun, epo, batch, nn)

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            mse = self.analysis.get_score_mse_test()
            rmse = self.analysis.get_score_rmse_test()
            mae = self.analysis.get_score_mae_test()
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure = self.analysis.get_plot()

            text = 'Keras Regressor (Regression) Analysis: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(
                test_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(mse) + '\ntest_MAE = ' + str(mae) + '\n\n................\n' + \
                   '\nOptimizer: ' + opt + '\nLoss: ' + loss + '\nActivation function: ' + act_fun + \
                   '\nInitialization function: ' + init_fun + '\nEpoch: ' + str(epo) + '\nBatch size: ' + str(batch) + \
                   '\nNumber of neurons: ' + str(nn)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5), columnspan=2)
            view_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'AI | Random Forrest Regressor':
            self.analysis = RFRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                no_of_estimators = app_constants.RFR_DEFAULT_NO_OF_ESTIMATORS
                criterion = app_constants.RFR_DEFAULT_MEASUREMENT_CRITERION
                max_depth = app_constants.RFR_DEFAULT_MAX_DEPTH
                min_samples_split = app_constants.RFR_DEFAULT_MIN_SAMPLES_SPLIT
                min_samples_leaf = app_constants.RFR_DEFAULT_MIN_SAMPLES_LEAF
                min_weight_fraction_leaf = app_constants.RFR_DEFAULT_MIN_WEIGHT_FRACTION_LEAF
            else:
                criterions = ['mse', 'mae']
                no_of_estimators = simpledialog.askinteger('Input', 'Enter number of trees in the forest',
                                                           parent=self.parent)
                if no_of_estimators is None:
                    return False

                dlg = OptionDialog(self.parent, 'Select One', 'Select measurement_criterion', criterions)

                if dlg.result is None:
                    return False

                criterion = dlg.result
                min_samples_split = simpledialog.askinteger('Input', 'Enter the min number of samples required '
                                                                     'to split an internal node', parent=self.parent)
                if min_samples_split is None:
                    return False

                max_depth = simpledialog.askinteger('Input', 'Enter maximum depth of the tree',
                                                    parent=self.parent)

                if max_depth is None:
                    return False

                min_samples_leaf = simpledialog.askinteger('Input', 'Enter the minimum number of samples required '
                                                                    'to be at a leaf node', parent=self.parent)

                if min_samples_leaf is None:
                    return False

                min_weight_fraction_leaf = simpledialog.askfloat('Input', 'Enter The minimum weighted fraction of the '
                                                                          'sum total of weights (0 - 0.5)',
                                                                 parent=self.parent)
                if min_weight_fraction_leaf is None:
                    return False

            response = self.analysis.build_model(no_of_estimators, criterion, max_depth, min_samples_split,
                                                 min_samples_leaf,
                                                 min_weight_fraction_leaf)

            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            test_rmse = self.analysis.get_score_rmse_test()
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure = self.analysis.get_importances_graph()
            self.plot_list = self.analysis.get_tree_graph()

            text = 'Random Forest Regression: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(
                test_score) + '\ntest_RMSE = ' + str(test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) +'\n\n................\n' + \
                   '\nno_of_estimators: ' + str(no_of_estimators) + '\nmeasurement_criterion : ' + str(
                criterion) + '\nmax_depth: ' + \
                   str(max_depth) + '\nmin_samples_split: ' + str(min_samples_split) + '\nmin_samples_leaf: ' + str(
                min_samples_leaf) + \
                   '\nmin_weight_fraction_leaf: ' + str(min_weight_fraction_leaf)           

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View Trees', command=self.view_multiple_plots)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'AI | Gradient Boosting Regression':
            self.analysis = GradientBoostingReg(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                loss = app_constants.GBR_DEFAULT_LOSS
                learning_rate = app_constants.GBR_DEFAULT_LEARNING_RATE
                sub_sample = app_constants.GBR_DEFAULT_SUBSAMPLE
                no_of_estimators = app_constants.GBR_DEFAULT_NO_OF_ESTIMATORS
                criterion = app_constants.GBR_DEFAULT_MEASUREMENT_CRITERION
                max_depth = app_constants.GBR_DEFAULT_MAX_DEPTH
                min_samples_split = app_constants.GBR_DEFAULT_MIN_SAMPLES_SPLIT
                min_samples_leaf = app_constants.GBR_DEFAULT_MIN_SAMPLES_LEAF
                min_weight_fraction_leaf = app_constants.GBR_DEFAULT_MIN_WEIGHT_FRACTION_LEAF
            else:
                criterions = ['mse', 'mae']
                losses = ['ls', 'lad', 'huber', 'quantile']
                no_of_estimators = simpledialog.askinteger('Input', 'Enter number of trees in the forest',
                                                           parent=self.parent)
                if no_of_estimators is None:
                    return False

                dlg = OptionDialog(self.parent, 'Select One', 'Select measurement_criterion', criterions)

                if dlg.result is None:
                    return False

                criterion = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select loss', losses)

                if dlg.result is None:
                    return False

                loss = dlg.result

                min_samples_split = simpledialog.askinteger('Input', 'Enter the min number of samples required '
                                                                     'to split an internal node', parent=self.parent)
                if min_samples_split is None:
                    return False

                min_samples_leaf = simpledialog.askinteger('Input', 'minimum number of samples required to be '
                                                                    'at a leaf node', parent=self.parent)
                if min_samples_leaf is None:
                    return False

                max_depth = simpledialog.askinteger('Input', 'Enter maximum depth of the tree',
                                                    parent=self.parent)

                if max_depth is None:
                    return False

                learning_rate = simpledialog.askfloat('Input', 'Enter learning rate',
                                                      parent=self.parent)

                if learning_rate is None:
                    return False

                sub_sample = simpledialog.askfloat('Input', 'Enter sub sample',
                                                   parent=self.parent)

                if sub_sample is None:
                    return False

                min_weight_fraction_leaf = simpledialog.askfloat('Input', 'Enter The minimum weighted fraction of the '
                                                                          'sum total of weights (0 - 0.5)',
                                                                 parent=self.parent)
                if min_weight_fraction_leaf is None:
                    return False

            response = self.analysis.build_model(loss, learning_rate, sub_sample, no_of_estimators, criterion,
                                                 max_depth,
                                                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            test_rmse = self.analysis.get_score_rmse_test()
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure1 = self.analysis.get_deviance_graph()
            figure2 = self.analysis.get_feature_importance_graph()
            figure3 = self.analysis.get_permutation_graph()
            self.plot_list = [figure2, figure3]

            text = 'Gradient Boosting Regression: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(
                test_score) + '\ntest_RMSE = ' + str(test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + '\n\n................\n' + \
                   '\nno_of_estimators: ' + str(no_of_estimators) + '\nmeasurement_criterion : ' + str(criterion) + \
                   '\nmax_depth: ' + str(max_depth) + '\nmin_samples_split: ' + str(min_samples_split) + \
                   '\nlearning_rate: ' + str(learning_rate) + '\nloss: ' + str(loss) + \
                   '\nmin_samples_leaf: ' + str(min_samples_leaf) + \
                   '\nmin_weight_fraction_leaf: ' + str(min_weight_fraction_leaf)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure1)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure1):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View Other plots', command=self.view_multiple_plots)
            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)
            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'AI | Bagging Regressor':
            self.analysis = BaggingReg(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                no_of_estimators = app_constants.BR_DEFAULT_NO_OF_ESTIMATORS
                max_features = app_constants.BR_DEFAULT_MAX_FEATURES
                max_samples = app_constants.BR_DEFAULT_MAX_SAMPLES
                random_state = app_constants.DEFAULT_RANDOM_STATE
                oob_score = app_constants.BR_DEFAULT_OOB_SCORE
            else:
                no_of_estimators = simpledialog.askinteger('Input', 'Enter number of trees in the forest',
                                                           parent=self.parent)
                if no_of_estimators is None:
                    return False

                dlg = OptionDialog(self.parent, 'Select One',
                                   'Use out-of-bag samples to estimate the generalization error', ['Yes', 'No'])

                if dlg.result is None:
                    return False

                oob_score = dlg.result == 'Yes'

                max_features = simpledialog.askfloat('Input', 'Enter max features',
                                                     parent=self.parent)

                if max_features is None:
                    return False

                max_samples = simpledialog.askfloat('Input', 'Enter max samples',
                                                    parent=self.parent)

                if max_samples is None:
                    return False

                random_state = simpledialog.askinteger('Input', 'Enter seed to randomly resample the original dataset',
                                                       parent=self.parent)

                if max_samples is None:
                    return False

            response = self.analysis.build_model(no_of_estimators, max_features, max_samples, random_state, oob_score)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            test_rmse = self.analysis.get_score_rmse_test()
            test_mse = self.analysis.get_score_mse_test()
            test_mae = self.analysis.get_score_mae_test()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            oob_output_score = self.analysis.get_oob_score()
            figure = self.analysis.get_oob_prediction_graph()
            self.plot_list = self.analysis.get_tree_graph()

            text = 'Bagging Regression: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(
                test_score) + '\ntest_RMSE = ' + str(test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae) + '\ntest_OOB = ' + str(oob_output_score) + '\n\n................\n' + \
                   '\nno_of_estimators: ' + str(no_of_estimators) + '\nmax_features : ' + str(max_features) + \
                   '\nmax_samples: ' + str(max_samples) + '\nrandom_state: ' + str(random_state) + \
                   '\noob_score: ' + str(oob_score)           

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View Tree plot', command=self.view_multiple_plots)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))

        # Classification
        elif self.algorithm_name.get() == 'AI | Random Forrest Classifier':
            self.analysis = AiRFClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                n_estimator = app_constants.RFC_DEFAULT_N_ESTIMATORS
                criterion = app_constants.RFC_DEFAULT_CRITERION
                min_sample_split = app_constants.RFC_DEFAULT_MIN_SAMPLES_SPLIT
                max_depth = app_constants.RFC_DEFAULT_MAX_DEPTH
                max_feature = app_constants.RFC_DEFAULT_MAX_FAETURES
                random_state = app_constants.DEFAULT_RANDOM_STATE
            else:
                criterions = ['gini', 'entropy']
                max_features = ['auto', 'sqrt', 'log2', None, 'Enter value']

                n_estimator = simpledialog.askinteger('Input', 'Enter value of n estimator (2-200) defualt is 100',
                                                      parent=self.parent)
                if n_estimator is None or n_estimator not in range(2, 200):
                    return False

                dlg = OptionDialog(self.parent, 'Select One', 'Select criterion', criterions)

                if dlg.result is None:
                    return False

                criterion = dlg.result
                min_sample_split = simpledialog.askinteger('Input',
                                                           'Enter minimum number of samples required to split a node on. Range: 2 to ~ 10',
                                                           parent=self.parent)
                if min_sample_split is None:
                    return False

                max_depth = simpledialog.askinteger('Input',
                                                    'Max tree depth: Range 2 - ~ say 200. If NONE, expands until all leaves pure. Enter 0 for NONE',
                                                    parent=self.parent)

                if max_depth is None:
                    return False

                if max_depth == 0:
                    max_depth = None

                dlg = OptionDialog(self.parent, 'Select One', 'Select max features', max_features)

                if dlg.result is None:
                    return False

                max_feature = dlg.result
                if max_feature == 'Enter value':
                    max_feature = simpledialog.askinteger('Input', 'Range 1 to ~ say 100 features at each split',
                                                          parent=self.parent)
                if max_feature is None:
                    return False

                random_state = simpledialog.askinteger('Input', 'Enter random state',
                                                       parent=self.parent)
                if random_state is None:
                    return False

            response = self.analysis.build_model(n_estimator, criterion, min_sample_split, max_depth, max_feature,
                                                 random_state)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure1 = self.analysis.get_confusionPlot()
            figure2 = self.analysis.get_visual_plot()
            self.plot_list = self.analysis.get_tree_graph()
            self.plot_list.append(figure2)

            text = 'Random Forest Classifier: \ntrain_score = ' + str(train_score) + ' \ntest_score = ' + str(
                test_score) + '\n\n................\n' + \
                   '\nn_estimator: ' + str(n_estimator) + '\ncriterion: ' + str(criterion) + '\nmin_sample_split: ' + \
                   str(min_sample_split) + '\nmax_depth: ' + str(max_depth) + '\nmax_feature: ' + str(max_feature) + \
                   '\nrandom_state: ' + str(random_state)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure1)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure1):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View Trees', command=self.view_multiple_plots)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'AI | Gradient Boosting Classifier':
            self.analysis = GradientBoostingClassification(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                learning_rate = app_constants.GB_DEFAULT_LEARNING_RATE
                n_estimator = app_constants.GB_DEFAULT_N_ESTIMATOR
                subsample = app_constants.GB_DEFAULT_SUB_SAMPLE
                criterion = app_constants.GB_DEFAULT_CRITERIION
                min_sample_split = app_constants.GB_DEFAULT_MIN_SAMPLES_SPLIT
                max_depth = app_constants.GB_DEFAULT_MAX_DEPTH
                max_feature = app_constants.GB_DEFAULT_MAX_FAETURES
                init = app_constants.GB_DEFAULT_INIT
                random_state = app_constants.DEFAULT_RANDOM_STATE
                max_leaf_nodes = app_constants.GB_DEFAULT_MAX_LEAF_NODE
            else:
                criterions = ['friedman_mse', 'mse', 'mae']
                max_features = ['auto', 'sqrt', 'log2', None, 'Enter value']
                learning_rate = simpledialog.askfloat('Input', 'Enter learning rate (0.1 - 1.0)',
                                                      parent=self.parent)
                if learning_rate is None:
                    return False

                subsample = simpledialog.askfloat('Input',
                                                  'Enter Fraction of samples used for fitting the individual base learners',
                                                  parent=self.parent)
                if subsample is None:
                    return False

                n_estimator = simpledialog.askinteger('Input', 'Enter value of n estimator (2-200) defualt is 100',
                                                      parent=self.parent)
                if n_estimator is None or n_estimator not in range(2, 200):
                    return False

                dlg = OptionDialog(self.parent, 'Select One', 'Select function (criterion) to measure split quality',
                                   criterions)

                if dlg.result is None:
                    return False

                criterion = dlg.result
                min_sample_split = simpledialog.askinteger('Input',
                                                           'Enter minimum number of samples required to split a node on. Range: 2 to ~ 10',
                                                           parent=self.parent)
                if min_sample_split is None:
                    return False

                max_leaf_nodes = simpledialog.askinteger('Input', 'Define the leaves allowed at any node',
                                                         parent=self.parent)
                if max_leaf_nodes is None:
                    return False

                max_depth = simpledialog.askinteger('Input',
                                                    'Max tree depth: Range 2 - ~ say 200. If NONE, expands until all leaves pure. Enter 0 for NONE',
                                                    parent=self.parent)

                if max_depth is None:
                    return False

                if max_depth == 0:
                    max_depth = None

                dlg = OptionDialog(self.parent, 'Select One', 'Select max features', max_features)

                if dlg.result is None:
                    return False

                max_feature = dlg.result
                if max_feature == 'Enter value':
                    max_feature = simpledialog.askinteger('Input', 'Range 1 to ~ say 100 features at each split',
                                                          parent=self.parent)
                if max_feature is None:
                    return False

                random_state = simpledialog.askinteger('Input', 'Enter random state',
                                                       parent=self.parent)
                if random_state is None:
                    return False

                init = None

            response = self.analysis.build_model(learning_rate, n_estimator, subsample, criterion,
                                                 min_sample_split, max_depth, max_feature, init, random_state,
                                                 max_leaf_nodes)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False
            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure1 = self.analysis.get_confusion_plot(self.y_predict_from_test)
            self.plot_list = []
            figure2 = self.analysis.get_visual_plot()
            self.plot_list.append(figure2)

            text = 'Gradient Boosting Classifier: \ntrain_score = ' + str(train_score) + '\ntest_score = ' + \
                   str(test_score) + '\n\n................\n' + \
                   '\nn_estimator: ' + str(n_estimator) + '\ncriterion: ' + str(criterion) + \
                   '\nmin_sample_split: ' + str(min_sample_split) + '\nmax_depth: ' + str(max_depth) + \
                   '\ninit: ' + str(init) + '\nmax_leaf_nodes: ' + str(max_leaf_nodes) + \
                   '\nmax_feature: ' + str(max_feature) + '\nrandom_state: ' + str(random_state)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure1)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure1):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View more plots', command=self.view_multiple_plots)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))
        elif self.algorithm_name.get() == 'AI | Bagging Classifier':
            self.analysis = BgnClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                max_samples = app_constants.BC_DEFAULT_MAX_SAMPLE
                n_estimator = app_constants.BC_DEFAULT_N_ESTIMATOR
                max_feature = app_constants.BC_DEFAULT_MAX_FAETURE
                random_state = app_constants.DEFAULT_RANDOM_STATE
            else:
                n_estimator = simpledialog.askinteger('Input', 'Enter value of n estimator (2-200) defualt is 100',
                                                      parent=self.parent)
                if n_estimator is None or n_estimator not in range(2, 200):
                    return False

                max_feature = simpledialog.askinteger('Input',
                                                      'Enter the number of samples to draw from X to train each base estimator',
                                                      parent=self.parent)
                if max_feature is None:
                    return False

                max_samples = simpledialog.askinteger('Input',
                                                      'Enter the number of features to draw from X to train each base estimator',
                                                      parent=self.parent)
                if max_samples is None:
                    return False

                random_state = simpledialog.askinteger('Input', 'Enter random state',
                                                       parent=self.parent)
                if random_state is None:
                    return False

            response = self.analysis.build_model(random_state, n_estimator, max_feature, max_samples)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure = self.analysis.get_plot()
            self.plot_list = self.analysis.get_tree_plot()

            text = 'Bagging Classifier: \ntrain_score = ' + str(train_score) + '\ntest_score = ' + \
                   str(test_score) + '\n\n................\n' + \
                   '\nn_estimator: ' + str(n_estimator) + '\nmax_feature: ' + str(max_feature) + \
                   '\nmax_samples: ' + str(max_samples) + '\nrandom_state: ' + str(random_state)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)

            view_plots_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                       text='View Trees', command=self.view_multiple_plots)

            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            view_plots_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=4, pady=(0, 10), padx=(5, 5))

        elif self.algorithm_name.get() == 'AI | Neural Network Classifier':
            self.analysis = NeuralClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            if self.use_default_option.get() == 'Use Defaults':
                opt = app_constants.NNC_DEFAULT_OPT
                fun = app_constants.NNC_DEFAULT_FUN
                init = app_constants.NNC_DEFAULT_INIT
                epo = app_constants.NNC_DEFAULT_EPO
                batch = app_constants.NNC_DEFAULT_BATCH
                nn = app_constants.NNC_DEFAULT_NN
            else:
                opts = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop']
                funs = ['sigmoid', 'softplus', 'softmax']
                inits = ['uniform', 'normal', 'glorot_normal', 'he_normal']
                epos = [100, 200, 300]
                batches = [1, 2, 3]

                dlg = OptionDialog(self.parent, 'Select One', 'Select optimizer to compile the model',
                                   opts)

                if dlg.result is None:
                    return False

                opt = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select activation function',
                                   funs)

                if dlg.result is None:
                    return False

                fun = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select initializacion function',
                                   inits)

                if dlg.result is None:
                    return False

                init = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select epo',
                                   epos)

                if dlg.result is None:
                    return False

                epo = dlg.result

                dlg = OptionDialog(self.parent, 'Select One', 'Select batch',
                                   batches)

                if dlg.result is None:
                    return False

                batch = dlg.result

                nn1 = simpledialog.askinteger('Input', 'Enter number of neurons in layer 1 (value from 8 to 16)',
                                              parent=self.parent)
                if nn1 is None or nn1 not in range(8, 16):
                    return False

                nn2 = simpledialog.askinteger('Input', 'Enter number of neurons in layer 2 (value from 4 to 10)',
                                              parent=self.parent)
                if nn2 is None or nn2 not in range(4, 8):
                    return False

                nn = [nn1, nn2]

            response = self.analysis.build_model(opt, fun, init, epo, batch, nn)
            if not response['success']:
                messagebox.showerror('Error', response['message'])
                return False

            train_score = self.analysis.get_train_score()
            test_score = self.analysis.get_test_score()
            self.y_predict_from_test = self.analysis.predict_y_from_x_test()
            figure = self.analysis.get_plot()

            text = 'Keras Neural Network Classifier: \ntrain_score = ' + str(train_score) + '\ntest_score = ' + \
                   str(test_score) + '\n\n................\n' + \
                   '\nopt: ' + str(opt) + '\nfun: ' + str(fun) + \
                   '\ninit: ' + str(init) + '\nepo: ' + str(epo) + \
                   '\nbatch: ' + str(batch) + '\nnn: ' + str(nn)

            result_text = Text(self.center_frame, width=50, height=18)
            result_text.insert(INSERT, text)
            result_text.config(state=DISABLED)
            image = Image.open(figure)
            image = image.resize((400, 300), Image.ANTIALIAS)
            self.plot_image = ImageTk.PhotoImage(image)
            plot_container = Label(self.center_frame, image=self.plot_image)

            # Use a temporal function to call image zoom box
            def temp(event, image=figure):
                self.zoom_image(image)

            plot_container.bind("<Button-1>", temp)

            export_result_button = Button(self.center_frame, text='Export y_test and y_predict as data frame',
                                          background='green',
                                          fg='white', command=self.export_result)

            view_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                 text='View y_test and y_predict as data frame',
                                 command=self.view_result)

            predict_button = Button(self.center_frame, background='green', fg='white', relief='raised',
                                    text='Predict',
                                    command=self.predict)
            result_text.grid(row=0, column=0, pady=(30, 30), padx=(5, 5))
            plot_container.grid(row=0, column=1, pady=(0, 10), padx=(5, 5), columnspan=3)
            export_result_button.grid(row=1, column=0, pady=(0, 10), padx=(5, 5))
            view_button.grid(row=1, column=2, pady=(0, 10), padx=(5, 5))
            predict_button.grid(row=1, column=3, pady=(0, 10), padx=(5, 5))

        # Compare Regression Algorithms
        elif self.algorithm_name.get() == 'Compare Regression Algorithms':
            # Linear Regression Analysis
            text = ''
            analysis = LinRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                y_predict = analysis.predict_y_from_x_test()
                train_r2_score = analysis.get_train_score()
                test_r2_score = analysis.get_test_score()
                rmse = analysis.get_rmse(y_predict)
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + 'Linear Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)          

            # Lasso Regression Analysis
            analysis = LassoRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            alpha = app_constants.LASSO_ALPHA
            response = analysis.build_model(False, alpha)
            if response['success']:
                y_predict = analysis.predict_y_from_x_test()
                train_r2_score = analysis.get_train_score()
                test_r2_score = analysis.get_test_score()
                rmse = analysis.get_rmse(y_predict)
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nLasso Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)          


            # Ridge Regression
            analysis = RidgeRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            alpha = app_constants.RIDGE_ALPHA
            response = analysis.build_model(False, alpha)
            if response['success']:
                y_predict = analysis.predict_y_from_x_test()
                train_r2_score = analysis.get_train_score()
                test_r2_score = analysis.get_test_score()
                rmse = analysis.get_rmse(y_predict)
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nRidge Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)          


            # KNN Regression
            analysis = KNNRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            tuned_alpha = False
            response = analysis.build_model(tuned_alpha)
            if response['success']:
                y_predict = analysis.predict_y_from_x_test()
                train_r2_score = analysis.get_train_score()
                test_r2_score = analysis.get_test_score()
                rmse = analysis.get_rmse(y_predict)
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nKNN Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)          



            # Decision Tree Regression Analysis
            analysis = DTRRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            tuned_alpha = False
            response = analysis.build_model(tuned_alpha)
            if response['success']:
                y_predict = analysis.predict_y_from_x_test()
                train_r2_score = analysis.get_train_score()
                test_r2_score = analysis.get_test_score()
                rmse = analysis.get_rmse(y_predict)
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nDecision Tree Regression Analysis: \ntrain_R2 = ' + str(train_r2_score) + '\ntest_R2 = ' + str(test_r2_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)          


            # Keras Regressor
            analysis = NNRegressor(self.x_train, self.x_test, self.y_train, self.y_test)
            opt = app_constants.NNR_DEFAULT_OPT
            loss = app_constants.NNR_DEFAULT_LOSS
            act_fun = app_constants.NNR_DEFAULT_FUN
            init_fun = app_constants.NNR_DEFAULT_INIT
            epo = app_constants.NNR_DEFAULT_EPO
            batch = app_constants.NNR_DEFAULT_BATCH
            nn = app_constants.NNR_DEFAULT_NN

            response = analysis.build_model(opt, loss, act_fun, init_fun, epo, batch, nn)
            if response['success']:
                mse = analysis.get_score_mse_test()
                rmse = analysis.get_score_rmse_test()
                mae = analysis.get_score_mae_test()
                test_score = analysis.get_test_score()
                train_score = analysis.get_train_score()
                text = text + '\n................\n\nKeras Regressor (Regression) Analysis: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(test_score) + '\ntest_RMSE = ' + str(rmse) + '\ntest_MSE = ' + str(mse) + '\ntest_MAE = ' + str(mae)           

            # Random Forrest Regressor
            analysis = RFRegression(self.x_train, self.x_test, self.y_train, self.y_test)
            no_of_estimators = app_constants.RFR_DEFAULT_NO_OF_ESTIMATORS
            criterion = app_constants.RFR_DEFAULT_MEASUREMENT_CRITERION
            max_depth = app_constants.RFR_DEFAULT_MAX_DEPTH
            min_samples_split = app_constants.RFR_DEFAULT_MIN_SAMPLES_SPLIT
            min_samples_leaf = app_constants.RFR_DEFAULT_MIN_SAMPLES_LEAF
            min_weight_fraction_leaf = app_constants.RFR_DEFAULT_MIN_WEIGHT_FRACTION_LEAF
            response = analysis.build_model(no_of_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                                            min_weight_fraction_leaf)
            if response['success']:
                train_score = analysis.get_train_score()
                test_score = analysis.get_test_score()
                test_rmse = analysis.get_score_rmse_test()
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nRandom Forest Regression: \ntrain_R2 = ' + str(train_score) + '\ntest_R2 = ' + str(test_score) + '\ntest_RMSE = ' + str(test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)              

            # Gradient Boosting Regression
            analysis = GradientBoostingReg(self.x_train, self.x_test, self.y_train, self.y_test)
            loss = app_constants.GBR_DEFAULT_LOSS
            learning_rate = app_constants.GBR_DEFAULT_LEARNING_RATE
            sub_sample = app_constants.GBR_DEFAULT_SUBSAMPLE
            no_of_estimators = app_constants.GBR_DEFAULT_NO_OF_ESTIMATORS
            criterion = app_constants.GBR_DEFAULT_MEASUREMENT_CRITERION
            max_depth = app_constants.GBR_DEFAULT_MAX_DEPTH
            min_samples_split = app_constants.GBR_DEFAULT_MIN_SAMPLES_SPLIT
            min_samples_leaf = app_constants.GBR_DEFAULT_MIN_SAMPLES_LEAF
            min_weight_fraction_leaf = app_constants.GBR_DEFAULT_MIN_WEIGHT_FRACTION_LEAF
            response = analysis.build_model(loss, learning_rate, sub_sample, no_of_estimators, criterion, max_depth,
                                            min_samples_split, min_samples_leaf, min_weight_fraction_leaf)
            if response['success']:
                train_score = analysis.get_train_score()
                test_score = analysis.get_test_score()
                test_rmse = analysis.get_score_rmse_test()
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                text = text + '\n................\n\nGradient Boosting Regression: \nScore = ' + str(
                    train_score) + '\ntest_R2 = ' + str(test_score) + '\ntest_RMSE = ' + str(
                        test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(test_mae)              

            # Bagging Regressor
            analysis = BaggingReg(self.x_train, self.x_test, self.y_train, self.y_test)
            no_of_estimators = app_constants.BR_DEFAULT_NO_OF_ESTIMATORS
            max_features = app_constants.BR_DEFAULT_MAX_FEATURES
            max_samples = app_constants.BR_DEFAULT_MAX_SAMPLES
            random_state = app_constants.DEFAULT_RANDOM_STATE
            oob_score = app_constants.BR_DEFAULT_OOB_SCORE
            response = analysis.build_model(no_of_estimators, max_features, max_samples, random_state, oob_score)
            if response['success']:
                train_score = analysis.get_train_score()
                test_score = analysis.get_test_score()
                test_rmse = analysis.get_score_rmse_test()
                test_mse = analysis.get_score_mse_test()
                test_mae = analysis.get_score_mae_test()
                oob_output_score = analysis.get_oob_score()
                text = text + '\n................\n\nBagging Regression: \nScore = ' + str(
                    train_score) + '\ntest_R2 = ' + str(test_score) + '\ntest_RMSE = ' + str(
                        test_rmse) + '\ntest_MSE = ' + str(test_mse) + '\ntest_MAE = ' + str(
                            test_mae)  + '\nOOB Score = ' + str(oob_output_score)

            self.result_text = Text(self.center_frame, width=100, height=18)
            self.result_text.insert(INSERT, text)
            self.result_text.config(state=DISABLED)
            self.result_text.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

            # create a Scrollbar and associate it with txt
            scrollb = Scrollbar(self.center_frame, command=self.result_text.yview)
            scrollb.grid(row=0, column=1, sticky='nsew')
            self.result_text['yscrollcommand'] = scrollb.set

        # Compare Classification Algorithms
        elif self.algorithm_name.get() == 'Compare Classification Algorithms':
            # Logistic Regression Classification
            text = ''
            analysis = LrClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                score = analysis.get_test_score()
                text = text + 'Logistic Regression Classification: \nScore = ' + str(score)

            # K-Nearest Neighbours Classification
            analysis = KnnClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                score = analysis.get_test_score()
                text = text + '\n................\n\nK-Nearest Neighbours Classification: \nScore = ' + str(score)

            # State Vector Machine Classification
            analysis = SvmClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            score = analysis.get_test_score()
            text = 'State Vector Machine Classification: \nScore = ' + str(score)

            # Naive Bayes Classification
            analysis = NbClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                score = analysis.get_test_score()
                text = text + '\n................\n\nNaive Bayes Classification: \nScore = ' + str(score)

            # Decision Tree Classification
            analysis = DtClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                score = analysis.get_test_score()
                text = text + '\n................\n\nDecision Tree Classification: \nScore = ' + str(score)

            # Keras Neural Network Classifier
            analysis = NeuralClassifier(self.x_train, self.x_test, self.y_train, self.y_test)

            opt = app_constants.NNC_DEFAULT_OPT
            fun = app_constants.NNC_DEFAULT_FUN
            init = app_constants.NNC_DEFAULT_INIT
            epo = app_constants.NNC_DEFAULT_EPO
            batch = app_constants.NNC_DEFAULT_BATCH
            nn = app_constants.NNC_DEFAULT_NN

            response =analysis.build_model(opt, fun, init, epo, batch, nn)
            if response['success']:
                test_score = analysis.get_test_score()
                text = text + '\n................\n\nAI | Keras Neural Network Classifier: \nTest Score = ' + str(
                    test_score) + '\n................\n'

            # Random Forest Classification
            analysis = RfClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            response = analysis.build_model()
            if response['success']:
                score = analysis.get_test_score()
                text = text + '\n................\n\nRandom Forest Classification: \nScore = ' + str(score)

            # AI | Random Forest Classifier
            analysis = AiRFClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            n_estimator = app_constants.RFC_DEFAULT_N_ESTIMATORS
            criterion = app_constants.RFC_DEFAULT_CRITERION
            min_sample_split = app_constants.RFC_DEFAULT_MIN_SAMPLES_SPLIT
            max_depth = app_constants.RFC_DEFAULT_MAX_DEPTH
            max_feature = app_constants.RFC_DEFAULT_MAX_FAETURES
            random_state = app_constants.DEFAULT_RANDOM_STATE
            response = analysis.build_model(n_estimator, criterion, min_sample_split, max_depth, max_feature,
                                            random_state)
            if response['success']:
                score = analysis.get_test_score()
                text = text + '\n................\n\nAI | Random Forest Classifier: \nScore = ' + str(score)

            # Gradient Boosting Classifier
            analysis = GradientBoostingClassification(self.x_train, self.x_test, self.y_train, self.y_test)
            learning_rate = app_constants.GB_DEFAULT_LEARNING_RATE
            n_estimator = app_constants.GB_DEFAULT_N_ESTIMATOR
            subsample = app_constants.GB_DEFAULT_SUB_SAMPLE
            criterion = app_constants.GB_DEFAULT_CRITERIION
            min_sample_split = app_constants.GB_DEFAULT_MIN_SAMPLES_SPLIT
            max_depth = app_constants.GB_DEFAULT_MAX_DEPTH
            max_feature = app_constants.GB_DEFAULT_MAX_FAETURES
            init = app_constants.GB_DEFAULT_INIT
            random_state = app_constants.DEFAULT_RANDOM_STATE
            max_leaf_nodes = app_constants.GB_DEFAULT_MAX_LEAF_NODE
            response = analysis.build_model(learning_rate, n_estimator, subsample, criterion,
                                            min_sample_split, max_depth, max_feature, init, random_state,
                                            max_leaf_nodes)
            if response['success']:
                test_score = analysis.get_test_score()
                text = text + '\n................\n\nAI | Gradient Boosting Classifier: \nTest Score = ' + str(
                    test_score)

            # Bagging Classifier
            analysis = BgnClassifier(self.x_train, self.x_test, self.y_train, self.y_test)
            max_samples = app_constants.BC_DEFAULT_MAX_SAMPLE
            n_estimator = app_constants.BC_DEFAULT_N_ESTIMATOR
            max_feature = app_constants.BC_DEFAULT_MAX_FAETURE
            random_state = app_constants.DEFAULT_RANDOM_STATE
            response = analysis.build_model(random_state, n_estimator, max_feature, max_samples)
            if response['success']:
                test_score = analysis.get_test_score()
                text = text + '\n................\n\nAI | Bagging Classifier: \nTest Score = ' + str(
                    test_score)

            self.result_text = Text(self.center_frame, width=100, height=18)
            self.result_text.insert(INSERT, text)
            self.result_text.config(state=DISABLED)
            self.result_text.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

            # create a Scrollbar and associate it with txt
            scrollb = Scrollbar(self.center_frame, command=self.result_text.yview)
            scrollb.grid(row=0, column=1, sticky='nsew')
            self.result_text['yscrollcommand'] = scrollb.set

    def view(self):
        """
        This function is to view the data frame
        :return:
        """
        self.w = TextScrollCombo(self.master,
                                 self._df.to_string(header=True, max_rows=None, min_rows=None, max_cols=None))
        self.master.wait_window(self.w.top)

    def export_result(self):
        """
        This function is to export data frame
        :return:
        """
        try:
            export_file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                            initialfile='result data_frame',
                                                            title='Save (result) data frame as')
            if export_file_path:
                df = pd.DataFrame(list(zip(self.y_test, self.y_predict_from_test)),
                                  columns=['Y Test', 'Y Predict'])
                df.to_csv(export_file_path, index=False, header=True)
        except:
            messagebox.showerror('Error', 'Error exporting data frame')

    def view_result(self):
        """
        This function is to view the result
        :return:
        """
        df = pd.DataFrame(list(zip(self.y_test, self.y_predict_from_test)),
                          columns=['Y Test', 'Y Predict'])
        self.w = TextScrollCombo(self.master, df.to_string(header=True, max_rows=None, min_rows=None, max_cols=None))
        self.master.wait_window(self.w.top)

    def view_multiple_plots(self):
        """
        This function is to view other plots
        :return:
        """
        if self.plot_list is None or len(self.plot_list) == 0:
            return False

        self.w = MultiplePlotBox(self.master, self.plot_list)
        self.master.wait_window(self.w.top)

    def predict(self):
        """
        This function is to to make predictions
        :return:
        """
        self.w = PredictBox(self.master, self.x_columns, self.y_column, self.analysis)
        self.master.wait_window(self.w.top)

    def zoom_image(self, image):
        print(image)
        self.w = ZoomBox(self.master, image)
        self.master.wait_window(self.w.top)
