from tkinter import *
from tkinter import messagebox, simpledialog, filedialog

import numpy as np

import app_constants
from data_preparation.interpolation1d import Interp1D
from data_preparation.interpolation_linear import InterpLinear
from data_preparation.other_functions import CleanData
from data_preparation.remove_outlier import OutlierRemoval
from data_preparation.smoothen import Smoothing
from gui_components.widgets.table_view import TableView
from gui_components.dialogs.option_dialog import OptionDialog
from gui_components.dialogs.text_scrollbar import TextScrollCombo
from gui_components.dialogs.visualize_setup import VisualizeSetupBox

"""
This class provides view for data cleaning
"""


class CleanDataFrame(Frame):
    def __init__(self, parent, data_frame, master=None):
        super().__init__(master)

        self.parent = parent

        self._df = data_frame.copy()
        self.original_data_frame = data_frame.copy()

        # get non-numerical dataframe columns
        self.numerical_columns = []
        for col in list(data_frame.columns.values):
            data_type = data_frame.dtypes[col]
            if data_type == np.int64 or data_type == np.float64:
                self.numerical_columns.append(col)

        self._clean_df = self._df.copy()

        self.main_frame = Frame(self, pady=3)
        self.main_frame.grid(row=0, sticky='ew')

        self.top_frame1 = Frame(self.main_frame, pady=3)
        self.top_frame1.grid(row=0)

        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)

        self.clean_action_list = []

        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        title_label = Label(self.top_frame1, text='Data Preparation/Cleaning',
                            fg='blue', font=('Arial', 16))
        column_label = Label(self.top_frame1, text='Select Column:', bg='white')

        self.column_name = StringVar(self)

        col_options = ['All possible columns']
        for col in list(self.numerical_columns):
            col_options.append(col)
        self.column_name.set(col_options[0])
        columns_option_menu = OptionMenu(self.top_frame1, self.column_name, *col_options)

        action_label = Label(self.top_frame1, text='Select how to clean:', bg='white')

        self.action_name = StringVar(self)

        options = ['Remove outlier using IQR', 'Remove outlier using Z-Score',
                   'Replace missing values with 1d interpolation',
                   'Replace missing values with linear interpolation', 'Replace outliers with interpolation',
                   'Smooth with simple moving average', 'Smooth with cumulative moving average',
                   'Smooth with exponential moving average', 'Remove missing value', 'Replace missing value',
                   'Remove values above', 'Remove values below', 'Replace values above', 'Replace values below']

        self.action_name.set(options[0])
        action_option_menu = OptionMenu(self.top_frame1, self.action_name, *options)

        add_button = Button(self.top_frame1, text='Add', background='White',
                            command=self.add)
        preview_button = Button(self.top_frame1, text='Preview', background='White',
                                command=self.preview)

        reset_button = Button(self.top_frame1, text='Reset', background='White', command=self.reset)
        visualize_button = Button(self.top_frame1, background='blue', fg='white', relief='raised',
                                  text='Visualize',
                                  command=self.visualise)

        export_button = Button(self.top_frame1, background='red', fg='white', relief='raised',
                               text='Export',
                               command=self.export)
        view_button = Button(self.top_frame1, background='green', fg='white', relief='raised',
                             text='View DF',
                             command=self.view)

        next_button = Button(self.top_frame1, text='Next', background='White', fg='blue', command=self.next_step)

        self.use_default_option = StringVar(self)
        use_default_options = ['Use Defaults', 'Change Parameters']
        self.use_default_option.set(use_default_options[0])

        use_default_option_menu = OptionMenu(self.top_frame1, self.use_default_option, *use_default_options)

        title_label.grid(row=0, column=0, columnspan=12)
        column_label.grid(row=1, column=0, padx=(5, 2))
        columns_option_menu.grid(row=1, column=1, padx=(2, 5))
        action_label.grid(row=1, column=2, padx=(5, 2))
        action_option_menu.grid(row=1, column=3, padx=(2, 5))
        use_default_option_menu.grid(row=1, column=4, padx=(5, 2))
        add_button.grid(row=1, column=5, padx=(5, 5))
        reset_button.grid(row=1, column=6, padx=(5, 5))
        preview_button.grid(row=2, column=1, padx=(5, 5))
        visualize_button.grid(row=2, column=2, padx=(5, 5))
        export_button.grid(row=2, column=3, padx=(5, 5))
        view_button.grid(row=2, column=4, padx=(5, 5))
        next_button.grid(row=2, column=5, padx=(5, 20))

    def add(self):
        """
        This function is used to add a cleaning action/function to the cleaning list
        :return:
        """
        column = self.column_name.get()
        action = self.action_name.get()

        for var in self.clean_action_list:
            if action == var['action'] and (column in var['column'] or var['column'] == 'All possible columns'):
                messagebox.showerror('Error', 'Selected action already exist')
                return False

        if action == 'Replace missing value':
            value = simpledialog.askfloat('Input', 'Enter value to replace with',
                                          parent=self.parent)
            if value is None:
                return False
            var = dict()
            var['column'] = column
            var['action'] = action
            var['value'] = value
            self.clean_action_list.append(var)

        if action == 'Remove missing value':
            var = dict()
            var['column'] = column
            var['action'] = action
            self.clean_action_list.append(var)

        if action == 'Remove values above' or action == 'Remove values below':
            threshold = simpledialog.askfloat('Input', 'Enter threshold',
                                              parent=self.parent)
            if threshold is None:
                return False
            var = dict()
            var['column'] = column
            var['action'] = action
            var['threshold'] = threshold
            self.clean_action_list.append(var)

        if action == 'Replace values above' or action == 'Replace values below':

            threshold = simpledialog.askfloat('Input', 'Enter threshold',
                                              parent=self.parent)
            if threshold is None:
                return False
            value = simpledialog.askfloat('Input', 'Enter value to replace with',
                                          parent=self.parent)
            var = dict()
            var['column'] = column
            var['action'] = action
            var['threshold'] = threshold
            var['value'] = value
            self.clean_action_list.append(var)

        if action == 'Remove outlier using IQR':
            if len(self.numerical_columns) != len(self._clean_df.columns):
                messagebox.showerror('Error', 'This action cannot be performed on this data frame')
                return False

            var = dict()
            var['column'] = 'All possible columns'
            var['action'] = action
            self.clean_action_list.append(var)

        if action == 'Remove outlier using Z-Score':
            if len(self.numerical_columns) != len(self._clean_df.columns):
                messagebox.showerror('Error', 'This action cannot be performed on this data frame')
                return False
            if self.use_default_option.get() == 'Use Defaults':
                threshold = app_constants.OR_THRESHOLD
            else:
                threshold = simpledialog.askfloat('Input',
                                                  'Enter threshold for calculating z index, recommended value is 2',
                                                  parent=self.parent)
                if threshold is None:
                    return False

            var = dict()
            var['column'] = 'All possible columns'
            var['action'] = action
            var['threshold'] = threshold
            self.clean_action_list.append(var)

        if action == 'Smooth with simple moving average':

            if self.use_default_option.get() == 'Use Defaults':
                number_of_decimals = app_constants.SMA_DEFASULT_NUMBER_OF_DECIMALS
                moving_average_range = app_constants.SMA_DEFAULT_MOVING_AVERAGE_RANGE
            else:
                number_of_decimals = simpledialog.askinteger('Input', 'Enter number of decimals',
                                                             parent=self.parent)
                if number_of_decimals is None:
                    return False

                moving_average_range = simpledialog.askinteger('Input', 'Enter moving average Range',
                                                               parent=self.parent)
                if moving_average_range is None:
                    return False

            var = dict()
            var['column'] = column
            var['action'] = action
            var['number_of_decimals'] = number_of_decimals
            var['moving_average_range'] = moving_average_range
            self.clean_action_list.append(var)

        if action == 'Smooth with cumulative moving average':

            if self.use_default_option.get() == 'Use Defaults':
                number_of_decimals = app_constants.CMA_DEFAULT_NUMBER_OF_DECIMALS
            else:
                number_of_decimals = simpledialog.askinteger('Input', 'Enter number of decimals',
                                                             parent=self.parent)
                if number_of_decimals is None:
                    return False

            var = dict()
            var['column'] = column
            var['action'] = action
            var['number_of_decimals'] = number_of_decimals
            self.clean_action_list.append(var)

        if action == 'Smooth with exponential moving average':

            if self.use_default_option.get() == 'Use Defaults':
                number_of_decimals = app_constants.EMA_DEFASULT_NUMBER_OF_DECIMALS
                smoothing_factor_alpha = app_constants.EMA_DEFAULT_ALPHA
            else:
                number_of_decimals = simpledialog.askinteger('Input', 'Enter number of decimals',
                                                             parent=self.parent)
                if number_of_decimals is None:
                    return False

                smoothing_factor_alpha = simpledialog.askfloat('Input', 'Enter smoothing factor alpha (0-1)',
                                                               parent=self.parent)
                if smoothing_factor_alpha is None:
                    return False

            var = dict()
            var['column'] = column
            var['action'] = action
            var['number_of_decimals'] = number_of_decimals
            var['smoothing_factor_alpha'] = smoothing_factor_alpha
            self.clean_action_list.append(var)

        if action == 'Replace missing values with linear interpolation':
            if len(self.numerical_columns) != len(self._clean_df.columns):
                messagebox.showerror('Error', 'This action cannot be performed on this data frame')
                return False

            if self.use_default_option.get() == 'Use Defaults':
                auto = app_constants.I2D_DEFAULT_AUTO
                argument_column_initial = app_constants.I2D_DEFAULT_ARGUMENT_COLUMN_INITIAL
                list_remove_column = app_constants.I2D_DEFAULT_LIST_REMOVE_COLUMN
            else:
                auto = simpledialog.askinteger('Input', 'Enter 1 to set auto true or 0 for false',
                                               parent=self.parent)
                if auto is None or auto not in [0, 1]:
                    return False
                auto = auto == 1

                text = 'Enter argument initial column'
                for index, col in enumerate(list(self._df.columns.values)):
                    text = text + '\n{} for {},'.format(index, col)

                argument_column_initial = simpledialog.askinteger('Input', text,
                                                                  parent=self.parent)
                if argument_column_initial is None:
                    return False

                text = text.replace('Enter argument initial column', 'Enter list remove columns separated by comma(,)')

                list_remove_column = simpledialog.askstring('Input', text,
                                                            parent=self.parent)
                if list_remove_column is None or len(list_remove_column.strip()) == 0:
                    list_remove_column = []
                else:
                    list_remove_column = [int(s) for s in list_remove_column.strip().split(',')]

                if len(list_remove_column) > len(list(self._df.columns.values)):
                    return False

            var = dict()
            var['column'] = 'All possible columns'
            var['action'] = action
            var['auto'] = auto
            var['argument_column_initial'] = argument_column_initial
            var['list_remove_column'] = list_remove_column
            print(var)
            self.clean_action_list.append(var)

        if action == 'Replace missing values with 1d interpolation':

            dlg = OptionDialog(self.parent, 'Select One', 'Select the independent column', self._df.columns.values)

            if dlg.result is None:
                return False
            independent_col = dlg.result

            if self.use_default_option.get() == 'Use Defaults':
                method = app_constants.I1D_DEFAULT_METHOD
            else:
                text = 'Enter interpolation method\n 1 for linear, \n2 for nearestm' \
                       '\n3 for zero,\n4 for slinear,\n5 for quandratic,\n6 for cubic'

                method = simpledialog.askinteger('Input', text,
                                                 parent=self.parent)
                if method is None or method not in [1, 2, 3, 4, 5]:
                    return False

            var = dict()
            var['column'] = 'All possible columns'
            var['action'] = action
            var['independent_col'] = independent_col
            var['method'] = method
            self.clean_action_list.append(var)

        if action == 'Replace outliers with interpolation':

            dlg = OptionDialog(self.parent, 'Select One', 'Select the independent column', self._df.columns.values)

            if dlg.result is None:
                return False

            independent_col = dlg.result

            if self.use_default_option.get() == 'Use Defaults':
                method = app_constants.I1D_DEFAULT_METHOD
            else:
                text = 'Enter interpolation method\n1 for linear,\n2 for nearestm' \
                       '\n3 for zero,\n4 for slinear,\n5 for quandratic,\n6 for cubic'

                method = simpledialog.askinteger('Input', text,
                                                 parent=self.parent)
                if method is None or method not in [1, 2, 3, 4, 5]:
                    return

            var = dict()
            var['column'] = 'All possible columns'
            var['action'] = action
            var['independent_col'] = independent_col
            var['method'] = method
            self.clean_action_list.append(var)

        self.clean()
        self.preview()

    def reset(self):
        """
        This function is used to reset the data frame back to the original data frame
        :return:
        """
        self.clean_action_list = []
        self._clean_df = self._df
        self.center_frame.destroy()
        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)

    def preview(self):
        """
        This function is used to preview the data frame
        :return:
        """
        self.center_frame.destroy()
        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)

        actions = self.clean_action_list

        cols = ['Column', 'Action']
        rows = []

        for action in actions:
            rows.append((action['column'], action['action']))

        preview_frame = Frame(self.center_frame)
        TableView(preview_frame, cols, rows)

        preview_label = Label(self.center_frame, text=' ')
        describe_label = Label(self.center_frame, text='Cleaned Data Description')

        describe_df = self._clean_df.describe(include='all')
        describe_df.loc['data_type'] = list(describe_df.dtypes)
        describe_df.insert(0, ' ', describe_df.index)
        df_col = list(describe_df.columns.values)
        df_row = list(describe_df.values)
        describe_frame = Frame(self.center_frame)
        TableView(describe_frame, df_col, df_row)

        preview_label.grid(row=0, column=0, padx=(5, 5))
        describe_label.grid(row=0, column=2, padx=(5, 5))
        preview_frame.grid(row=1, column=0, padx=(1, 1), columnspan=2)
        describe_frame.grid(row=1, column=2, padx=(1, 1), columnspan=2)

    def clean(self):
        """
        This function is called for the actual cleaning to be performed based
        on the added actions
        :return:
        """
        data_frame = self._df.copy()
        for var in self.clean_action_list:
            if var['action'] == 'Replace missing value':
                if var['column'] == 'All possible columns':
                    response = CleanData.replace_missing_value(data_frame, var['value'])
                else:
                    response = CleanData.replace_missing_value(data_frame, var['value'], var['column'])

                if response['success']:
                    data_frame = response['data']
                else:
                    messagebox.showerror('Error', response['message'])
                    return False

            if var['action'] == 'Remove missing value':
                if var['column'] == 'All possible columns':
                    response = CleanData.remove_missing_value(data_frame)
                else:
                    response = CleanData.remove_missing_value(data_frame, var['column'])
                if response['success']:
                    data_frame = response['data']
                else:
                    messagebox.showerror('Error', response['message'])
                    return False

            if var['action'] == 'Remove values above':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = CleanData.remove_value_above(data_frame, col, var['threshold'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = CleanData.remove_value_above(data_frame, var['column'], var['threshold'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Remove values below':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = CleanData.remove_value_below(data_frame, col, var['threshold'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = CleanData.remove_value_below(data_frame, var['column'], var['threshold'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Replace values above':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = CleanData.replace_value_above(data_frame, col, var['threshold'], var['value'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = CleanData.replace_value_above(data_frame, var['column'], var['threshold'], var['value'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Replace values below':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = CleanData.replace_value_below(data_frame, col, var['threshold'], var['value'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = CleanData.replace_value_below(data_frame, var['column'], var['threshold'], var['value'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Replace missing values with linear interpolation':
                response = InterpLinear.replace_missing_values(data_frame, var['auto'], var['argument_column_initial'],
                                                               var['list_remove_column'])
                if response['success']:
                    data_frame = response['data']
                else:
                    messagebox.showerror('Error', response['message'])
                    return False

            if var['action'] == 'Replace outliers with interpolation':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Interp1D.intrp_outlier(data_frame, col, var['independent_col'], var['method'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = Interp1D.intrp_outlier(data_frame, var['column'], var['independent_col'], var['method'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Replace missing values with 1d interpolation':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Interp1D.intrp_missing(data_frame, col, var['independent_col'], var['method'])
                        if response['success']:
                            data_frame = response['data']
                else:
                    response = Interp1D.intrp_missing(data_frame, var['column'], var['independent_col'], var['method'])
                    if response['success']:
                        data_frame = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Remove outlier using IQR':
                response = OutlierRemoval.remove_outlier_iqr(data_frame)
                if response['success']:
                    data_frame = response['data']
                else:
                    messagebox.showerror('Error', response['message'])
                    return False
            if var['action'] == 'Remove outlier using Z-Score':
                response = OutlierRemoval.remove_outlier_zscore(data_frame, var['threshold'])
                if response['success']:
                    data_frame = response['data']
                else:
                    messagebox.showerror('Error', response['message'])
                    return False

            if var['action'] == 'Smooth with simple moving average':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Smoothing.simple_moving_average(list(data_frame[col]), var['number_of_decimals'],
                                                                   var['moving_average_range'])
                        if response['success']:
                            data_frame[col] = response['data']
                else:
                    response = Smoothing.simple_moving_average(list(data_frame[var['column']]),
                                                               var['number_of_decimals'], var['moving_average_range'])
                    if response['success']:
                        data_frame['column'] = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Smooth with cumulative moving average':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Smoothing.cumulative_moving_average(list(data_frame[col]), var['number_of_decimals'])
                        if response['success']:
                            data_frame[col] = response['data']
                else:
                    response = Smoothing.cumulative_moving_average(list(data_frame[var['column']]),
                                                                   var['number_of_decimals'])
                    if response['success']:
                        data_frame['column'] = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Smooth with cumulative moving average':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Smoothing.cumulative_moving_average(list(data_frame[col]), var['number_of_decimals'])
                        if response['success']:
                            data_frame[col] = response['data']
                else:
                    response = Smoothing.cumulative_moving_average(list(data_frame[var['column']]),
                                                                   var['number_of_decimals'])
                    if response['success']:
                        data_frame['column'] = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

            if var['action'] == 'Smooth with exponential moving average':
                if var['column'] == 'All possible columns':
                    for col in list(self.numerical_columns):
                        response = Smoothing.exponential_moving_average(list(data_frame[col]), var['number_of_decimals'],
                                                                        var['smoothing_factor_alpha'])
                        if response['success']:
                            data_frame[col] = response['data']
                else:
                    response = Smoothing.exponential_moving_average(list(data_frame[var['column']]),
                                                                    var['number_of_decimals'],
                                                                    var['smoothing_factor_alpha'])
                    if response['success']:
                        data_frame['column'] = response['data']
                    else:
                        messagebox.showerror('Error', response['message'])
                        return False

        self._clean_df = data_frame

    def next_step(self):
        """
        This function is used to change the view
        If there was no cleaning the user is warned
        :return:
        """
        if len(self.clean_action_list) == 0:
            response = messagebox.askquestion('Proceed', 'Do you really want to proceed without cleaning data')
            if response != 'yes':
                return False

        self.parent.set_cleaned_data_frame(self._clean_df)
        self.parent.analyse()

    def visualise(self):
        """
        This function is used to visualize the cleaned data frame
        :return:
        """
        self.w = VisualizeSetupBox(self.master, self._clean_df)
        self.master.wait_window(self.w.top)

    def export(self):
        """
        This function is used to export the cleaned data frame as csv
        :return:
        """
        try:
            export_file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                            initialfile='data_frame', title='Save data frame as')
            if export_file_path:
                self._clean_df.to_csv(export_file_path, index=False, header=True)
        except:
            messagebox.showerror('Error', 'Error exporting data frame')

    def view(self):
        """
        This function is used to view the data frame
        :return:
        """
        self.w = TextScrollCombo(self.master,
                                 self._clean_df.to_string(header=True, max_rows=None, min_rows=None, max_cols=None))
        self.master.wait_window(self.w.top)
