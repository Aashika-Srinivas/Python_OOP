# Common unit for group 2 provided by Fernando Tovar
import numpy as np
import pandas as pd
from scipy import interpolate
import math
import datetime

from response import Response


class CheckData:
    def isDFNumeric(df):
        non_numeric = df[~df.applymap(np.isreal).all(1)]
        return non_numeric.empty  # Returns True if all data is numeric else False

    def isColumnNumeric(df, col):
        non_numeric = df[col][~df[col].apply(np.isreal)]
        return non_numeric.empty  # Returns True if all data is numeric else False  In [2]:  #OOP Programming


# Numeric-type datasets
# Group 2, Andrey Domnyshev
# Fill in missing values using linear interpolation for 2-dimensional data, where all data is taken into account
# 29/03/2021


# This function fills the missing entities on 'pandas.core.frame.DataFrame' object. 
# Each row must have at least two not empty cells (entities)
# User should specify which column should be considered as 'argument' values
# Column with missing value is considered as 'function' values
# Values of all data set are taken into account in order to find best input for interpolation
# Examples files and description are in report.

# ARGUMENTS
# data *frame: 'pandas.core.frame.DataFrame' object that can consist of NaN entities. 
# Each row must have at least two not empty cells (entities). 

# auto *boolean: if auto=False, user does not allow program to change argument column.
# If auto=True, program still uses user’s preference, but also provides interpolation using
# other columns as argument.

# argument_column_initial *integer: integer value, that indicates which column of frame
# should be used as argument column – column consists of 1-D array with “x” values used
# to approximate some function f: y = f(x). Every time when program find empty entity, program
# considers the column with empty entity as function column – column consists of 1-D
# array with “y” values.

# list_remove_column *list: list with columns numbers, that must not be considered 
# as argument column (1-D array with “x” values). For example, 1st 'No' column in 'Real estate'
# dataset that consist only of sequence, that does not have any sense for data analyzation.

# Method RETURNs new dataframe without missing values.

# User should be able to choose 'auto' and input 'argument_column_initial' and 'list_remove_column'.

# The GLOBAL VARIABLE 'message_from_interp_2D' consists of report of the last 'interp_2D' method calling.
class InterpLinear:
    @classmethod
    def replace_missing_values(cls, data, auto, argument_column_initial, list_remove_column):

        def digits(number):
            # this method is takren from: https://ru.stackoverflow.com/questions/658009/
            string = str(number)
            if '.' in string:
                return abs(string.find('.') - len(string)) - 1
            elif 'e' in string:
                return -1
            else:
                return 0

        global message_from_interp_2D

        now = datetime.datetime.now()
        message_from_interp_2D = 'This is the report from last Fill in missing values using linear interpolation'
        message_from_interp_2D = message_from_interp_2D + '\n---START--- \n' + now.strftime('%d-%m-%Y %H:%M:%S.%f')

        if CheckData.isDFNumeric(data) and (type(data) is pd.core.frame.DataFrame) and (
                type(auto) is bool) and (type(argument_column_initial) is int) and (type(list_remove_column) is list):

            matrix = data.to_numpy(dtype='float64')
            counter = 0

            # define max number of digits in matrix
            max_digits = -1
            for i in range(0, len(matrix), 1):
                for j in range(0, len(matrix[0]), 1):
                    if (digits(matrix[i][j]) > max_digits) and (digits(matrix[i][j]) != -1):
                        max_digits = digits(matrix[i][j])
            message_from_interp_2D = message_from_interp_2D + '\nMaximum decimal digits is ' + str(max_digits)

            # if auto is true, we assign loops in which we change argument_column
            list_argument_column = list(range(argument_column_initial, len(matrix[0]), 1)) + list(
                range(0, argument_column_initial, 1))

            # remove columns from list_argument_column that must be ignored
            for re in list_remove_column:
                list_argument_column.remove(re)

            # display message about chosen mode
            if not auto:
                message_from_interp_2D = message_from_interp_2D + '\nYou have chosen hand mode: auto==False, only ' \
                                                                  'column ' + str(
                    argument_column_initial) + ' will be considered as argument'
            else:
                message_from_interp_2D = message_from_interp_2D + '\nYou have chosen auto mode: auto==True, columns ' \
                                                                  'in order' + str(
                    list_argument_column) + ' will be considered as argument'

            # strt the main loop, we check if a belongs to list_argument_column, that means, that interpolation is
            # porovided usin argument column no. a
            for a in list_argument_column:
                if argument_column_initial > len(matrix[0]):
                    message_from_interp_2D = message_from_interp_2D + '\nWrong argument_column_initial value'
                    break
                if not auto and (a != argument_column_initial):
                    message_from_interp_2D = message_from_interp_2D + '\n-> Skip non-argument column no ' + str(a)
                    argument_column = argument_column_initial
                else:
                    argument_column = a
                    message_from_interp_2D = message_from_interp_2D + '\n-> Consider argument column no ' + str(a)

                    # find rows with missing
                    for i in range(0, len(matrix), 1):
                        for j in range(0, len(matrix[0]), 1):
                            if math.isnan(matrix[i][j]):

                                if j == argument_column:
                                    if not auto:
                                        message_from_interp_2D = message_from_interp_2D + '\nArgument column consists ' \
                                                                                          'of NaN element(s) and you ' \
                                                                                          'use hand mode (' \
                                                                                          'Auto==False). \nYour ' \
                                                                                          'output dataframe will ' \
                                                                                          'still consist of empty ' \
                                                                                          'entities '
                                    break

                                # column with missing is playing role of 'function' values
                                row_with_missing = i
                                function_column = j

                                # find quite similar rows, one where argument lower and one where argument higher
                                diference_array = np.array([], dtype='float64')
                                for m in range(0, len(matrix), 1):
                                    diference = 0.0
                                    element_counter = 0.0001  # avoid devision by 0

                                    for n in range(0, len(matrix[0]), 1):
                                        if (m != row_with_missing) and (n != function_column) and (
                                                n != argument_column) and (math.isnan(matrix[m][n]) == False) and (
                                                not (n in list_remove_column)):
                                            buffer2 = element_counter
                                            element_counter = buffer2 + 1
                                            buffer = diference
                                            diference = buffer + (
                                                abs(matrix[m][n] - matrix[row_with_missing][n])) / abs(
                                                matrix[row_with_missing][n])
                                    # we introduce counter in order to consider mean value of difference (in
                                    # different rows can be different number of elements) moreover, if there are all
                                    # nan elements in row, element_counter will be very large and it will not be used
                                    # in next step
                                    mean_diference = diference / element_counter
                                    diference_array = np.concatenate((diference_array, [mean_diference]), axis=0)

                                # find similar row, where argument lower

                                Row1_index = -1
                                Row2_index = -1
                                # first assign very big value
                                smallest = 999 * (
                                        np.amax(diference_array[np.logical_not(np.isnan(diference_array))]) + 1)
                                for k in range(0, len(diference_array)):
                                    if (diference_array[k] < smallest) and (
                                            matrix[k][argument_column] < matrix[row_with_missing][argument_column]):
                                        Row1_index = k
                                        smallest = diference_array[k]

                                # find similar row, where argument higher
                                smallest = 999 * (
                                        np.amax(diference_array[np.logical_not(np.isnan(diference_array))]) + 1)
                                for k in range(0, len(diference_array)):
                                    if (diference_array[k] < smallest) and (
                                            matrix[k][argument_column] > matrix[row_with_missing][argument_column]):
                                        Row2_index = k
                                        smallest = diference_array[k]

                                try:
                                    # make linear interpolation
                                    X_array = np.array(
                                        [matrix[Row1_index][argument_column], matrix[Row2_index][argument_column]],
                                        dtype='float64')
                                    Y_array = np.array(
                                        [matrix[Row1_index][function_column], matrix[Row2_index][function_column]],
                                        dtype='float64')

                                    fun = interpolate.interp1d(X_array, Y_array, kind='linear')
                                    value_found = fun(matrix[row_with_missing][argument_column])
                                    message_from_interp_2D = message_from_interp_2D + '\n-----> Put value ' + str(
                                        np.round(value_found, max_digits)) + ' in row ' + str(
                                        row_with_missing) + ' column ' + str(function_column)
                                    counter += 1

                                    if max_digits != -1:
                                        try:
                                            # Round
                                            # put this value into matrix
                                            matrix[row_with_missing][function_column] = np.round(value_found,
                                                                                                 max_digits)

                                        except:
                                            # in case of problems with round
                                            matrix[row_with_missing][function_column] = value_found
                                            message_from_interp_2D = message_from_interp_2D + '\n--------> Problem ' \
                                                                                              'with round ' + str(
                                                value_found) + ' in row ' + str(row_with_missing) + ' column ' + str(
                                                function_column)
                                    else:
                                        matrix[row_with_missing][function_column] = value_found


                                except:
                                    # in case of problem with interpolation
                                    message_from_interp_2D = message_from_interp_2D + '\n-----> Interpolation failed ' \
                                                                                      'in row ' + str(
                                        row_with_missing) + ' column ' + str(function_column)

            now2 = datetime.datetime.now()
            message_from_interp_2D = message_from_interp_2D + '\nTotal number of added values: ' + str(
                counter) + now2.strftime('\n%d-%m-%Y %H:%M:%S.%f') + '\n---FINISH--- '
            newdata = pd.DataFrame(matrix)
            newdata.columns = data.columns
            return Response.success(newdata)
        else:
            now2 = datetime.datetime.now()
            message_from_interp_2D = message_from_interp_2D + '\nFAILED\nInterpolation algorithm can not work with ' \
                                                              'such arguments type' + now2.strftime(
                '\n%d-%m-%Y %H:%M:%S.%f') + '\n---FINISH--- '
            return Response.failure(message_from_interp_2D)
