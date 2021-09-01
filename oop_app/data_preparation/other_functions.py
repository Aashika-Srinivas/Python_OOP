import numpy as np
from response import Response
from sklearn.model_selection import train_test_split

'''
This class is used to perform some data filtering function such as:
1. Removing missing values
2. Replacing missing values with another value.
3. Removing/replacing values above or below a threshod.

Also this class is equally used to split the data set into training and test data.
All functions are class methods.
'''


class CleanData:

    # This function is used to split into training and test data set.
    @classmethod
    def split_data(cls, x, y, train_percent, shuffle_split=False, seed=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_percent / 100, shuffle=shuffle_split,
                                                            random_state=seed)

        result = dict()
        result['x_train'] = x_train
        result['x_test'] = x_test
        result['y_train'] = y_train
        result['y_test'] = y_test
        return Response.success(result)

    # This functions remove missing values.
    @classmethod
    def remove_missing_value(cls, data_frame, columns=None):
        if columns is None:
            return Response.success(data_frame.dropna())
        if type(columns) == str:
            columns = [columns]
        return Response.success(data_frame.dropna(how='any', subset=columns))

    # This function is used to replace values missing values with another value
    @classmethod
    def replace_missing_value(cls, data_frame, value, columns=None):
        if columns is None:
            return Response.success(data_frame.fillna(value=value))
        if type(columns) == str:
            columns = [columns]
        data_frame[columns] = data_frame[columns].fillna(value=value)
        return Response.success(data_frame)

    # This function is used to replace all valuse greater than threshold with another value
    @classmethod
    def replace_value_above(cls, data_frame, column, base, value):
        np_array = np.array(data_frame[column].values.tolist())
        data_frame[column] = np.where(np_array > base, value, np_array).tolist()
        return Response.success(data_frame)

    # This function is used to replace all valuse less than threshold with another value
    @classmethod
    def replace_value_below(cls, data_frame, column, base, value):
        np_array = np.array(data_frame[column].values.tolist())
        data_frame[column] = np.where(np_array < base, value, np_array).tolist()
        return Response.success(data_frame)

    # This functions remove all values less than a threshod.
    @classmethod
    def remove_value_below(cls, data_frame, column, base):
        data_frame = data_frame[data_frame[column] >= base]
        return Response.success(data_frame)

    # This functions remove all values greater than a threshod.
    @classmethod
    def remove_value_above(cls, data_frame, column, base):
        data_frame = data_frame[data_frame[column] <= base]
        return Response.success(data_frame)
