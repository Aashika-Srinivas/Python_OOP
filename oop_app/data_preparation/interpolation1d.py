import numpy as np
import pandas as pd
from scipy.interpolate import interpolate

from response import Response


class Interp1D:

    # Class method to interpolate outliers
    @classmethod
    def intrp_outlier(cls, df, feature_col, ind_col, method_no=1):
        # ind_col # independent variable, the column that is used for interpolation
        # feature_col # column/variable that ouliers have to be corrected in
        # print('Choose a number of interpolation methods:', method_dict)
        method_dict = {1: 'linear', 2: 'nearest', 3: 'zero', 4: 'slinear', 5: 'quadratic', 6: 'cubic'}
        interp_method = method_dict[method_no]  # the interpolation method
        df_corrected = pd.DataFrame({'A': []})  # with values corrected once the interpolation function applied to it
        f = None  # the interpolation function
        outlier_col = None  # column of zero and ones indicating ouliers as 1 and inliers as 0
        df_no_outlier_no_missing = None  # dataframe with outliers and missing values removed
        df_only_outlier = None  # dataframe with outliers only

        # We now detect outliers and add a column with 0 if it is not outlier and 1 if it is outlier
        # First we get the quantiles at 0.25, 0.50 and 0.75, which are the median, lower and upper quartiles.
        dfMedian = df[feature_col].quantile(0.50)
        lower_quartile = df[feature_col].quantile(0.25)
        upper_quartile = df[feature_col].quantile(0.75)
        # Then we get the IQR by resting the upper quartile - lower quartile
        iqr = upper_quartile - lower_quartile
        # With the quartiles and IQR, we get now the upper and lower whisker
        upper_whisker = df[feature_col][df[feature_col] <= upper_quartile + 1.5 * iqr].max()
        lower_whisker = df[feature_col][df[feature_col] >= lower_quartile - 1.5 * iqr].min()
        # Finally we use the whiskers to add a new column with 0 if the row value is inside the whiskers and
        # 1 if is outside the whiskers
        outlier_col = np.where((df[feature_col] > upper_whisker) | (df[feature_col] < lower_whisker), 1, 0)
        df['outlier'] = outlier_col
        # We create two dataframes, one with no outliers and no missing values
        df_no_outlier_no_missing = df[outlier_col == 0].dropna()
        # And another with outliers only.
        df_only_outlier = df[outlier_col == 1]

        # We use the 1 dimensional interpolation method to interpolate either the missing values or the outliers
        # First we divide the process depending on if the interpolation method selected is quadratic/cubic or something else
        if interp_method in ['quadratic', 'cubic']:
            # We print a warning saying that the independent variable should be strictly ascending
            print('warning: for this interpolation algorithm, the independent variables should be stricktly ascending!')
            # We first sort the dataframe with all the inliers
            df_no_outlier_no_missing.sort_values(by=[ind_col], inplace=True)
            # Creates function, where outliers will then be interpolated
            # Parameters are (x,y,kind of interpolation)        
            f = interpolate.interp1d(np.array(df_no_outlier_no_missing[ind_col]),
                                     np.array(df_no_outlier_no_missing[feature_col]),
                                     kind=interp_method,
                                     bounds_error=False,
                                     # fill_value='extrapolate')
                                     fill_value=df_no_outlier_no_missing[feature_col].mean())
        else:
            # Creates function, where outliers will then be interpolated
            # Parameters are (x,y,kind of interpolation)
            f = interpolate.interp1d(np.array(df_no_outlier_no_missing[ind_col]),
                                     np.array(df_no_outlier_no_missing[feature_col]),
                                     kind=interp_method,
                                     bounds_error=False,
                                     # fill_value='extrapolate')
                                     fill_value=np.array(df_no_outlier_no_missing[feature_col]).mean())
        # Finally we use the function f to replace the outliers
        if df_corrected.empty:
            df_corrected = df.copy(deep=True)
        df_corrected[feature_col][outlier_col == 1] = np.array(f(df_only_outlier[ind_col]))

        df_corrected = df_corrected.drop(columns=['outlier'])

        return Response.success(df_corrected)

    #Class method to interpolate missing values
    @classmethod
    def intrp_missing(cls, df, feature_col, ind_col, method_no=1):
        ind_col = ind_col  # independent variable, the column that is used for interpolation
        feature_col = feature_col  # column/variable that ouliers have to be corrected in
        # print('Choose a number of interpolation methods:', method_dict)
        method_dict = {1: 'linear', 2: 'nearest', 3: 'zero', 4: 'slinear', 5: 'quadratic', 6: 'cubic'}
        interp_method = method_dict[method_no]  # the interpolation method
        df = df  # original dataframe
        df_corrected = pd.DataFrame({'A': []})  # with values corrected once the interpolation function applied to it
        f = None  # the interpolation function
        outlier_col = None  # column of zero and ones indicating outliers as 1 and inliers as 0
        df_no_outlier_no_missing = None  # dataframe with outliers and missing values removed
        df_only_outlier = None  # dataframe with outliers only

        # We now detect outliers and add a column with 0 if it is not outlier and 1 if it is outlier
        # First we get the quantiles at 0.25, 0.50 and 0.75, which are the median, lower and upper quartiles.
        dfMedian = df[feature_col].quantile(0.50)
        lower_quartile = df[feature_col].quantile(0.25)
        upper_quartile = df[feature_col].quantile(0.75)
        # Then we get the IQR by resting the upper quartile - lower quartile
        iqr = upper_quartile - lower_quartile
        # With the quartiles and IQR, we get now the upper and lower whisker
        upper_whisker = df[feature_col][df[feature_col] <= upper_quartile + 1.5 * iqr].max()
        lower_whisker = df[feature_col][df[feature_col] >= lower_quartile - 1.5 * iqr].min()
        # Finally we use the whiskers to add a new column with 0 if the row value is inside the whiskers and
        # 1 if is outside the whiskers
        outlier_col = np.where((df[feature_col] > upper_whisker) | (df[feature_col] < lower_whisker), 1, 0)
        df['outlier'] = outlier_col
        # We create two dataframes, one with no outliers and no missing values
        df_no_outlier_no_missing = df[outlier_col == 0].dropna()
        # And another with outliers only.
        df_only_outlier = df[outlier_col == 1]

        # We use the 1 dimensional interpolation method to interpolate either the missing values or the outliers
        # First we divide the process depending on if the interpolation method selected is quadratic/cubic or something else
        if interp_method in ['quadratic', 'cubic']:
            # We print a warning saying that the independent variable should be strictly ascending
            print('warning: for this interpolation algorithm, the independent variables should be stricktly ascending!')
            # We first sort the dataframe with all the inliers
            df_no_outlier_no_missing.sort_values(by=[ind_col], inplace=True)
            # Creates function, where missing values will then be interpolated
            # Parameters are (x,y,kind of interpolation)        
            f = interpolate.interp1d(np.array(df_no_outlier_no_missing[ind_col]),
                                     np.array(df_no_outlier_no_missing[feature_col]),
                                     kind=interp_method,
                                     bounds_error=False,
                                     # fill_value='extrapolate')
                                     fill_value=df_no_outlier_no_missing[feature_col].mean())
        else:
            # Creates function, where missing values will then be interpolated
            # Parameters are (x,y,kind of interpolation)
            f = interpolate.interp1d(np.array(df_no_outlier_no_missing[ind_col]),
                                     np.array(df_no_outlier_no_missing[feature_col]),
                                     kind=interp_method,
                                     bounds_error=False,
                                     # fill_value='extrapolate')
                                     fill_value=np.array(df_no_outlier_no_missing[feature_col]).mean())

        # Finally we use the function f to replace the missing values    
        if df_corrected.empty:
            df_corrected = df.copy(deep=True)
        df_corrected[feature_col][df[feature_col].isna()] = np.array(f(df[df[feature_col].isna()][ind_col]))

        df_corrected = df_corrected.drop(columns=['outlier'])
        return Response.success(df_corrected)
