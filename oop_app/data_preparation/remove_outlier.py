# importing library

from response import Response


class OutlierRemoval:
    def __init__(self, dataframe, threshold):
        self.dataframe = dataframe
        self.threshold = threshold

    ####################################################################################################################
    # Method for outlier removal with the Inter Quartile Range IQR
    @classmethod
    def remove_outlier_iqr(cls, dataframe):
        for col in dataframe.columns:
            print('Removing the outliers ', col)
            if ((dataframe[col].dtype == 'float64') | (
                    dataframe[col].dtype == 'int64')):  # Check for the numerical columns
                # Defining quantile ranges
                Q1 = dataframe.quantile(0.25)
                Q3 = dataframe.quantile(0.75)
                # IQR score
                IQR = Q3 - Q1
                df_clean = dataframe[~((dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR))).any(axis=1)]

        else:
            df_clean[col] = dataframe[col]

        # Amount of outlier rows eliminated
        originalData = int(dataframe.shape[0])
        finalData = int(df_clean.shape[0])
        rowsEliminated = originalData - finalData

        print('Amount of Eliminated Rows: ', rowsEliminated)
        print('')

        # Returning the output as a dataframe
        return Response.success(df_clean)

    ####################################################################################################################
    # Method for outlier removal with the Z-score
    @classmethod
    def remove_outlier_zscore(cls, dataframe, threshold):
        # Original Data size
        data_col = dataframe.shape[1]
        data_row = dataframe.shape[0]

        # Z-Score calculation signed number of standard deviations by which the value of an observation or data point
        # is above the mean value Z-Score tells how far a point is from the mean of dataset in terms of standard
        # deviation

        cols = list(dataframe.columns)

        for col in cols:
            col_zscore = col + '_zscore'
            dataframe[col_zscore] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std(ddof=0)

        # Threashold for z-score is an empirical value. Might be changed according the dataset
        t = threshold
        # Selecting half of the dataset where Z columns are located (down half)
        dataSizeHalf_col = int(dataframe.shape[1] / 2)
        dataSize_col = int(dataframe.shape[1])
        # Removing Outliers where z < threashold and adding to dataNew
        dataNew = dataframe[(dataframe[dataframe.columns[dataSizeHalf_col:dataSize_col]] < t).all(axis=1)]
        # Removing Z Colums to final matrix (Upper half)
        data_final = dataNew.iloc[:, 0:dataSizeHalf_col]

        # Amount of outlier rows eliminated
        originalData = int(dataframe.shape[0])
        finalData = int(data_final.shape[0])
        rowsEliminated = originalData - finalData

        print('Amount of Eliminated Rows: ', rowsEliminated)
        print('')

        # Returning the output as a dataframe
        return Response.success(data_final)

        # ###################################################################################################################
    # End of the Outlier Removal Class
