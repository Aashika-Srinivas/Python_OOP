#The smoothing algorithm consists of a class name Smoothing. The following class has three methods implemented which apply different way to smooth data points. 
#The three methods are the simple moving average, the cumulative moving average, and the exponential moving average. 
# The simple moving average has a loop with five possible smoothing cases which smooth the data point of interest based 
#on certain condition(s). 
   # Case 1. Smoothing the first data point by considering only the next data point
      #Three conditions need to be fulfilled: 
           # 1. The data point of interest is the first data point --> i==0
           # 2. The moving average range is larger than zero --> MovingAverageRange>0.
           # 3. The length of the list containing data is larger than 1 --> len(RawData)>1.
   # Case 2. Smoothing when moving average range > available data points at the left side of the data point of interest
      #Scale down the moving average to the number of available data points at the left side of that data point
      #Two conditions need to be fulfilled: 
           #1. The available data points at the left side are smaller than the moving average range --> i-MovingAverageRange<0
           #2. The data point of interest is not the last data point --> i != len(RawData)-1
      #There are two possible cases: 
        #case 1: sufficient data points at  the right side of the data point of interest --> proceed with the scaled moving average
            #If available data points at the right side of the data point of interest is equal or larger 
            #than the available data points at the left side of the data point of interest --> (AvailableDataPoints_RightSide >= i)
        #Case 2: insufficient data points at  the right side of the data point of interest --> scale down the moving average further
           # If available data points at the right side of the data point of interest is smaller 
           #than the available data points at the left side of the data point of interest, then scale down the moving average range 
           #to the number of the data point available at the right side of the data point of interest 
   # Case 3. Smoothing the last data point by considering only the previous smoothed data point
      # Two conditions need to be fulfilled: 
         # 1. The data point of interest is the last data point --> i == len(RawData)-1
         # 2. The moving average range is larger than zero --> MovingAverageRange>0   
   # Case 4.Smoothing when moving average range is larger than the available data points at the right side of the data point of interest
      #Scale down the moving average to the number of available data points at the right side of that data point of interest  
   # Case 5: Smoothing considering data with a suitable value of moving average    

# The Cumulative moving average smooth the data point of interest by taking the sum of previous data points including 
#the data point of interest divided by the number of the data points considered in that sum

# The Exponential moving average Smoothing the data points based on the current data point and the previous data point.
# The SmoothingFactorAlpha decide on the weight of the data points, which data point has more impact (current? or the previous?)


from response import Response


class Smoothing:

    @classmethod  # Method 1: Simple moving average
    def simple_moving_average(cls, raw_data, number_of_decimals=2, moving_average_range=1):
        smoothed_data_ema = []  # initializing an empty list
        try:
            for i in range(0, len(raw_data)):

                # Case 1: Smoothing the first data point by considering only the next data point
                if i == 0 and moving_average_range > 0 and len(raw_data) > 1:
                    # Smoothing considering only the data point at the right side of the first data point
                    smoothed_data_ema.append(round(sum(raw_data[0:2]) / 2, number_of_decimals))

                    # Case 2: Smoothing when moving average range > available data points at the left side of the data
                    # point of interest 
                elif i - moving_average_range < 0 and i != len(raw_data) - 1:
                    # Variable indicating the number of avialable data points at the right side of the data point of 
                    # interest 
                    available_data_points_right_side = len(raw_data) - i - 1

                    # Sufficient data points at  the right side of the data point of interest --> proceed with scaled
                    # moving average 
                    if available_data_points_right_side >= i:
                        # Smoothing data point based on the avialable data points at the left side of the data point 
                        # of interest 
                        smoothed_data_ema.append(round(sum(raw_data[0:i * 2 + 1]) / (i * 2 + 1), number_of_decimals))

                        # Insufficient data points at  the right side of the data point of interest --> scale down 
                        # the moving average further 
                    else:
                        # The scaled down moving average based on the avialable data point at the right side of the 
                        # data point of interest 
                        scaled_down_moving_average = available_data_points_right_side
                        # Smoothing considering the scaled down moving average
                        smoothed_data_ema.append(round(
                            sum(raw_data[i - scaled_down_moving_average:i + scaled_down_moving_average + 1]) / (
                                        scaled_down_moving_average * 2 + 1), number_of_decimals))

                # Case 3: Smoothing the last data point by considering only the previous smoothed data point
                elif i == len(raw_data) - 1 and moving_average_range > 0:
                    # Smoothing considering only the data point at the left side of the last data point
                    smoothed_data_ema.append(round((raw_data[i] + smoothed_data_ema[i - 1]) / 2, number_of_decimals))

                # Case 4: Smoothing when moving average range > available data points at the right side of the data 
                # point of interest 
                elif i + moving_average_range > len(raw_data) - 1:
                    # The scaled down moving average based on the available data points at the right side of the data
                    # point of interest 
                    scaled_down_moving_average = len(raw_data) - i - 1
                    # Smoothing considering the scaled down moving average
                    smoothed_data_ema.append(round(
                        sum(raw_data[i - scaled_down_moving_average:i + scaled_down_moving_average + 1]) / (
                                    scaled_down_moving_average * 2 + 1), number_of_decimals))

                # Case 5: Smoothing considering data with a suitable value of moving average
                else:
                    smoothed_data_ema.append(round(
                        sum(raw_data[i - moving_average_range:i + moving_average_range + 1]) / (
                                    moving_average_range * 2 + 1), number_of_decimals))

            return Response.success(
                smoothed_data_ema)  # return smoothed data points based on simple moving average algorithm
        except:

            if type(raw_data) is not list:
                return Response.success("Error, data with " + str(
                    type(raw_data)) + " is not suitable, pass data with class List or pandas.core.series.Series")
            elif type(number_of_decimals) is not int:
                return Response.success('Error, ' + str(
                    number_of_decimals) + 'is not a suitable type for the number of decimals, please enter an integer '
                                          'value for the number of decimals.')
            elif type(moving_average_range) is not int:
                return Response.success('Error, ' + str(
                    moving_average_range) + 'is not a suitable type for the moving average range, please enter an '
                                            'integer value for the moving average.')
            else:
                return Response.success('Error, check the quality of the data')

    @classmethod  # Method 2: Cumulative moving average
    def cumulative_moving_average(cls, raw_data, number_of_decimals=2):
        try:
            smoothed_data_cma = []  # initializing an empty list
            for i in range(0, len(raw_data)):
                # Computing the smoothed data point of interest based on all the previou(s) data points
                smoothed_data_cma.append(round(sum(raw_data[0:i + 1]) / (i + 1), number_of_decimals))

            return Response.success(
                smoothed_data_cma)  # return smoothed data based on Cumulative moving average algorithm

        except:

            if type(raw_data) is not list:
                return Response.success('Error, data with ' + str(
                    type(raw_data)) + ' is not suitable, pass data with class List or pandas.core.series.Series')
            elif type(number_of_decimals) is not int:
                return Response.success('Error, ' + str(
                    number_of_decimals) + 'is not a suitable type for the number of decimals, please enter an integer '
                                          'value for the number of decimals.')
            else:
                return Response.success('Error, check the quality of the data')

    @classmethod  # Method 3: Exponential moving average
    def exponential_moving_average(cls, raw_data, number_of_decimals=2, smoothing_factor_alpha=0.2):
        # Overwriting the list
        smoothed_data_ema = []
        try:
            for i in range(0, len(raw_data)):

                # Passing the first data point without any smoothing
                if i == 0:
                    smoothed_data_ema.append(round(raw_data[i], number_of_decimals))
                # Smoothing the data points based on the current data point and
                # the previous data point while considering a weight factor Alpha
                else:
                    smoothed_data_ema.append(
                        round(smoothing_factor_alpha * raw_data[i] + (1 - smoothing_factor_alpha) * smoothed_data_ema[i - 1],
                              number_of_decimals))
            return Response.success(
                smoothed_data_ema)  # return smoothed data based on exponential moving average algorithm

        except:

            if type(raw_data) is not list:
                return Response.success('Error, data with ' + str(
                    type(raw_data)) + ' is not suitable, pass data with class List or '
                                      'pandas.core.series.Series')
            elif type(number_of_decimals) is not int:
                return Response.success('Error, ' + str(
                    number_of_decimals) + ' is not a suitable type for the number of decimals, '
                                          'please enter an integer value for the number of decimals.')
            else:
                return Response.success('Error, check the quality of the data')
