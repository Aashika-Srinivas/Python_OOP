# Python_OOP
This is a repository of classification and regression code using Python (OOP)


# Interpretation of Programming Task based on the Meeting with Prof. Wolf last week

@the ones who also participated: please directly interrupt as soon as I say something that you understood in a different way!


1. The user has to be able to upload a random (maybe can make some limitation if we need) data set (.csv). The 2 data sets we received by the Prof are just examples and should help us to develop the program. 

2. The raw data should be visualized to the user (e.g. describe() + first 10 rows)

3. User has to define which columns should be the x parameters and which one should be the y.

4. The data has to be preprocessed (smoothing, handling of missing values etc.) 
    - questions: 
        - Is this (the preprocessing of the data set) the task of group 2?
        - Should we provide opportunities to the user to choose how the data should be preprocessed (e.g. how he wants to handle missing values) ??? I would say yes

5. The data set has to be split into training and testing data in a way that makes sense (e.g. guarantee, that training data not only consists of low price houses and testing of high price houses - do not trust the data set that is "well distributed")

6. It should be displayed which possibilities the user has with the uploaded data (classification OR regression based on data of y column (continuous vs. discrete))

7. The user should be able to choose which method he wants to use for the processing of his data (several methods implemented by groups 3 and 4).

8. For the chosen method the user should be able to choose the settings (e.g. no. of layers, weights etc. for a neural network - maybe even which kind of measure or graphic he wants if we decide to provide more than one possibility)

9. The main result of the program should be that the user sees the quality of the model performance both in a graphical representation and a performance measure (e.g. mse for regression, confusion matrix for classification).

10. Remark (by Mirco): the focus of our program should be definitively a clean code and good object oriented programming ??? commenting, using classes, objects etc.

11. Further ideas: we could extend the functionality in different ways:
    - Give the user the possibility to predict house prices or classify flower or anything else based on the model built by his choices
    - Compare different models by saving them into a "results" file or something similar ??? user can see the differences between a linear regression, (several) neural network(s), random forest regressor etc.
    - Many more


# Questions/Notes:

- Include functionality to upload a .txt-file?
- Slitting Data into Training and Testing is done by group 3 and 4
- Clarify Testing Procedure with Prof.


# Work flow: new proposal of Group 3 from 14.01.2021 

1. User launches application.
2. Upload data set.
3. Display data set (summary, histogram, table etc.).
4. Select between classification and regression.
5. User selects columns for x and y.
6. User selects how the data should be cleaned. --> group 2
	6a. Display cleaned data.
7. User selects spliting algorithm and ration of training and testing.
	7b. Display split (and cleaned) data
8. User selects algorithm. --> group 3 and 4
9. User chooses settings for algorithm (hyperparameters).
10. Model is trained and tested/evaluated. 
11. Result of testing is displayed to the user both graphically and "numerically".
12. User can insert values to perform prediction.
13. Result (settings and quality) of the algorithm can be saved.
14. History Interface


# Work flow (old version - see new proposal above):

1. User launches application.
2. Upload data set.
3. Display data set (summary, histogram, table etc.).
4. Select between classification and regression.
5. User selects columns for x and y.
6. User selects how the data should be cleaned. 
7. User selects algorithm.
8. User chooses settings for algorithm (hyperparameters). ??? ?
9. User chooses ratio of training and testing data.
10. Model is trained and tested/evaluated. 
11. Result of testing is displayed to the user both graphically and "numerically".
12. User can insert values to perform prediction.


