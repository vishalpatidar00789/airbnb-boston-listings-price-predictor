# airbnb-boston-listings-price-predictor

This repository contains a python script which requires a python and kaggle notebook environment to run. It loads Airbnb Boston Listings dataset from kaggle. Here is a full link of the notebook to refer to: https://www.kaggle.com/vishalpatidar00789/boston-airbnb-price-predictor

# Business Understanding
Airbnb is an American vacation rental online marketplace company based in San Francisco, California. Airbnb maintains and hosts a marketplace, accessible to consumers on its website or app. Users can arrange lodging, primarily homestays, and tourism experiences or list their spare rooms, properties, or part of it for rental. On the other hand, users who are traveling are looking for stays search properties and rooms by neighborhood or location. Airbnb recommends the best price in the neighborhood and users book the best deal. Thanks to Kaggle and Udacity that I got a chance to analyze Airbnb listings of Boston city. Boston Airbnb listings dataset has various features such as neighborhood, property_type, bedrooms, bathrooms, beds, price, reviews, ratings, etc. It was interesting to see what features are affecting the price in Boston city and draw interesting conclusions. I was more interested in training and evaluating the model and to see how the model has performed while predicting the prices in Boston city at Airbnb.

# Data Understanding
To understand the data we have to explore it. Thanks to python, pandas, numpy, matplot, seaborn, and sklearn aka scikit learn which made my life easy to perform data science activities. Pandas is been excellent when it comes to load, clean and transform the data sets. Seaborn is a handy package to visualize data concluded from pandas transformation functions. It offers high-level functions to plot bar charts, histograms, distributions, box plots, etc. I have used all these packages to explore the data. I have performed the following data science activities to explore the data:

1. Import packages and read Boston Airbnb datasets
2. Data cleaning and transformation
3. Numerical features analysis
4. Categorical features analysis

# Train and Evaluate model
Here comes the most exciting part of my study. The training and Evaluating model is been always interesting to me. Training and Evaluation of the model are divided into the following steps:

1. Extracting input (X) and output (y) features from the dataset.
2. Split the X and y samples into training and testing samples. train_test_split function of sklearn does the job for us. It takes X and y and splits it into train and test datasets respectively.
3. Instantiate and Fit the model using x_train and y_train samples. sklearn’s model.fit function does the magic here by taking x_train and y_train as input.
4. Predict price using the given model by providing x_test samples. sklearn’s model.predict method takes x_test samples and returns prediction.
5. Calculate mean absolute error by providing y_test (actual output) and prediction samples. mean_absolute_error of sklearn helps to calculate the same.
6. Plot actual values vs predicted values and see how our model performed throughout its journey of learning.

I have used LinearRegressor and RandomForestRegressor as my model.
