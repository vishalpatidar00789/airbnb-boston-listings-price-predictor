# airbnb-boston-listings-price-predictor

This repository contains a python script and python notebook which requires a python environment to run. Boston Airbnb dataset is stored under datasets/boston folder(path: './datasets/boston/listings.csv').

**Libraries Used**

NumPy for linear algebra.
Pandas for reading csv file, processing and transformations.
Matplot and Seaborn for plotting charts.
Sklearn for splitting samples into train and test samples, training and evaluation of model.

# Business Understanding
Airbnb is an American vacation rental online marketplace company based in San Francisco, California. Airbnb maintains and hosts a marketplace, accessible to consumers on its website or app. Users can arrange lodging, primarily homestays, and tourism experiences or list their spare rooms, properties, or part of it for rental. On the other hand, users who are traveling are looking for stays search properties and rooms by neighborhood or location. Airbnb recommends the best price in the neighborhood and users book the best deal. Thanks to Kaggle and Udacity that I got a chance to analyze Airbnb listings of Boston city. Boston Airbnb listings dataset has various features such as neighborhood, property_type, bedrooms, bathrooms, beds, price, reviews, ratings, etc. It was interesting to see what features are affecting the price in Boston city and draw interesting conclusions. I was more interested in training and evaluating the model and to see how the model has performed while predicting the prices in Boston city at Airbnb. 

My primary goal would be to answer the following questions:

1. What Features are affecting the price most? name the features that affect the price most.
2. How do features affect the price of listings? Do experience and comfort cost more to the user?
3. Can we predict the price of a listing in Boston AirBnB?

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

# Conclusions

Finally, I have trained and evaluated two models and I can see RandomForestRegressor did a better job as compare to LinearRegressor. I can see the absolute mean error for RFR is 0.31 whereas 0.35 for LR. I can see the scope of doing better here, nevertheless, Kaggle and Udacity will give me many more opportunities to perform better.

Let's try to answer the following questions based on the study we did.

**1. What Features are affecting the price most? name the features that affect the price most.**

Answer: Based on the study we can see that the following features are affecting the price:

Selected Numerical Features:

price, latitude,longitude, accommodates, bedrooms, bathrooms, beds, security_deposit, cleaning_fee, guests_included, availability_30, availability_60, availability_90, availability_365, review_score_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_location, review-scores_value, calculated_host_listings_count.

Selected Categorical Features:

host_response_time, host_is_superhost, room_type, bed_type, neighbourhood_cleansed, cancellation_policy, property_type, host_identity_verified, instant_bookable, host_has_profile_pic, require_guest_profile_picture, require_guest_phone_verification.


**2. How do features affect the price of listings? Do experience and comfort cost more to the user?**

Answer: Based on the study above it is proven that experience and comfort cost more to the user!

1. It is seen that when the user seeks an extra bathroom it costs 40USD more on average.
2. Interestingly, beds, bedroom and price pattern is not consistent. Though it slightly indicates that if users seek an extra bed it might cost them extra.
3. From the above charts I can see that Jamaika Plains, South End, and Back Bay have a high volume of listings. Lowest in Leather District and Longwood Medical Area.
4. It is also seen that Leather District, China Town, and Downtown are super expensive while Hyde Park, Mattapan, and Dorchester are the cheapest neighborhoods.
5. Guesthouses, boats, lofts, and villas are expensive as compare to homes and apartments.
6. As expected Entire home is expensive while private rooms and shared rooms are the cheapest for users.
7. Listings that offer Real Beds are comparatively expensive while Airbeds are cheaper. It's all about comfort. As comfort level goes up price also tends to go up.
8. Listings that have high prices tend to have a strict cancellation policy while listings of cheaper prices are super flexible.
9. Superhost lists their prices slightly higher than those who are not Superhost. Again experience matters which costs more to the user.
10. Listings that offer instant booking are slightly cheaper than listings that don't offer.
11. Listings where the host is verified are slightly higher in price than listings where the host is not verified.
12. Listings that require guest phone verification are comparatively higher in price than listings that don't require.

**3. Can we predict the price of a listing in Boston AirBnB?**

Answer: RandomForestRegressor did a better job as compare to LinearRegressor. The absolute mean error for RFR is 0.31 whereas 0.35 for LR.

blogs: 

https://www.kaggle.com/vishalpatidar00789/boston-airbnb-price-predictor

https://vishubemine.medium.com/what-features-do-affect-the-price-of-airbnb-in-boston-e961cceb31cd
