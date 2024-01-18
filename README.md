# WINE SELECTOR
Test case for Kaulfland 

## Objective
1. Identify the main characteristics (features) that drive a good rating and ultimately could define a good wine
   (using wine ratings as a proxy target variable)
2. Provide recommendations on how good wines can be identified and potentially added to the assortment

## Conclusions from the EDA:
1. Considering the distribution of the target variable and the goal of the use case which is to understand
   and being able to determine the high quality wine, it would make most sense to group the 'quality' target variable into
   two buckets: high_quality: 7, 8; normal_and_below_normal_quality: 3, 4, 5, 6. Trying to put 3 and 4 in a
   separate bucket (low quality) didn't work well with the models and anyway is not the focus of the use case. If needed
   would make more sense to have a separate model to classify low quality vs. rest
2. Correlation matrix doesn't show problematic relations between the features and highlights potential strong predictors
   such as alcohol, volatile acidity, sulphates and citric acid
3. For some feature variables there are outliers that would require additional review and might need to be excluded
4. The distribution of the feature variables differs from normal, so for outlier removal IQR would be used (instead of z-score)
   In general would be good to use non-parametric methods for any tests/analysis
5. Feature visualizations would be re-done with the new bucketized target variable
6. Variables to consider in the outlier removal - 'alcohol', 'volatile acidity', 'sulphates', 'citric acid'
7. There doesn't seem to be much room for feature engineering
8. EDA is great and already is showing insights, but it focuses on the relation between a feature and the target variable
   As wine is not simple the sum of its ingredients, we should explore the relations between the feature and how they
   impact the reviews/quality. For this we would use ML modeling. If we can build a good enough model we can further 
   check the importance of its features and use it as proxy to identify what drives the good wine reviews

## Data Preparation
1. Putting all features to outlier removal is eliminating 30% of the train data. The scope of outlier removal would
   be limited only to variables that could influence the model
2. For the training there would be 3 datasets - original; without outliers; without outliers and scaled

## Build the ML model
1. 5 types of explainable ML models were tested – Logistic regression, Decision Tree, Random Forest, GBM and GBM Lite
2. Grid search with 5-fold cross validation was done over the 3 data sets (80/20 split).
3. Overall more than 850 were trained and tested
4. Random Forest was selected as the model with best performance. 
5. The focus is on the high quality class (the minority class):
   Precision: 0.80, Recall: 0.54, F1: 0.65
6. Consideration: Macro Precision was used as a scoring function. The reasoning is that it would be more costly to add
   a wine that is with lower quality. High precision would mean that is not likely that a wine marked as high quality
   would turn out to be regular/low quality. This would make even more sense if the goal is to add new wines in
   the assortment as the shelf space is limited and ideally it should be filled only with high quality wines
7. Conclusions
   1. Alcohol, Sulphates, Volatile acidity and Citric acid are the most important components that determine a red wine rating.
   2. Their total contribution is app. 60%
   3. From data exploratory the relations are – higher Alcohol, Sulphates and Citric Acid and lower Volatile acidity,
      the better can be associated with better scores

## API
The Model is exposed with a simple dockerized Flask API. To run do:

Spin the container:
`docker build -t wine-app .`

Run the container:
`docker run --name red_wine_predictor -d -p 5002:5002 wine-app`

Stop the container:
`docker stop red_wine_predictor `

Start again the container:
`docker start red_wine_predictor` 

Example curl
`curl -X POST -H 'Accept: */*' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 250' -H 'Content-Type: application/json' -H 'User-Agent: python-requests/2.31.0' -d '{"features": {"fixed acidity": 7.2, "volatile acidity": 0.38, "citric acid": 0.38, "residual sugar": 2.8, "chlorides": 0.068, "free sulfur dioxide": 10.0, "total sulfur dioxide": 42.0, "density": 0.99, "pH": 3.34, "sulphates": 0.72, "alcohol": 12.9}}' http://127.0.0.1:5002/predict`

## TODO
1. Test using binned data for modeling. This most likely would reduce the performance, but would increase
   interpretability
2. Try to find more data as 1.6K samples is not great for modeling. There should be wine organizations that keep wine
   statistics
3. Move the API to a deployment space and enhance it if needed