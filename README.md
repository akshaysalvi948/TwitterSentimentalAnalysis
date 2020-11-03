# TweeterSentimentalAnalysis
## Sentimental analysis based upon tweeter tweets for maggi company.
### step1 : Add keys and tokens from the Twitter Dev Console to the `config.py` file

### step2 : Run train.py file to train the sentimental analysis model,
I had trained on three algorithms the Accuracy are as follows :

Naive Bayes: 66.00%
Accuracy for Logistics Regression: 68.00%
Accuracy for XGBOOST: 75.00%

### step3 : Results
Results are stored in  `results.csv` file which contains sentiment prediction based upon the XGBOOST algorithm, the sentiment are
1. positive 
2. slightly positive
3. neutral
4. slightly negative
5. negative

### Note : To access the twitter API'S,please use your credentials.