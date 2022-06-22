# tweet-stock-prediction

use tweets to predict public mood of market, weight, and stock price (gain/loss)

## Usage

### 1. 下載Tesla的data到./data/中
- [Stock tweets dataset](https://drive.google.com/drive/folders/1ijq999xwXj03xw6I5Q6Dxd8ix8lynXui?usp=sharing)
```shell=
cd tweet-stock-prediction
mkdir data
```

### 2. Test whether code is fine
```python=
python ./run_unittest.py
```


### 3. Something you can use :+1: 
```python=
python .\src\tweet_score\model.py
python .\src\tweet_score\dataLoader.py
```

## 資料集描述
| Data | Source | How to Collect | Volume | Velocity | Variety |
| --- | --- | --- | --- | --- | --- |
| stock price | yfinance | API | 549 days stock data | real time | csv |
| Tweets about TSLA | stocktwits | API | 2204889 Tweets 4.07 GB | About 122.59 Tweets/hour | json csv |



- Model design

![flow chat of All-in-one Model](https://i.imgur.com/9OGnXY3.png)

- Feature Importance

![](https://i.imgur.com/CSZmfis.png)

- All-in-one Model

![](https://i.imgur.com/DgL1n48.png)



## 分析結果


#### 股票價格預測的不同數據表格

| Stock price | Model | Output | Evaluation | 最終選擇 | 超參數 |
| --- | --- | --- | --- | --- | --- |
| select only “close” feature | LSTM price prediction  | price, MAPE | val_loss: 0.067  F1 Score:0.514 MAPE: 4.026 % | ❌ |n_LSTM_layer: 2 n_Dense_layer: 1 LSTM_0_units: 64 LSTM_1_units: 64 |
| all stock feature | LSTM price prediction | price, MAPE | val_loss: 0.034  F1 Score: 0.492 MAPE: 4.157 % | ❌ | n_LSTM_layer: 2 n_Dense_layer: 1 LSTM_0_units: 64 LSTM_1_units: 64 |
| stock price + technical indicators | LSTM price prediction | price, MAPE | val_loss: 0.027 F1 Score: 0.497 MAPE: 4.299 % | ❌ | n_LSTM_layer: 2 n_Dense_layer: 1 LSTM_0_units: 64 LSTM_1_units: 64 |
| All-in-one Model  | LSTM price prediction | price, MAPE | val_loss: 0.052 F1 Score: 0.503   MAPE: 3.792 %| ✅ | n_LSTM_layer: 2 n_Dense_layer: 1 LSTM_0_units: 64 LSTM_1_units: 64 |

---

#### Tweets重要性以及情緒分析各種數據表格

| Tweets | Model | Prediction output | Evaluation | 最終選擇 |
| --- | --- | --- | --- | --- |
| Sentiment Model | Transformer (RoBerta) | Bearish  Bullish (probability) | accuracy=0.93 | ✅ |
| Tweets Importance Analyzer 1 | K-neighbors classifier K=10 | Decide which tweet is foreteller (0/1) | 1’s Precision=0.59 All days accuracy=0.62 | ✅ |
| Tweets Importance Analyzer 2 | Xgboost classifier n_estimators = 100 learning_rate = 0.3 | Decide which tweet is foreteller (0/1) | 1’s Precision=0.56 All days accuracy=0.598 | ❌ |
| Tweet Recommendation 1 | LinUCB | Select 2% tweets’ inference as predict metrics | All days accuracy=0.578 | ❌ |
| Tweet Recommendation 2 | NeuralUCB | Select 2% tweets’ inference as predict metrics | All days accuracy=0.612 | ✅ |