# tweet-stock-prediction

use tweets to predict public mood of market, weight, and stock price (gain/loss)

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


#### Something you can use :+1: 
```python=
python .\src\tweet_score\model.py
python .\src\tweet_score\dataLoader.py
```
- Model design

![flow chat of All-in-one Model](https://i.imgur.com/9OGnXY3.png)

- Feature Importance

![](https://i.imgur.com/CSZmfis.png)
