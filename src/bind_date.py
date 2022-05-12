from dask import dataframe as dd
import os

dateDir = "./data/TSLA_2020_2022/date_data/"
labeledDir = "./data/TSLA_2020_2022/labeled_data/"

datePaths = [
    dateDir + f
    for f in os.listdir(dateDir)
    if f.endswith(".csv")
]

labeledPaths = [
    labeledDir + f
    for f in os.listdir(labeledDir)
    if f.endswith(".csv")
]

date_data = dd.read_csv(datePaths)
labeled_data = dd.read_csv(labeledPaths)

date_data = date_data.repartition(npartitions=1000)
labeled_data = labeled_data.repartition(npartitions=1000)

labeled_data = labeled_data.merge(date_data, on="id")
            
os.makedirs("./data/TSLA_2020_2022/labeled_data_with_date/")
labeled_data.to_csv("./data/TSLA_2020_2022/labeled_data_with_date/" + "TSLA_2020_2022_*.csv")


