import requests
import pandas as pd


def add_headers_and_replace_none(df):
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
    return df.applymap(lambda d: None if d == " ?" else d)


def download():
    train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
    test  = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header=None, skiprows=1)
    train = add_headers_and_replace_none(train)
    test  = add_headers_and_replace_none(test)
    test["label"] = test["label"].str[:-1]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

    content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names").content
    with open("names.txt", "wb") as f:
        f.write(content)


if __name__ == "__main__":
    download()
