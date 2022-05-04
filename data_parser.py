from ast import Pass
import numpy as np
import pandas as pd
import os
from passenger import Passenger
from random import shuffle

def read_data(file_name):
    path = os.path.join(os.path.dirname(__file__))
    titanic_data = pd.read_csv(path + "\\" + file_name, index_col=0)
    return titanic_data

def read_training_data():
    df = read_data("train.csv")
    training_set = df.iloc[:(len(df.index) // 5) * 3]
    return training_set

def read_test_data():
    df = read_data("train.csv")
    test_set = df.iloc[(len(df.index) // 5) * 3:]
    return test_set

def create_passengers_objects(df):
    df = df.reset_index()  # make sure indexes pair with number of rows
    clear_cd = df.dropna(thresh=1)
    passengers =  [Passenger(row) for _, row in clear_cd.iterrows()]
    shuffle(passengers)
    return passengers

# ps = create_passengers_objects(read_data("train.csv"))
# ps[0].display()
"""
Survived
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
"""

