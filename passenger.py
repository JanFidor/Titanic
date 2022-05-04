from numpy import zeros
import pandas as pd

class Passenger:
    def __init__(self, data: pd.Series) -> None:
        self.survived = int(data["Survived"])
        self.ticket_class = int(data["Pclass"])
        self.sex = data["Sex"]
        self.age = float(data["Age"])
        self.sibsp = int(data["SibSp"])
        self.parch = int(data["Parch"])
        self.fare = float(data["Fare"])

    def display(self):
        print(self.survived, self.ticket_class, self.sex, self.age,
            self.sibsp, self.parch, self.fare)
    
    def ticket_class_as_tuple(self):
        zeros = [0, 0, 0]
        zeros[int(self.ticket_class) - 1] = 1
        return tuple(zeros)
    
    def is_male(self):
        return int(self.sex == "male")

    def actuator(self):
        return self.ticket_class_as_tuple() + (self.is_male(), self.sibsp, self.parch, self.fare)
    
    def fitness(self, guess):
        return int(self.survived == round(guess))


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

