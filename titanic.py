import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#
#
# load data
#
#

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


#
#
# clean data
#
#

# check any missing values, training set has missing values in Age and Embarked
train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].isnull().values.any()

# impute missing value of Age
train["Age"] = train["Age"].fillna(train["Age"].median())

# impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

target = train["Survived"].values


#
#
# train random forrest
#
#

# train random forrest with Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))


#
#
# predict on test set
#
#

# check any missing values, test set has missing values in Age and Fare
test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].isnull().values.any()

# impute missing value of age
test["Age"] = test["Age"].fillna(train["Age"].median())
print(test[["Age"]].isnull().values.any())

# impute missing value of fare
test["Fare"] = test["Fare"].fillna(train["Fare"].median())
print(test[["Fare"]].isnull().values.any())

# convert the male and female groups to integer form
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# predict on test set
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction = my_forest.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("solution.csv", index_label = ["PassengerId"])
