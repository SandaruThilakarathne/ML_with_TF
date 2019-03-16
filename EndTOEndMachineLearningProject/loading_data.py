import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
# from EndTOEndMachineLearningProject.handling_categorical_attributes import
# from EndTOEndMachineLearningProject.CoustomTransforms import CombinedAttributesAdder
# from EndTOEndMachineLearningProject.TransfromationPipelines import piplining
from sklearn.pipeline import Pipeline
# from EndTOEndMachineLearningProject.CoustomTransforms import CombinedAttributesAdder, DataFrameSelector
# from EndTOEndMachineLearningProject.TransfromationPipelines import full

HOUSING_PATH = r"C:\Users\ChampsoftWK26\Desktop\ML_with_TF\EndTOEndMachineLearningProject\housing.csv"

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)


# loading data

def load_hosing_data(data_path=HOUSING_PATH):
    return pd.read_csv(data_path)


housing = load_hosing_data()


# creating test set and train set
def split_trains_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indces = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indces]


trainset, testset = split_trains_test(housing, 0.2)


# print(len(trainset), "train", len(testset), "test")


# adding a unique identifier for the test set and the train set
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

t_set, tr_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

# print(housing["income_cat"].value_counts() / len(housing))
for set in (start_test_set, start_train_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = start_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100,
             label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, )
# plt.legend()
# plt.show()

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# pandas plotting
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

'''
Prepare data for the machine learning
'''
housing = start_train_set.copy()
housing = housing.drop("median_house_value", axis=1)


''' Data Cleaning '''

# using dropna
housing2 = housing.dropna(subset=["total_bedrooms"])
# print(housing)

# using drop
housing3 = housing.drop("total_bedrooms", axis=1)

# using fillna

median = housing["total_bedrooms"].median()
housing4 = housing["total_bedrooms"].fillna(median)



