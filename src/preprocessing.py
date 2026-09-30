import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_base_data(df):
    data = df.copy()
    data = data.drop_duplicates()
    data = data.dropna(subset = ["price"])
    

    data = data[
        data["category"] == "housing/rent/apartment"
        ].copy()

    data = data[
        data["price_type"] == "Monthly"
        ].copy()

    data = data[
        data["fee"] == "No"
        ].copy()

    data["listing_time"] = pd.to_datetime(
        data["time"],
        unit= 's'
    )
    data["month"] = data["listing_time"].dt.month

    data["sin_month"] = np.sin(data["month"]* 2 *np.pi /12)
    data["cos_month"] = np.cos(data["month"]* 2 *np.pi /12)

    amenity_groups = {
    "has_parking": [
        "parking",
        "garage",
        "carport",
        "driveway"
    ],

    "has_recreation": [
        "gym",
        "fitness",
        "clubhouse",
        "pool",
        "spa",
        "hot tub",
        "tennis"
    ],

    "has_comfort": [
        "washer",
        "dryer",
        "laundry",
        "dishwasher",
        "fireplace",
        "ac",
        "air conditioning",
        "internet",
        "cable"
    ]
    }

    for feature, keywords in amenity_groups.items():
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        matches = data["amenities"].str.contains(
            pattern,
            case = False,
            na = False,
        )
        data[feature] = matches.astype("int")
    selected_columns = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude",
        "sin_month",
        "cos_month",
        "has_photo",
        "has_parking",
        "has_recreation",
        "has_comfort",
        "price"
        ]

    data = data[selected_columns].copy()
    data = data.reset_index(drop = True)

    return data




def split_data(data):

    X = data.drop(columns = ["price"])
    y = data["price"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, 
        y,
        test_size = 0.15,
        random_state = 42
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size= 0.1765,
        random_state= 42 
    )
    return X_train, X_val, X_test, y_train, y_val, y_test 




def input_missing(X_train, X_val, X_test):

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    numeric_columns = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude",
        "sin_month",
        "cos_month",
        "has_parking",
        "has_recreation",
        "has_comfort"
    ]
    train_medians = X_train[numeric_columns].median()

    X_train[numeric_columns] = X_train[numeric_columns].fillna(train_medians)
    X_val[numeric_columns] = X_val[numeric_columns].fillna(train_medians)
    X_test[numeric_columns] = X_test[numeric_columns].fillna(train_medians)

    return X_train, X_val, X_test




def prepare_knn_data(X_train, X_val, X_test):

    X_train_knn = X_train.copy()
    X_val_knn = X_val.copy()
    X_test_knn = X_test.copy()

    X_train_knn = pd.get_dummies(
        X_train_knn,
        columns= ["has_photo"],
        dtype= int
    )

    X_val_knn = pd.get_dummies(
        X_val_knn,
        columns= ["has_photo"],
        dtype= int
    )

    X_test_knn = pd.get_dummies(
        X_test_knn,
        columns= ["has_photo"],
        dtype= int
    )

    X_val_knn = X_val_knn.reindex(
    columns=X_train_knn.columns,
    fill_value=0
)

    X_test_knn = X_test_knn.reindex(
        columns=X_train_knn.columns,
        fill_value=0
    )

    continuous_columns = [
    "bathrooms",
    "bedrooms",
    "square_feet",
    "latitude",
    "longitude",
    "sin_month",
    "cos_month"
]

    scaler = StandardScaler()

    X_train_knn[continuous_columns] = scaler.fit_transform(X_train_knn[continuous_columns])
    X_val_knn[continuous_columns] = scaler.transform(X_val_knn[continuous_columns])
    X_test_knn[continuous_columns] = scaler.transform(X_test_knn[continuous_columns])

    return X_train_knn, X_val_knn, X_test_knn

def prepare_dt_data(X_train, X_val, X_test):
    X_train_dt = X_train.copy()
    X_val_dt = X_val.copy()
    X_test_dt = X_test.copy()

    X_train_dt = pd.get_dummies(
        X_train_dt,
        columns= ["has_photo"],
        dtype= int
    )

    X_val_dt = pd.get_dummies(
        X_val_dt,
        columns= ["has_photo"],
        dtype= int
    )

    X_test_dt = pd.get_dummies(
        X_test_dt,
        columns= ["has_photo"],
        dtype= int
    )

    X_val_dt = X_val_dt.reindex(
    columns=X_train_dt.columns,
    fill_value=0
    )

    X_test_dt = X_test_dt.reindex(
        columns=X_train_dt.columns,
        fill_value=0
    )

    return X_train_dt, X_val_dt, X_test_dt
