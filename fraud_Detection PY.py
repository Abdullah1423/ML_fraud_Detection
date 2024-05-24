import threading

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn import tree
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    make_scorer,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


os.rename(
    "Data/PS_20174392719_1491204439457_log.csv", "Data/Online_Fraud.csv"
)  # renaming the file from kiggle


DB_Original = pd.read_csv(
    "Data/Online_Fraud.csv"
)  # read the data srt and save it to a var


DB_Original.head(10)  # first 10 rows


DB_Original.tail(10)  # last 10 rows


DB_Original.shape  # how many rows asd columns // Large number of rows can increase the performance of the model :)


DB_Original.info()  # what are the types of columns // object need to be redesigned or deleted


DB_Original.describe()  # this will give general statistics


DB_Original.isnull().sum()  # we have no null valuse so no need for imputations


DB_Original.duplicated().sum()  # no dublicates so no need for deleting dublicate function


DB_Original.type.value_counts()  # what are the the valuse redundency in column type


# colors for the uniform the look
Green = "#3ed33e"
Red = "#ff0000"


# This will help us determine which type are we mostly dealing with
type = DB_Original["type"].value_counts()  # same as above
transactions = type.index  # store keys of type
quantity = type.values  # store values of type

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.pie(quantity, labels=transactions, autopct="%1.1f%%")
plt.title("Distribution of Transaction Type")
plt.show()


def thread_function():

    # we foucus on the is fraud column to choose wich would be used as feature
    correlation = DB_Original.corr(numeric_only=True)
    sns.heatmap(
        correlation.round(2), annot=True, cmap="Greens", vmin=-1, vmax=1, linewidths=0.1
    )

    # In[48]:

    DB_Original.isFraud.value_counts().plot(
        title="Fraud vs non Fraud", kind="bar", color=["Green", "lightblue"]
    )
    # To visualize the rasio

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 4)
    )  # Adjust the number of subplots and figure size as needed
    axs[0].hist(DB_Original["step"], bins=5, color="Green")
    axs[1].hist(DB_Original["type"], bins=5, color="Green")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Visualizing the relationship between 'oldbalanceDest' and 'newbalanceDest'
    # we notice that most fraud users have 0 money as old balance wich make sense and new balance
    # in a non fraud situation the old and new balance increace proportionally
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x="oldbalanceDest", y="newbalanceDest", hue="isFraud", data=DB_Original
    )
    plt.title("Old Balance Dest vs New Balance Dest")
    plt.show()

    # Visualizing the relationship between 'oldbalanceOrig' and 'newbalanceOrig'
    # in a case of fraud the new balance drops by lage margine wich make it shife to the right
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x="oldbalanceOrig", y="newbalanceOrig", hue="isFraud", data=DB_Original
    )
    plt.title("Old Balance Org vs New Balance Orig")
    plt.show()

    # In[56]:

    # used to check for outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=DB_Original, orient="h")  # horizontal
    plt.title("Outliers")
    plt.grid(axis="y")
    plt.show()

    # there is no fraud in 'CASH_IN', 'DEBIT', 'PAYMENT' so they can be dropped
    fraud_transactions = DB_Original[
        DB_Original["isFraud"] == 1
    ]  # save draud rows in a new DF
    fraud_counts_by_type = fraud_transactions.groupby(
        "type"
    ).size()  # divide then based on type then calculate the number of accurance
    print(fraud_counts_by_type)

    plt.figure(figsize=(8, 8))
    plt.pie(
        fraud_counts_by_type,
        labels=fraud_counts_by_type.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[Green, Red],
        explode=(0, 0.1),
    )
    plt.title("Proportion of Fraud Transactions by Type")
    plt.show()

    # Exploare the customer
    DB_Original_copy = DB_Original.copy()
    DB_Original_copy["nameDest_first_letter"] = DB_Original_copy["nameDest"].str[0]
    DB_Original_copy["nameDest_first_letter"] = DB_Original_copy[
        "nameDest_first_letter"
    ].replace("C", "Customer")
    DB_Original_copy["nameDest_first_letter"] = DB_Original_copy[
        "nameDest_first_letter"
    ].replace("M", "Merchant")
    all_by_dest = DB_Original_copy.groupby("nameDest_first_letter").size()

    plt.figure(figsize=(8, 8))
    plt.pie(
        all_by_dest,
        labels=all_by_dest.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[Red, Green],
        explode=(0, 0.1),
    )
    plt.title("All Operations by Destination")
    plt.show()

    DB_Original_copy.head(10)

    # if the transaction is a fraus it is indeed between two customers with no Merchant
    fraud_transactions = DB_Original[DB_Original["isFraud"] == 1].copy()
    fraud_transactions["nameOrg_first_letter"] = fraud_transactions["nameOrig"].str[0]
    fraud_transactions["nameDest_first_letter"] = fraud_transactions["nameDest"].str[0]
    fraud_transactions["nameOrg_first_letter"] = fraud_transactions[
        "nameOrg_first_letter"
    ].replace("C", "Customer")
    fraud_transactions["nameDest_first_letter"] = fraud_transactions[
        "nameDest_first_letter"
    ].replace("C", "Customer")
    fraud_by_name_orig_first_letter = fraud_transactions.groupby(
        "nameOrg_first_letter"
    ).size()
    fraud_by_name_dest_first_letter = fraud_transactions.groupby(
        "nameDest_first_letter"
    ).size()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

    axes[0].pie(
        fraud_by_name_orig_first_letter,
        labels=fraud_by_name_orig_first_letter.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[Red],
    )
    axes[0].set_title("Fraud Transactions by Origin")

    axes[1].pie(
        fraud_by_name_dest_first_letter,
        labels=fraud_by_name_dest_first_letter.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[Green],
    )
    axes[1].set_title("Fraud Transactions by Destination")

    # In[9]:

    fraud_transactions.head(10)

    # Create a copy and drop isFlaggedFraud
    F_List = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "nameOrig",
        "nameDest",
    ]
    DB_Pros = DB_Original[F_List].copy(deep=True)

    # Not needed
    DB_Pros["isFraud"] = DB_Pros["isFraud"].replace(
        {1: "Fraud", 0: "Not Fraud"}
    )  # to rename the values

    # Not needed
    DB_Pros["isFraud"] = DB_Original["isFraud"].replace(
        {"Fraud": 1, "Not Fraud": 0}
    )  # to rename the values

    # Fix typo
    DB_Pros = DB_Pros.rename(columns={"oldbalanceOrg": "oldbalanceOrig"})

    # Exexlude Types That Never Have Fraud
    types_to_exclude = ["CASH_IN", "DEBIT", "PAYMENT"]
    DB_Pros = DB_Pros[~DB_Pros["type"].isin(types_to_exclude)]

    # to rename the values From Categorical to numerical
    DB_Pros["type"] = DB_Pros["type"].replace({"CASH_OUT": 0, "TRANSFER": 1})

    # Drop Any Operation That is not from customer to Customer
    DB_Pros["C2C"] = 0
    DB_Pros.loc[
        (DB_Pros["nameDest"].str.startswith("C"))
        & (DB_Pros["nameDest"].str.startswith("C")),
        "C2C",
    ] = 1
    DB_Pros = DB_Pros[DB_Pros["C2C"] == 1]

    # drop unnecessary columns for the current analysis
    DB_Pros = DB_Pros.drop(["nameOrig", "nameDest", "C2C"], axis=1)

    # Calculate the percentage and store it in a new column
    DB_Pros["percentage_OLD"] = (
        (DB_Pros["oldbalanceOrig"] - DB_Pros["newbalanceOrig"])
        / DB_Pros["oldbalanceOrig"]
        * 100
    )
    DB_Pros["percentage_NEW"] = (
        (DB_Pros["oldbalanceDest"] - DB_Pros["newbalanceDest"])
        / DB_Pros["oldbalanceDest"]
        * 100
    )
    # Replace null values with zeros
    DB_Pros = DB_Pros.replace([np.inf, -np.inf], np.nan)
    DB_Pros.fillna(0, inplace=True)

    X = DB_Pros[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrig",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "percentage_NEW",
            "percentage_OLD",
        ]
    ]
    Y = DB_Pros["isFraud"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Standardization (Z-Score Normalization)
    # StandardScaler for Data

    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(X_train)

    # Transform the training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)

    DB_Pros.reset_index(drop=True, inplace=True)
    DB_Pros.head(20)

    # Define the model
    modelSVM = SVC(kernel="rbf", C=30)

    # In[19]:

    # Train the model
    modelSVM.fit(X_train_scaled, Y_train)

    # Make predictions
    Y_pred_SVM = modelSVM.predict(X_test)

    # Evaluate the model
    accuracy_SVM = accuracy_score(Y_test, Y_pred_SVM)
    print("Accuracy:", accuracy_SVM)

    # Confusion matrix and classification report
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_SVM))
    print("Classification Report:\n", classification_report(Y_test, Y_pred_SVM))

    class FFNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(FFNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out

    # Convert pandas DataFrame to numpy arrays
    X_train_array = X_train.values
    y_train_array = Y_train.values
    X_test_array = X_test.values
    y_test_array = Y_test.values

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32)

    # Define hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    learning_rate = 0.001
    num_epochs = 100

    # Initialize the model, loss function, and optimizer
    model = FFNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_tensor)
        y_pred = (y_pred > 0.5).float()
        accuracy = accuracy_score(y_test_tensor, y_pred)
        precision = precision_score(y_test_tensor, y_pred)
        recall = recall_score(y_test_tensor, y_pred)
        f1 = f1_score(y_test_tensor, y_pred)
        conf_matrix = confusion_matrix(y_test_tensor, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)

    # ### Decision Tree Classifier

    # Define the model
    modelDTC = DecisionTreeClassifier()

    # Train the model
    modelDTC.fit(X_train, Y_train)

    # Make predictions
    y_pred = modelDTC.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy:", accuracy)

    # Confusion matrix and classification report
    print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))

    # Define the XGBoost model
    modelXGB = xgb.XGBClassifier()

    # Train the model
    modelXGB.fit(X_train, Y_train)

    # Make predictions on the test set
    y_pred = modelXGB.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    conf_matrix = confusion_matrix(Y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    y_pred = modelDTC.predict(X_test)
    predicted_probabilities = modelDTC.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print(classification_report(Y_test, y_pred))
    print(f"Matthews Correlation Coefficient: {accuracy:.2f}")
    print(f"ROC-AUC Score: {roc_auc_score(Y_test, predicted_probabilities[:, 1]):.2f}")

    def plot_confusion_matrix(modelDTC, X_test, Y_test, title):
        conf_matrix = confusion_matrix(X_test, Y_test)
        sns.heatmap(
            conf_matrix, display_labels=["Not Fraud", "Fraud"], cmap=plt.cm.Blue
        )

        plt.title("Decision Tree Confusion Matrix")
        plt.show()

    plot_confusion_matrix(modelDTC, X_test, Y_test, "Decision Tree Confusion Matrix")

    joblib.dump(
        model1DTC, "Fraud_detection_DTC.joblib"
    )  # save the model into a file [export]

    joblib.load("Fraud_detection_DTC.joblib")  # read the model from a file [import]

    # export a tree
    tree.export_graphviz(
        model1DTC,
        out_file="our_tree.dot",
        feature_names=["step", "type", "amount", "oldbalanceOrig", "newbalanceOrig"],
        class_names=sorted(
            DB_Pros["isFraud"].unique(), label="all", rounded=True, Filled=True
        ),
    )


thread_names = ["LoadAndProcess", "CreatePlot", "TrainSVM", "Task4", "Task5"]
threads = [
    threading.Thread(target=thread_function, args=(name,)) for name in thread_names
]

# Start threads
for thread in threads:
    thread.start()

# Join threads to the main thread
for thread in threads:
    thread.join()
