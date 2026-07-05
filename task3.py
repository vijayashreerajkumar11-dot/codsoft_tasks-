import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATASET

df = pd.read_csv("Churn_Modelling.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# REMOVE UNNECESSARY COLUMNS

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# ENCODE CATEGORICAL COLUMNS

geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

df["Geography"] = geo_encoder.fit_transform(df["Geography"])
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# FEATURES AND TARGET

X = df.drop("Exited", axis=1)
y = df["Exited"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# EVALUATION

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# USER INPUT SECTION (FIXED FOR IDLE)

print("\n" + "="*40)
print("     ENTER CUSTOMER DETAILS")
print("="*40)

credit_score = int(input("Credit Score >>> "))
geography = int(input("Geography (0-France,1-Germany,2-Spain) >>> "))
gender = int(input("Gender (0-Female,1-Male) >>> "))
age = int(input("Age >>> "))
tenure = int(input("Tenure >>> "))
balance = float(input("Balance >>> "))
products = int(input("Number of Products >>> "))
credit_card = int(input("Has Credit Card (0/1) >>> "))
active = int(input("Is Active Member (0/1) >>> "))
salary = float(input("Estimated Salary >>> "))

# SHOW ENTERED DATA

print("\n" + "-"*40)
print("     ENTERED DETAILS")
print("-"*40)

print("Credit Score:", credit_score)
print("Geography:", geography)
print("Gender:", gender)
print("Age:", age)
print("Tenure:", tenure)
print("Balance:", balance)
print("Products:", products)
print("Credit Card:", credit_card)
print("Active Member:", active)
print("Salary:", salary)

# PREDICTION

user_data = pd.DataFrame([[
    credit_score, geography, gender, age, tenure,
    balance, products, credit_card, active, salary
]], columns=X.columns)

prediction = model.predict(user_data)

print("\n" + "="*40)

if prediction[0] == 1:
    print("Prediction >>> Customer WILL CHURN")
else:
    print("Prediction >>> Customer WILL STAY")

print("="*40)
