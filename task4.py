import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score



# LOAD DATA

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

df = df[['v1','v2']]

df.columns = ['label','message']


df['label'] = df['label'].map(
    {
        'ham':0,
        'spam':1
    }
)


print("Dataset Loaded Successfully!")


# SPLIT DATA

X = df['message']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# TF-IDF

vectorizer = TfidfVectorizer(
    stop_words="english"
)


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)



# NAIVE BAYES

nb = MultinomialNB()

nb.fit(
    X_train,
    y_train
)

nb_pred = nb.predict(X_test)



# LOGISTIC REGRESSION

lr = LogisticRegression()

lr.fit(
    X_train,
    y_train
)

lr_pred = lr.predict(X_test)



# SVM

svm = LinearSVC()

svm.fit(
    X_train,
    y_train
)

svm_pred = svm.predict(X_test)



# OUTPUT ONLY RESULTS

print("\nModel Results")

print("----------------------")

print(
    "Naive Bayes Accuracy:",
    round(accuracy_score(y_test, nb_pred)*100,2),
    "%"
)


print(
    "Logistic Regression Accuracy:",
    round(accuracy_score(y_test, lr_pred)*100,2),
    "%"
)


print(
    "SVM Accuracy:",
    round(accuracy_score(y_test, svm_pred)*100,2),
    "%"
)



# TEST MULTIPLE MESSAGES

# TEST MULTIPLE MESSAGES

print("\nSpam Checker")
print("Type exit to stop")


while True:

    msg = input("\nEnter a message: ")

    if msg.lower().strip() == "exit":
        print("Exiting...")
        break


    if len(msg.strip()) == 0:
        print("Please enter a message!")
        continue


    msg_vector = vectorizer.transform([msg])


    result = svm.predict(msg_vector)


    if result[0] == 1:
        print("Result: SPAM")

    else:
        print("Result: LEGITIMATE")


print("\nTask Completed Successfully!")
