import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Load dataset
column_names = ["ID", "TITLE", "GENRE", "DESCRIPTION"]

df = pd.read_csv(
    "movies/train_data.txt",
    sep=" ::: ",
    names=column_names,
    engine="python"
)

print("Dataset loaded successfully!")


# Keep only main movie genres
df = df[df["GENRE"].isin([
    "action",
    "comedy",
    "horror",
    "romance",
    "thriller",
    "sci-fi",
    "fantasy"
])]


# Features and labels
X = df["DESCRIPTION"]
y = df["GENRE"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Train model
model = LinearSVC(
    class_weight="balanced",
    C=2
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Test accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", round(accuracy * 100, 2), "%")


# User prediction
while True:

    plot = input("\nEnter movie plot (type 'exit' to quit): ")

    if plot.lower() == "exit":
        break

    plot_vector = vectorizer.transform([plot])

    prediction = model.predict(plot_vector)

    print("Predicted Genre:", prediction[0])
