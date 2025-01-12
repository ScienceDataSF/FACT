import time
import os
import csv
import traceback
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

start_time = time.time()
error_log_file = "ContentAI/Statistics/error_log.txt"
csv_file = "ContentAI/Statistics/execution_times.csv"

try:
    df = pd.read_csv(r'C:\Users\Tanmay Somani\OneDrive\Desktop\Programming\!Projects\FACT\ContentAI\Dataset\Train_dts.csv')

    X = df['text']
    y = df['score_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    joblib.dump(vectorizer, 'ContentAI/Model/tfidf_vectorizer.pkl')
    joblib.dump(model, 'ContentAI/Model/llm_text_detection_model.pkl')

    print("Model and vectorizer saved successfully!")

    # Log execution time
    if not os.path.exists("ContentAI/Statistics"):
        os.makedirs("ContentAI/Statistics")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Execution Run", "Execution Time (seconds)", "Total Accumulated Time (seconds)"])

    total_time = 0
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                total_time += float(row[1])

    execution_time = time.time() - start_time
    total_time += execution_time
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        run_count = sum(1 for _ in open(csv_file)) - 1
        writer.writerow([run_count, f"{execution_time:.2f}", f"{total_time:.2f}"])

except Exception:
    if not os.path.exists("ContentAI/Statistics"):
        os.makedirs("ContentAI/Statistics")
    with open(error_log_file, mode='a') as log_file:
        log_file.write("-----\n")
        log_file.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(traceback.format_exc())
        log_file.write("-----\n")

def predict_text_class(sample_text):
    try:
        vectorizer = joblib.load('ContentAI/Model/tfidf_vectorizer.pkl')
        model = joblib.load('ContentAI/Model/llm_text_detection_model.pkl')

        sample_tfidf = vectorizer.transform([sample_text])
        prediction = model.predict(sample_tfidf)

        return f"Predicted class: {prediction[0]}"
    except Exception as e:
        print("Error during prediction:", e)
        return None

sample_text = """Python is a high-level, interpreted programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability, which makes it easy for developers to write and understand code. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.

Python is widely used in various fields, including:

1. **Web Development** – Frameworks like Django and Flask allow for the rapid development of web applications.
2. **Data Science & Machine Learning** – Libraries such as NumPy, pandas, TensorFlow, and scikit-learn make Python a go-to language for data analysis, machine learning, and artificial intelligence.
3. **Automation** – Python is commonly used for scripting and automating repetitive tasks.
4. **Software Development** – Python can be used to develop desktop applications, games, and other software.
5. **Scientific Computing** – Libraries like SciPy and Matplotlib help with scientific computations and visualizations.
6. **Internet of Things (IoT)** – Python is used to program devices in IoT due to its simplicity and wide availability of libraries.

Its clean syntax and large ecosystem of third-party libraries contribute to its popularity among beginners and experienced developers alike."""
result = predict_text_class(sample_text)
if result:
    print(result)
