# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def main():
    # Load the data from the Excel file
    file_path = '/data/Consents.xlsx'
    df = pd.read_excel(file_path)

    # Columns
    feature_columns = ['Country', 'Quarter', 'Shop', 'Childs', 'Sales']
    target_column = 'Consent'

    # Convert categorical variables to numeric (if any)
    df = pd.get_dummies(df, columns=feature_columns, drop_first=True)

    # Split the data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the logistic regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    main()


