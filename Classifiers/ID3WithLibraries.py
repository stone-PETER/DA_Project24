import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

class ID3ClassifierWithLibraries:
    def __init__(self, file_name='Data_Processing/train_data_processed.csv', test_file_name='Data_Processing/test_data_processed.csv'):
        self.train_file_name = file_name
        self.test_file_name = test_file_name
        self.train_data = self.load_data(self.train_file_name)
        self.test_data = self.load_data(self.test_file_name)
        self.vectorizer = CountVectorizer()
        self.classifier = DecisionTreeClassifier()
        self.accuracy = None

    def load_data(self, file_name=None):
        data = pd.read_csv(file_name)
        return data

    def preprocess_data(self):
        X = self.vectorizer.fit_transform(self.train_data['text'])
        y = self.train_data['is_spam'].astype('int')
        return X, y

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def evaluate_on_train_data(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        y_pred_train = self.predict(X_train)
        self.accuracy = accuracy_score(y_train, y_pred_train)
        print('ID3 Spam Classifier implemented with external libraries')
        print()
        print(f"The training accuracy obtained is:{self.accuracy * 100:.2f}%")

    def evaluate_on_test_data(self):
        X_test = self.vectorizer.transform(self.test_data['text'])
        y_test = self.test_data['is_spam'].astype('int')
        y_pred_test = self.predict(X_test)

        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred_test)) if true != pred]
        self.plot_misclassified_instances(misclassified_indices, y_test, y_pred_test)
        self.plot_misclassification_pie_chart(misclassified_indices, y_test, y_pred_test)

        self.accuracy = accuracy_score(y_test, y_pred_test)
        print(f"The test accuracy obtained is of: {self.accuracy * 100:.2f}%")

    def plot_misclassified_instances(self, misclassified_indices, y_test, y_pred_test):
        plt.figure(figsize=(10, 6))
        plt.scatter(misclassified_indices, y_pred_test[misclassified_indices], color='red', marker='x', label='Misclassified')
        plt.scatter(misclassified_indices, y_test[misclassified_indices], color='blue', marker='o', label='Real values')
        plt.title('Misclassification on test data set \n (Alg ID3)')
        plt.xlabel('Instance index')
        plt.ylabel('Label (0 - Non-Spam, 1 - Spam)')
        plt.legend()
        plt.show()

    def plot_misclassification_pie_chart(self, misclassified_indices, y_test, y_pred_test):
        correct_count = len(y_test) - len(misclassified_indices)
        misclassified_count = len(misclassified_indices)
        proportions = [correct_count, misclassified_count]

        plt.pie(proportions, labels=['Corect', 'Wrong'], autopct='%1.1f%%', colors=['#33B5E5', '#FF5733'])
        plt.title('Percentage correct/wrong classification (Alg ID3)')
        plt.show()

    def cross_validate(self, cv=5):
        X, y = self.preprocess_data()
        scores = cross_val_score(self.classifier, X, y, cv=cv)
        mean_accuracy = scores.mean()
        print(f"The cross-validation accuracy obtained is of: {mean_accuracy * 100:.2f}%")
    def save_model(self, file_path='id3_model.pkl'):
        with open(file_path, 'wb') as file:
            pickle.dump(self.classifier, file)
        print(f"Model saved to {file_path}")


if __name__ == '__main__':
    id3_classifier = ID3ClassifierWithLibraries()
    id3_classifier.evaluate_on_train_data()
    id3_classifier.cross_validate()
    id3_classifier.evaluate_on_test_data()
    id3_classifier.save_model()