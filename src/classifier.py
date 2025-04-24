from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

class MHIClassifier:
    def __init__(self, classifier_type='knn', k=5, ada_estimators=50, rf_estimators=50):
        """
        Initializes the MHIClassifier with the specified classifier type and parameters.

        Args:
            classifier_type (str): The type of classifier to use. Options are:
               'knn' (K-Nearest Neighbors),
               'svm' (Support Vector Machine),
               'ada' (AdaBoost),
               'ensemble' (Voting ensemble of KNN, SVM, RandomForest, and AdaBoost).
            k (int): Number of neighbors for the KNN classifier. Default is 5.
            ada_estimators (int): Number of weak learners (estimators) for the AdaBoost classifier. Default is 50.
            rf_estimators (int): Number of trees (estimators) for the RandomForest classifier in the ensemble. Default is 50.

        Raises:
            ValueError: If an unknown `classifier_type` is provided.
        """
        if classifier_type == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=k)
        elif classifier_type == 'svm':
            self.clf = SVC(kernel='rbf', probability=True)
        elif classifier_type == 'ada':
            self.clf = AdaBoostClassifier(n_estimators=ada_estimators)
        elif classifier_type == 'ensemble':
            knn = KNeighborsClassifier(n_neighbors=k)
            svm = SVC(kernel='rbf', probability=True)
            rf = RandomForestClassifier(n_estimators=rf_estimators)
            ada = AdaBoostClassifier(n_estimators=ada_estimators)
            # Include AdaBoost in voting ensemble
            self.clf = VotingClassifier(
                estimators=[('knn', knn), ('svm', svm), ('rf', rf), ('ada', ada)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=self.clf.classes_)
        report = classification_report(y, y_pred)
        return acc, cm, report

    def save(self, path):
        """
        Saves the trained classifier to a file.

        Args:
            path (str): Path to save the classifier.
        """
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        """
        Loads a trained classifier from a file.

        Args:
            path (str): Path to the saved classifier file.

        Returns:
            MHIClassifier: An instance of the `MHIClassifier` with the loaded model.
        """
        obj = cls()
        obj.clf = joblib.load(path)
        return obj