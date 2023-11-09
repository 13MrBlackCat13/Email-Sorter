
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(model_name, X_train, y_train):
    if model_name == 'lr':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'svm':
        model = SVC()
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    elif model_name == 'gb':
        model = GradientBoostingClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid model_name")

    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy
