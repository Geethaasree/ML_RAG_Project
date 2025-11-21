from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# fake dataset
X = np.random.rand(1000, 5)
y = (X[:,0] + X[:,1]*0.5 + np.random.rand(1000)*0.1 > 0.7).astype(int)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,y)
clf.feature_order = [f"f{i+1}" for i in range(5)]
joblib.dump(clf, "/path/to/ml-rag-project/backend/app/models/churn.pkl")
joblib.dump(clf, "/path/to/ml-rag-project/backend/app/models/fraud.pkl")
