from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# load dataset
X, y = load_iris(return_X_y=True)

# train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")