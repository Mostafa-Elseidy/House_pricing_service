import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_extraction import DictVectorizer


def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(random_state=42,
                                            max_depth=10, min_samples_split=5, max_leaf_nodes=20))
    ])
    pipeline.fit(X_train, y_train)
    return dv, pipeline


def predict(df, dv, pipeline):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = pipeline.predict(X)

    return y_pred


# load data
df = pd.read_csv("housing_prices.csv")

# X, y
X = df.drop("Price", axis=1)
y = df["Price"]
print(X.shape, y.shape)

# splitting
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0)
print("Train split shape:", X_train.shape, y_train.shape)
print("Validation split shape:", X_val.shape, y_val.shape)
print("Test split shape:", X_test.shape, y_test.shape)

# build the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', DecisionTreeRegressor(random_state=42,
     max_depth=10, min_samples_split=5, max_leaf_nodes=20))
])

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# training the final model
print('training the pipeline')
dv, pipeline = train(X_train, y_train)
y_pred = predict(X_test, dv, pipeline)
# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
explained_variance = explained_variance_score(y_val, y_pred)

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f"1. Mean Absolute Error (MAE): {mae:.4f}")
print(f"2. Mean Squared Error (MSE): {mse:.4f}")
print(f"3. Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"4. R² Score: {r2:.4f}")
print(f"5. Explained Variance Score: {explained_variance:.4f}")

# save the model
with open("dt_model.bin", 'wb') as f_out:
    pickle.dump((dv, pipeline), f_out)

print('model saved.')
