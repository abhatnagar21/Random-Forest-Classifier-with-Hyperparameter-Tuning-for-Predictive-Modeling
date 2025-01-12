'''# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load training data
data = pd.read_csv('Dataset.txt', delimiter='\t')

# Load test data
test_data = pd.read_csv('Dataset_test.txt', delimiter='\t')

# Preprocessing
# Convert date columns to numerical features (e.g., year difference)
for col in ['F15', 'F16']:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    test_data[col] = pd.to_datetime(test_data[col], errors='coerce')
    data[f'{col}_diff'] = (pd.Timestamp.now() - data[col]).dt.days
    test_data[f'{col}_diff'] = (pd.Timestamp.now() - test_data[col]).dt.days

# Drop original date columns
data.drop(columns=['F15', 'F16'], inplace=True)
test_data.drop(columns=['F15', 'F16'], inplace=True)

# Handle missing values (if any)
data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Separate features and target variable
X = data.drop(columns=['Index', 'C'])
y = data['C']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Validate the model
y_pred_val = rf_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Classification Report:\n", classification_report(y_val, y_pred_val))

# Predict on training data
y_pred_train = rf_model.predict(X)
predictions_train = pd.DataFrame({'Index': data['Index'], 'Class': y_pred_train})
predictions_train.to_csv('predictions_train.txt', sep='\t', index=False)

# Predict on test data
y_pred_test = rf_model.predict(test_data.drop(columns=['Index']))
predictions_test = pd.DataFrame({'Index': test_data['Index'], 'Class': y_pred_test})
predictions_test.to_csv('predictions_test.txt', sep='\t', index=False)

# Save the model script
with open('training_script.py', 'w') as f:
    f.write(open(__file__).read())

print("Files saved: predictions_train.txt, predictions_test.txt, training_script.py")'''
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load training data
data = pd.read_csv('Dataset.txt', delimiter='\t')

# Load test data
test_data = pd.read_csv('Dataset_test.txt', delimiter='\t')

# Preprocessing
# Convert date columns to numerical features (e.g., year difference)
for col in ['F15', 'F16']:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    test_data[col] = pd.to_datetime(test_data[col], errors='coerce')
    data[f'{col}_diff'] = (pd.Timestamp.now() - data[col]).dt.days
    test_data[f'{col}_diff'] = (pd.Timestamp.now() - test_data[col]).dt.days

# Drop original date columns
data.drop(columns=['F15', 'F16'], inplace=True)
test_data.drop(columns=['F15', 'F16'], inplace=True)

# Handle missing values (if any)
data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Encode categorical features (if any)
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    if col != 'Index':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        test_data[col] = le.transform(test_data[col])
        label_encoders[col] = le

# Separate features and target variable
X = data.drop(columns=['Index', 'C'])
y = data['C']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data.drop(columns=['Index']))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model with hyperparameter tuning
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Validate the model
y_pred_val = rf_model.predict(X_val)
y_pred_val_proba = rf_model.predict_proba(X_val)[:, 1]
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("ROC-AUC Score:", roc_auc_score(y_val, y_pred_val_proba))
print("Classification Report:\n", classification_report(y_val, y_pred_val))

# Predict on training data
y_pred_train = rf_model.predict(X)
predictions_train = pd.DataFrame({'Index': data['Index'], 'Class': y_pred_train})
predictions_train.to_csv('predictions_train.txt', sep='\t', index=False)

# Predict on test data
y_pred_test = rf_model.predict(test_data_scaled)
predictions_test = pd.DataFrame({'Index': test_data['Index'], 'Class': y_pred_test})
predictions_test.to_csv('predictions_test.txt', sep='\t', index=False)

# Save the model script
with open('training_script.py', 'w') as f:
    f.write(open(__file__).read())

print("Files saved: predictions_train.txt, predictions_test.txt, training_script.py")

