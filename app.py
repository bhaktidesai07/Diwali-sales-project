import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Diwali_Sales_Data_100k.csv", encoding='latin1')

# Drop unnecessary columns
cols_to_drop = ['Status', 'unnamed1', 'User_ID', 'Product_ID']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Remove null values
df.dropna(inplace=True)

# Convert categorical to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df_encoded.drop('Amount', axis=1)
y = df_encoded['Amount']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "columns.pkl")

print("Model saved successfully!")
