import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


house_df = pd.read_csv('Housing.csv')

# read dataset columns
# for i, ele in enumerate(house_df):
#     print(f'{i}: {ele}')

# Binary data, ordinal and nominal encoding
bi_mapping = {"no" : 0, "yes" : 1}
furnishingstatus_mapping = {"unfurnished" : 0, "semi-furnished" : 1, "furnished" : 2}

house_df['hotwaterheating_encoded'] = house_df['hotwaterheating'].map(bi_mapping)
house_df['airconditioning_encoded'] = house_df['airconditioning'].map(bi_mapping)
house_df['prefarea_ecoded'] = house_df['prefarea'].map(bi_mapping)
house_df['furnishingstatus_encoded'] = house_df['furnishingstatus'].map(furnishingstatus_mapping)
house_df = house_df.drop(columns=['hotwaterheating','airconditioning','furnishingstatus'])
house_df = house_df.drop(columns=['mainroad', 'guestroom', 'basement', 'prefarea'])

# Dividing up the dependent and independent varaibes to X and y
X = house_df.iloc[:, 1:]
y = house_df.iloc[:, 0].values
y = y.reshape(len(y),1)

# Spliting the training and test set to a 70:30 proportion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Applying StandScaler to make sure all variable are on the same scale
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Model Training : Mult-Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Create y_pred
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#plot graph
# Plot Actual vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Fit")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Mult-Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()

#model performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared (R2): {r2:.4f}")