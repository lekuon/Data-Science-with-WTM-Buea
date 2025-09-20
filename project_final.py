import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# To load the Ames Housing dataset
df = pd.read_csv('AmesHousing.csv')

remove_cols = ['Order', 'PID', 'Lot Shape','Alley',
               'Utilities', 'Lot Config',
               'Roof Style', 'Roof Matl','fence', 'Heating QC',
               'Central Air', 'Electrical', 'Functional', 'Fireplace Qu', 'Pool QC', 'Fence', 'Misc Feature',
               'Sale Type', '3Ssn Porch'] # Columns to remove due to high cardinality or irrelevance

# This code is to get a quick overview of the dataset
print(df.shape)
print(df.info())
print(df.describe())

# Check for missing values and print the in the terminal in a descending order
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# delete the columns in remove_cols if they exist in the dataframe
df_ready = df.drop(columns=[col for col in remove_cols if col in df.columns])
df_ready.to_csv('AmesHousing_ready.csv', index=False)
print("File AmesHousing_ready.csv stored successfully.")

# Load the dataset just saved
df = pd.read_csv('AmesHousing_ready.csv')

# Handle missing values

# 1. Lot Frontage : replacing the missing values by the median of the neighborhood
if 'Lot Frontage' in df.columns and 'Neighborhood' in df.columns:
    df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
        lambda x: x.fillna(x.median())
    )

# 2. Alley : replacing the missing values by "NoAlley"
if 'Alley' in df.columns:
    df['Alley'] = df['Alley'].fillna('NoAlley')

# 3. Mas Vnr Type : replacing the missing values by "None"
if 'Mas Vnr Type' in df.columns:
    df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')

# 4. Mas Vnr Area : replacing the missing values by 0
if 'Mas Vnr Area' in df.columns:
    df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

# 5. variables de underground basement
bsmt_cat_vars = [
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2'
]
bsmt_num_vars = [
    'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
    'Bsmt Full Bath', 'Bsmt Half Bath'
]
#The code below is to convert the missing values in categorical variables to 'None' and in numerical variables to 0
for col in bsmt_cat_vars:
    if col in df.columns:
        df[col] = df[col].fillna('None')
for col in bsmt_num_vars:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 6. garage variables
garage_cat_vars = [
    'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond'
]
garage_num_vars = [
    'Garage Yr Blt', 'Garage Cars', 'Garage Area'
]
# The code below is to convert the missing values in categorical variables to 'None' and in numerical variables to 0
for col in garage_cat_vars:
    if col in df.columns:
        df[col] = df[col].fillna('None')
for col in garage_num_vars:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# storage of the file
df.to_csv('AmesHousing_ready.csv', index=False)
print("File AmesHousing_ready.csv save with the missing values.")

# Verify that there is no more missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# load the dataset again
df = pd.read_csv('AmesHousing_ready.csv')

# numerical columns to normalize
num_cols_norm = [
    'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
    'Mas Vnr Area', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Gr Liv Area',
    'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd',
    'Garage Cars', 'Garage Area',
    'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 'Screen Porch'
]

# application of the StandardScaler to the numerical columns
scaler = StandardScaler()
for col in num_cols_norm:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# categorical columns to encode (OneHotEncoder recommended for most algorithms)
cat_cols_encode = [
    'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Lot Config', 'Land Slope',
    'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
    'Exterior 1st', 'Exterior 2nd', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond',
    'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC', 'Central Air', 'Electrical',
    'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond',
    'Paved Drive', 'Pool QC', 'Fence', 'Misc Feature', 'Sale Type', 'Sale Condition'
]

# encoding categorical columns using pd.get_dummies
df_encoded = pd.get_dummies(df, columns=[col for col in cat_cols_encode if col in df.columns], drop_first=True)

# Save the normalized and encoded file
df_encoded.to_csv('AmesHousing_ready_normalized.csv', index=False)
print("File AmesHousing_ready_normalized.csv saved with normalization and encoding.")


# Feature engineering: creating new features

# 1. Load the normalized dataset 
# Replace with your CSV file name
df = pd.read_csv("AmesHousing_ready_normalized.csv")

# 2. Creation of the new features 

# Total surface area (basement + ground floor + upper floors)
df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]

# Total number of bathrooms (weighted)
df["TotalBaths"] = (
    df["Full Bath"] +
    0.5 * df["Half Bath"] +
    df["Bsmt Full Bath"] +
    0.5 * df["Bsmt Half Bath"]
)

# age of the house at the time sold
df["HouseAge"] = df["Yr Sold"] - df["Year Built"]

# years since last remodel
df["SinceRemodel"] = df["Yr Sold"] - df["Year Remod/Add"]

# Total surface area of porches
df["TotalPorchSF"] = (
    df["Open Porch SF"] +
    df["Enclosed Porch"] +
    df["Screen Porch"]
)

# Interaction Surface * Quality
df["AreaQualInteraction"] = df["Gr Liv Area"] * df["Overall Qual"]

# Score of overall quality and condition
df["OverallScore"] = df["Overall Qual"] * df["Overall Cond"]

# 3. Creation of binary variables
df["HasPool"] = (df["Pool Area"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["HasGarage"] = (df["Garage Area"] > 0).astype(int)
df["HasBsmt"] = (df["Total Bsmt SF"] > 0).astype(int)
df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)

# 4. useful ratios
df["BsmtRatio"] = df["Total Bsmt SF"] / df["TotalSF"]
df["GarageRatio"] = df["Garage Area"] / df["TotalSF"]
df["BathPerRoom"] = df["TotalBaths"] / df["TotRms AbvGrd"]

# Replace NaN created by divisions by 0
df.fillna(0, inplace=True)

# 5. Save the new dataset
df.to_csv("AmesHousing_ready_normalized.csv", index=False)
print("✅ New features added and dataset saved as 'AmesHousing_ready_normalized.csv'")
print(f"New shape of the dataset: {df.shape}")

# EDA (Exploratory Data Analysis)
# Load the dataset with new features
df = pd.read_csv("AmesHousing_ready_normalized.csv")

#  1. an overview of the target variable (SalePrice)
plt.figure(figsize=(8,4))
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# 2. Boxplot of the target variable
plt.figure(figsize=(8,2))
sns.boxplot(x=df["SalePrice"])
plt.title("Boxplot of SalePrice")
plt.xlabel("SalePrice")
plt.show()

# 3. scatter plots of numerical features vs SalePrice
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in num_cols:
    if col != "SalePrice":
        plt.figure(figsize=(8,4))
        sns.scatterplot(x=df[col], y=df["SalePrice"])
        plt.title(f"Scatter plot of {col} vs SalePrice")
        plt.xlabel(col)
        plt.ylabel("SalePrice")
        plt.show()
        
correlation = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
high_corr = correlation[correlation.abs() > 0.4]
print(high_corr)

# 4. Correlation heatmap
plt.figure(figsize=(12,10))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
im = plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

df["SalePrice_log"] = np.log(df["SalePrice"])
df.to_csv("AmesHousing_ready_normalized.csv", index=False)
print("✅ Transformation log(SalePrice) apply and save in 'AmesHousing_ready_normalized.csv'")

df = pd.read_csv("AmesHousing_ready_normalized.csv")

#  1. an overview of the target variable (SalePrice_log)
plt.figure(figsize=(8,4))
sns.histplot(df["SalePrice_log"], bins=30, kde=True)
plt.title("Distribution of SalePrice_log")
plt.xlabel("SalePrice_log")
plt.ylabel("Frequency")
plt.show()

# 2. Boxplot of the target variable
plt.figure(figsize=(8,2))
sns.boxplot(x=df["SalePrice_log"])
plt.title("Boxplot of SalePrice_log")
plt.xlabel("SalePrice_log")
plt.show()

# 3. scatter plots of numerical features vs SalePrice
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in num_cols:
    if col != "SalePrice_log":
        plt.figure(figsize=(8,4))
        sns.scatterplot(x=df[col], y=df["SalePrice_log"])
        plt.title(f"Scatter plot of {col} vs SalePrice_log")
        plt.xlabel(col)
        plt.ylabel("SalePrice_log")
        plt.show()

correlation = df.corr(numeric_only=True)["SalePrice_log"].sort_values(ascending=False)
high_corr = correlation[correlation.abs() > 0.4]
print(high_corr)

# 4. Correlation heatmap
plt.figure(figsize=(12,10))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
im = plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

df = pd.read_csv("AmesHousing_ready_normalized.csv")

# Identify categorical columns (if any remain)
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# One-Hot Encode any remaining categorical variables
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target variable (y)

X = df.drop(columns=["SalePrice", "SalePrice_log"], errors="ignore")
y = df["SalePrice_log"]


# 2. TRAIN/TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# 3. LINEAR REGRESSION


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Evaluate Linear Regression
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)

print("🔹 Linear Regression Results")
print(f"RMSE: {rmse_lin:.4f}")
print(f"R² Score: {r2_lin:.4f}")



# 4. RANDOM FOREST REGRESSOR


rf_reg = RandomForestRegressor(
    n_estimators=200,   # Number of trees
    max_depth=None,     # No maximum depth (fully grown trees)
    random_state=42,
    n_jobs=-1           # Use all CPU cores for faster training
)

rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Evaluate Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Regressor Results")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")



# 5. PREDICTION VISUALIZATION - LINEAR REGRESSION


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_lin, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Predictions - Linear Regression")
plt.show()



# 6. PREDICTION VISUALIZATION - RANDOM FOREST


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Predictions - Random Forest")
plt.show()



# 7. COMPARISON OF BOTH MODELS


plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lin, alpha=0.5, label="Linear Regression", color="blue")
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest", color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Model Comparison - Linear Regression vs Random Forest")
plt.legend()
plt.show()

# For Linear Regression: absolute value of coefficients
lin_coef = pd.Series(np.abs(lin_reg.coef_), index=X.columns)
top_lin_features = lin_coef.sort_values(ascending=False).head(15)
print("\nTop 15 Important Features (Linear Regression):")
print(top_lin_features)

# For Random Forest: feature importances
rf_importances = pd.Series(rf_reg.feature_importances_, index=X.columns)
top_rf_features = rf_importances.sort_values(ascending=False).head(15)
print("\nTop 15 Important Features (Random Forest):")
print(top_rf_features)
