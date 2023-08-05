from pyspark.sql.types import *
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import (VectorAssembler, StringIndexer)
from pyspark.ml import Pipeline
# from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier)
from pyspark.sql.functions import col,isnan,when,count,lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.ml.feature import Imputer
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import corr
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import PCA



spark = SparkSession.builder.appName("Genoprophet").getOrCreate();

# df=spark.read.csv('C:/Users/LENOVO/Desktop/Project/gwas-all-associations.csv', inferSchema=True, header=True)
df=spark.read.csv('gwas-all-associations.csv', inferSchema=True, header=True)
print("no of rows",df.count())
df.printSchema()

# Select the desired fields
selected_fields = ["initial sample size","snps","p-value", "pvalue_mlog", "or or beta", "95% ci (text)", "risk allele frequency", "context", "intergenic", "snp_id_current", "disease/trait"]
df_selected = df.select(*selected_fields)

print("no of columns",len(df_selected.columns))
print("data types",df_selected.dtypes)
df_selected.printSchema()

#checking missing values

missing_count = df_selected.select([count(when(col(c).contains('Unknown') | \
                            col(c).contains('N/A') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c
                           )).alias(c)
                    for c in df_selected.columns])
missing_count.show()

df_without_missing_values = df_selected.filter(col("initial sample size").isNotNull() & col("or or beta").isNotNull() & col("95% ci (text)").isNotNull() & col("context").isNotNull() & col("intergenic").isNotNull() & col("snp_id_current").isNotNull())

# Print the number of missing values after deletion
missing_count = df_without_missing_values.select([count(when(col(c).contains('Unknown') | \
                            col(c).contains('N/A') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c
                           )).alias(c)
                    for c in df_without_missing_values.columns])
missing_count.show()

# print("no of rows",df_without_missing_values.count())

df = df_without_missing_values.distinct()
print("no of rows",df.count())


#imbalancing problem

data = df.groupBy('disease/trait').count()
print("disease/trait",data.count())


# Calculate the percentage of each class

# Calculate the total count of instances
total_count = df.count()

# Calculate the percentage of instances in each group
data = data.withColumn('percentage', col('count')/total_count * 100)

# Show the percentages for each label
data.show()

cols = df.dtypes

# Get a list of numerical features
numerical_cols = [col_name for col_name, dtype in cols if dtype in ["int", "double", "float", "long"]]

# Select only numerical features
numerical_df = df.select(*numerical_cols)
print(numerical_df.columns)

categorical_columns = [col_name for col_name, col_type in cols if col_type in ['string', 'boolean']]

# print the list of categorical columns
print(categorical_columns)


categ_cols = ['initial sample size', 'snps', '95% ci (text)', 'risk allele frequency', 'context', 'snp_id_current','disease/trait']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categ_cols]
indexer_pipeline = Pipeline(stages=indexers)
indexed_df = indexer_pipeline.fit(df).transform(df)

# VECTOR ASSEMBLING

assembler = VectorAssembler(inputCols=numerical_df.columns + [col+"_index" for col in categ_cols], outputCol="features")
assembled_df = assembler.transform(indexed_df)

assembled_df.show()

# SCALER

scaler = MinMaxScaler(inputCol="features",\
         outputCol="scaledFeatures")
scalerModel =  scaler.fit(assembled_df.select("features"))
scaledData = scalerModel.transform(assembled_df)
scaledData.show()


#  PCA


# PCA with all principal components
pca = PCA(k=len(df_selected.columns), inputCol="scaledFeatures", outputCol="pca_features")
pca_model = pca.fit(scaledData)
pca_df = pca_model.transform(scaledData)
pca_df.show()

# Calculate explained variance for each principal component
variance_explained = pca_model.explainedVariance.toArray()
cumulative_variance = [sum(variance_explained[:i + 1]) for i in range(len(variance_explained))]
component_count = list(range(1, len(cumulative_variance) + 1))

# Determine the number of principal components to capture at least 95% of the variance
target_variance = 0.95
num_components_95var = next(i for i, variance in enumerate(cumulative_variance) if variance >= target_variance) + 1

# Plotting the Explained Variance
plt.figure(figsize=(10, 5))
plt.plot(component_count, variance_explained)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance per Principal Component')
plt.grid(True)
plt.show()


# Plotting the PCA curve
plt.plot(component_count, cumulative_variance, marker='o')
plt.axvline(num_components_95var, color='r', linestyle='--', label=f'{num_components_95var} components for 95% variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA Cumulative Variance Explained')
plt.legend()
plt.grid(True)
plt.show()


# extract the PCA features from the DataFrame pca_df and store them in a new DataFrame

def extract(row):
    return tuple(row.pca_features.toArray().tolist())
final_data = pca_df.select("pca_features").rdd\
               .map(extract).toDF(df_selected.columns)
final_data.show()

final_data=final_data.toPandas()
import pandas as pd
final_data.to_csv("dataset_final.csv",index=False)


######################### MODELING #########################




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



data_reduce = pd.read_csv("dataset_final.csv")

X = data_reduce.drop(["disease/trait"], axis=1)
y = data_reduce["disease/trait"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


######   LINEAR REGRESSION   #######



# Create and train a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

#######################################


####### DECISION TREE  ################


# Create a Decision Tree Regressor
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_tree.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title("Actual vs. Predicted Values (Decision Tree Regressor)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.show()


#########################################

########  RANDOM FOREST   ##############



# Create a Random Forest Regressor
reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
reg_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Visualize the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title("Actual vs. Predicted Values (Random Forest Regressor)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.show()

##############################################

########## SVM ###########

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVR model
reg_svm = SVR(kernel='linear')  # You can also try other kernels like 'rbf'
reg_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_svm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title("Actual vs. Predicted Values (SVR)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.show()

###########################################


###### CNN #######



# Load your dataset
data_reduce = pd.read_csv("dataset_final.csv")

X = data_reduce.drop(["disease/trait"], axis=1).values
y = data_reduce["disease/trait"].values

# Reshape your data to be compatible with CNN (as if it's an image)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN regression model
model = keras.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the model architecture
plot_model(model, to_file='cnn_regression_model.png', show_shapes=True, show_layer_names=True)


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# Create scatter plots
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title("Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

# Add a diagonal line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)

plt.show()

##################################################


