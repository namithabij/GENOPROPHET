
from pyspark.sql.types import *
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import (VectorAssembler, StringIndexer)
from pyspark.ml import Pipeline
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier)
from pyspark.sql.functions import col,isnan,when,count,lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.ml.feature import Imputer
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import corr
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.feature import MinMaxScaler



spark = SparkSession.builder.appName("Genoprophet").getOrCreate();

df=spark.read.csv('gwas.csv', inferSchema=True, header=True)
# df=spark.read.csv('gwas.csv', inferSchema=True, header=True)
print("no of rows",df.count())
df.printSchema()

# Select the desired fields
selected_fields = ["initial sample size","snps","p-value", "pvalue_mlog", "or or beta", "95% ci (text)", "risk allele frequency", "context", "intergenic", "snp_id_current", "disease/trait"]
# selected_fields = ["initial sample size","snps","p-value", "pvalue_mlog", "or or beta", "95% ci (text)", "risk allele frequency", "context", "intergenic", "snp_id_current"]

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

# indexer = StringIndexer(inputCol="disease/trait", outputCol="target")
# indexer_model = indexer.fit(df)
# df = indexer_model.transform(df)
# df.show()

categ_cols = ['initial sample size', 'snps', '95% ci (text)', 'risk allele frequency', 'context', 'snp_id_current','disease/trait']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categ_cols]
indexer_pipeline = Pipeline(stages=indexers)
indexed_df = indexer_pipeline.fit(df).transform(df)



assembler = VectorAssembler(inputCols=numerical_df.columns + [col+"_index" for col in categ_cols], outputCol="features")
assembled_df = assembler.transform(indexed_df)

assembled_df.show()


scaler = MinMaxScaler(inputCol="features",\
         outputCol="scaledFeatures")
scalerModel =  scaler.fit(assembled_df.select("features"))
scaledData = scalerModel.transform(assembled_df)
scaledData.show()

from pyspark.ml.feature import PCA 


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

# Plotting the PCA curve
plt.plot(component_count, cumulative_variance, marker='o')
plt.axvline(num_components_95var, color='r', linestyle='--', label=f'{num_components_95var} components for 95% variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA Cumulative Variance Explained')
plt.legend()
plt.grid(True)
plt.show()

def extract(row):
    return tuple(row.pca_features.toArray().tolist())
final_data = pca_df.select("pca_features").rdd\
               .map(extract).toDF(df_selected.columns)
final_data.show()



# SMOTE



from imblearn.over_sampling import SMOTE
import numpy as np

# Convert PySpark DataFrame columns to NumPy arrays
features = np.array(final_data.drop("disease/trait").collect())
labels = np.array(final_data.select("disease/trait").collect(), dtype=str).flatten()

# Create an instance of the SMOTE class
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to balance the dataset
X, y = smote.fit_resample(features, labels)



# PLOT 



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


columns_without_target = [col for col in final_data.columns if col != "disease/trait"]
smote_df = pd.DataFrame(data=X, columns=columns_without_target)


# Add the balanced labels as a column
smote_df["disease/trait"] = y

# Plotting the distribution of classes
sns.countplot(x="disease/trait", data=smote_df)
plt.xticks(rotation=90)
plt.xlabel("Disease/Trait")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()

# Plotting a pairplot of features
sns.pairplot(smote_df, hue="disease/trait")
plt.suptitle("Pairplot of Features")
plt.show()



############  CLASSIFICATION  #############



# SVM



from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X = smote_df.drop(['disease/trait'],axis=1)
y = smote_df["disease/trait"]
#create subset of the dataset
# subset_size = 0.1
# X_subset, _, y_subset, _ = train_test_split(X, y, train_size=subset_size, random_state=42)
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc = SVC(C=1.0, random_state=42, kernel='linear')

# Fit the model
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Calculate the accuracy score
accuracy = accuracy_score(y_test, svc.predict(X_test))

# Print the accuracy score
print("Accuracy:", accuracy)



# DECISION TREE


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on the test data
y_pred = svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)


# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


###### DECISION TREE -ANOTHER METHOD ######


# # Convert the Pandas DataFrame to a PySpark DataFrame
# smote_spark_df = spark.createDataFrame(smote_df)

# # Index the "disease/trait" column
# indexer = StringIndexer(inputCol="disease/trait", outputCol="disease/trait_index")
# indexed_data = indexer.fit(smote_spark_df).transform(smote_spark_df)

# # Assemble features into a vector column
# feature_columns = [col for col in indexed_data.columns if col != "disease/trait" and col != "disease/trait_index"]
# assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
# assembled_data = assembler.transform(indexed_data)

# # Split the data into training and testing sets
# train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

# # Create a Decision Tree classifier
# dt_classifier = DecisionTreeClassifier(featuresCol="features", labelCol="disease/trait_index")

# # Train the model
# dt_model = dt_classifier.fit(train_data)

# # Make predictions
# dt_predictions = dt_model.transform(test_data)

# # Evaluate the model
# evaluator = MulticlassClassificationEvaluator(labelCol="disease/trait_index", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(dt_predictions)
# print("Accuracy:", accuracy)

##################################

###########   CLUSTERING   ###############


from pyspark.ml.clustering import KMeans


# Convert the Pandas DataFrame to a PySpark DataFrame
smote_spark_df = spark.createDataFrame(smote_df)

# Assemble features into a vector column
feature_columns = [col for col in smote_spark_df.columns if col != "disease/trait"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(smote_spark_df)

# Create a KMeans clustering model
k = 3  # Number of clusters
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)

# Fit the KMeans model
kmeans_model = kmeans.fit(assembled_data)

# Make predictions
clustered_data = kmeans_model.transform(assembled_data)

# Show the cluster assignments
clustered_data.select("cluster", "disease/trait").show()

