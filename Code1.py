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

spark = SparkSession.builder.appName("Genoprophet").getOrCreate();

df=spark.read.csv('C:/Users/LENOVO/Desktop/Project/gwas-all-associations.csv', inferSchema=True, header=True).limit(10000)
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

indexer = StringIndexer(inputCol="disease/trait", outputCol="target")
indexer_model = indexer.fit(df)
df = indexer_model.transform(df)

categ_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categ_cols]

indexer_pipeline = Pipeline(stages=indexers)
