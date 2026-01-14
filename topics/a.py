# import berteley
# import berteley.models
# import berteley.preprocessing
# import berteley.recommendation

# help(berteley.models)
# help(berteley.recommendation)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
print("Spark OK")
import sparknlp
print("Imported sparknlp")

