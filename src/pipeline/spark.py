from pyspark.sql import SparkSession


def make_spark(
    app_name: str,
    driver_memory: str = "20g",
    local_cores: int = 6,
    shuffle_partitions: int = 16,
    default_parallelism: int = 16,
) -> SparkSession:
    master = f"local[{local_cores}]"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(default_parallelism))
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark



#
#
# def make_spark(
#     app_name: str,
#     driver_memory: str = "20g",
#     local_cores: int = 6,
#     shuffle_partitions: int = 16,
#     default_parallelism: int = 16,
# ) -> SparkSession:
#     master = f"local[{local_cores}]"
#
#     spark = (
#         SparkSession.builder
#         .appName(app_name)
#         .master(master)
#         .config("spark.driver.memory", driver_memory)
#         .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
#         .config("spark.default.parallelism", str(default_parallelism))
#         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
#         .getOrCreate()
#     )
#     spark.sparkContext.setLogLevel("WARN")
#     return spark
