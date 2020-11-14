import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, expr}
import org.apache.spark.sql.types._

object recommender {

  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local").appName("ecommerce_recommendation").getOrCreate
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.shuffle.partitions",10)
    spark.conf.set("spark.port.maxRetries","100")
    val data = spark.read.format("csv").option("header", "true")
      .load("data/ratings_Beauty.csv")

    val distproduct = data.select("ProductId").distinct()
    val productIds = distproduct.rdd.map{case Row(item: String) => item}
      .zipWithIndex().toDF("ProductId", "UniqueProdID")
      .withColumn("UniqueProdID", col("UniqueProdID").cast(IntegerType))

    val distUser = data.select("UserId").distinct()
    val userIds = distUser.rdd.map{case Row(item: String) => item}
      .zipWithIndex().toDF("UserId", "UniqueUserId")
      .withColumn("UniqueUserId", col("UniqueUserId").cast(IntegerType))

    val userData = data.join(userIds,Seq("UserId"))
    val joinedData = userData.join(productIds, Seq("ProductId"))
    val finalData = joinedData.withColumn("Rating", col("Rating").cast(FloatType))
    val Array(training, test) = finalData.randomSplit(Array(0.7, 0.3))
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("UniqueUserId")
      .setItemCol("UniqueProdID")
      .setRatingCol("Rating")
    als.setColdStartStrategy("drop")
    val alsModel = als.fit(training)
    val predictions = alsModel.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("Rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    als.setNonnegative(true)
    val alsModel2 = als.fit(training)
    val predictions2 = alsModel2.transform(test)

    val rmse2 = evaluator.evaluate(predictions2)
    println(s"Root-mean-square error after Non negative set to true = $rmse2")

    als.setRegParam(0.1)
    val alsModel3 = als.fit(training)
    val predictions3 = alsModel3.transform(test)

    val rmse3 = evaluator.evaluate(predictions3)
    println(s"Root-mean-square error after reg param set to 0.1 = $rmse3")

    als.setRegParam(1)
    val alsModel4 = als.fit(training)
    val predictions4 = alsModel4.transform(test)

    val rmse4 = evaluator.evaluate(predictions4)
    println(s"Root-mean-square error after reg param set to 1 = $rmse4")
    
    val perUserActual = predictions4
      .where("rating > 2")
      .groupBy("UniqueUserId")
      .agg(expr("collect_set(UniqueProdID) as Products"))

    val perUserPredictions = predictions4
      .orderBy(col("UniqueUserId"), col("prediction").desc)
      .groupBy("UniqueUserId")
      .agg(expr("collect_list(UniqueProdID) as Products"))

    val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("UniqueUserId"))
      .map(row => (
        row(1).asInstanceOf[Seq[Integer]].toArray,
        row(2).asInstanceOf[Seq[Integer]].toArray.take(15)
      ))
    val ranks = new RankingMetrics(perUserActualvPred.rdd)
    val map = ranks.meanAveragePrecision
    val rankprec = ranks.precisionAt(5)
    println(s"Mean Average Precision = $map")
    println(s"Average Precision at Rank 5 = $rmse4")
    spark.stop()
  }
}
