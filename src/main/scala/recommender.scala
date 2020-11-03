import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, expr}
import org.apache.spark.sql.types._

object recommender {

  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local").appName("ecommerce_recommendation").getOrCreate
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")
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

//    println("Product Id")
//    println(productIds.select("ProductId").distinct().count())
//    println(productIds.count())
//    println(productIds.agg(max("UniqueProdID")))
//    println(productIds.agg(max("UniqueProdID")).show())
//    println("User Id")
//    println(userIds.select("UserId").distinct().count())
//    println(userIds.count())
//    println(userIds.agg(max("UniqueUserId")))
//    println(userIds.agg(max("UniqueUserId")).show())
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
    println(predictions.show(5))

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("Rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")


    val regComparison = predictions.select("Rating", "prediction")
      .rdd.map(x => (x.getFloat(0).toDouble,x.getFloat(1).toDouble))
    val metrics = new RegressionMetrics(regComparison)
    
    val perUserActual = predictions
      .where("rating > 2.5")
      .groupBy("UniqueUserId")
      .agg(expr("collect_set(UniqueProdID) as Products"))

    val perUserPredictions = predictions
      .orderBy(col("UniqueUserId"), col("prediction").desc)
      .groupBy("UniqueUserId")
      .agg(expr("collect_list(UniqueProdID) as Products"))

    val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("UniqueUserId"))
      .map(row => (
        row(1).asInstanceOf[Seq[Integer]].toArray,
        row(2).asInstanceOf[Seq[Integer]].toArray.take(15)
      ))
    val ranks = new RankingMetrics(perUserActualvPred.rdd)
    ranks.meanAveragePrecision
    ranks.precisionAt(5)
  }
}
