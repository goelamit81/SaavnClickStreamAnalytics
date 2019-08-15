package com.ml.upgrad.saavnanalytics;

import scala.collection.Seq;
import static org.apache.spark.sql.functions.col;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.recommendation.ALS;

public class SaavnAnalyticsKMeans {

	/*
	 * UDF to convert array of features obtained from ALS into a Vector
	 */
	public static UDF1<Seq<Float>, Vector> convertToVector = new UDF1<Seq<Float>, Vector>() {

		private static final long serialVersionUID = 1L;

		public Vector call(Seq<Float> t1) throws Exception {

			List<Float> L = scala.collection.JavaConversions.seqAsJavaList(t1);
			double[] doubleArray = new double[t1.length()];
			for (int i = 0; i < L.size(); i++) {
				doubleArray[i] = L.get(i);
			}
			return Vectors.dense(doubleArray);
		}
	};

	public static void main(String[] args) {

		/*
		 * Check for input paramters. There should be 6 input parameters to the jar file: 
		 * 1. Checkpoint directory path on hadoop hdfs : (/user/ec2-user/checkpoint_dir) 
		 * 2. Input path to activity data : (s3a://bigdataanalyticsupgrad/activity/sample100mb.csv) 
		 * 3. Input path to new metadata : (s3a://bigdataanalyticsupgrad/newmetadata/*) 
		 * 4. Input path to notification clicks : (s3a://bigdataanalyticsupgrad/notification_clicks/*) 
		 * 5. Input path to notification artists : (s3a://bigdataanalyticsupgrad/notification_actor/*) 
		 * 6. Output directory path on hadoop hdfs : (/user/ec2-user/saavnanalytics)
		 */
		if (args.length < 6) {
			System.out.println("Please specify all 6 parameters");
			System.out.println(
					"Provide checkpoint dir path, input path to activity data, metadata, notification clicks data "
					+ "and notification artists data, output directory path");
			return;
		} else {

			// Setup logging to error only
			Logger.getLogger("org").setLevel(Level.ERROR);
			Logger.getLogger("akka").setLevel(Level.ERROR);

			// Setup SparkConf object
			SparkConf conf = new SparkConf().setAppName("SaavnAnalyticsCaseStudy");

			// Setup SparkContext
			SparkContext context = new SparkContext(conf);

			// Setup checkpoint directory
			context.setCheckpointDir(args[0]);

			// Create the spark session
			SparkSession spark = SparkSession.builder().appName("SaavnAnalyticsCaseStudy").getOrCreate();

			// Setup SQL context
			SQLContext sc = spark.sqlContext();

			// Print Start time
			System.out
					.println("\nStart Time : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()) + "\n");

			// Load User Activity Data into dataframe
			System.out.println("\nLoading user activity data....\n");
			Dataset<Row> userActivityData = spark.read().option("header", "false").csv(args[1])
					.select(col("_c0").as("UserID"), col("_c2").as("SongID")).na().drop();

			// Create temporary view on userActivityData
			userActivityData.createOrReplaceTempView("useractivitydata");

			// Load New Meta Data into dataframe
			System.out.println("\nLoading new metadata....\n");
			Dataset<Row> newMetaData = spark.read().option("header", "false").csv(args[2])
					.select(col("_c0").as("SongID"), col("_c1").as("ArtistID")).na().drop();

			// Create temporary view on newMetaData
			newMetaData.createOrReplaceTempView("newmetadata");

			// Load Notification Clicks Data into dataframe
			System.out.println("\nLoading notification clicks data....\n");
			Dataset<Row> notificationClickData = spark.read().option("header", "false").csv(args[3])
					.select(col("_c0").as("NotificationID"), col("_c1").as("UserID")).na().drop();

			// Create temporary view on notificationClickData
			notificationClickData.createOrReplaceTempView("notificationclickdata");

			// Load Notification Artists Data into dataframe
			System.out.println("\nLoading notification artists data....\n");
			Dataset<Row> notificationArtistData = spark.read().option("header", "false").csv(args[4])
					.select(col("_c0").as("NotificationID"), col("_c1").as("ArtistID")).na().drop();

			// Create temporary view on notificationArtistData
			notificationArtistData.createOrReplaceTempView("notificationartistdata");

			// Calculate frequency of a song listened by a user
			Dataset<Row> activityDataFrequency = sc
					.sql("select UserID, SongID, count(*) as Frequency from useractivitydata group by UserID, SongID");

			// This will convert the String values of UserID to numeric
			StringIndexer userIndexer = new StringIndexer().setInputCol("UserID").setOutputCol("userIndex");

			System.out.println("Changing UserID to numeric...\n");
			Dataset<Row> userIndexed = userIndexer.fit(activityDataFrequency).transform(activityDataFrequency);

			// This will convert the String values of SongID to numeric
			StringIndexer songIndexer = new StringIndexer().setInputCol("SongID").setOutputCol("songIndex");

			System.out.println("Changing SongID to numeric...\n");
			Dataset<Row> indexedFinal = songIndexer.fit(userIndexed).transform(userIndexed);

			// Use ALS algorithm for implicit learning and obtain features from it
			System.out.println("Starting the ALS algorithm to get features from implicit learning....\n");
			
			ALS als = new ALS().setRank(10).setMaxIter(10).setImplicitPrefs(true).setUserCol("userIndex")
					.setItemCol("songIndex").setRatingCol("Frequency").setSeed(46L);

			ALSModel model = als.fit(indexedFinal);

			Dataset<Row> userImplicitFactors = model.userFactors();

			/*
			 * Received the implicit factors from ALS. However, it is in array format. This
			 * needs to be changed into vector. Use toVector UDF for the same.
			 */
			System.out.println(
					"\nRecieved implicit factors from ALS. However, it is in array format. "
					+ "Working to change it into Vector now...\n");
			
			spark.udf().register("toVector", convertToVector, new VectorUDT());
			
			Dataset<Row> userFactorsAsVectors = userImplicitFactors.withColumn("features",
					functions.callUDF("toVector", userImplicitFactors.col("features")));

			// Use K-means algorithm to form clusters for users
			System.out.println("Starting K-means algorithm to form clusters....\n");

			KMeans KM = new KMeans().setK(20000).setMaxIter(10).setSeed(46L);
			
			KMeansModel KMmodel = KM.fit(userFactorsAsVectors);
			
			Dataset<Row> userClusters = KMmodel.transform(userFactorsAsVectors);

			// Evaluate clustering by computing Silhouette score
			ClusteringEvaluator evaluator = new ClusteringEvaluator();

			double silhouette = evaluator.evaluate(userClusters);
			System.out.println("Silhouette with squared euclidean distance = " + silhouette + "\n");

			// Cast id as DoubleType
			userClusters = userClusters.select(col("id").cast(DataTypes.DoubleType), col("prediction"));

			// Create temporary view on userClusters and input dataframe
			userClusters.createOrReplaceTempView("userclusters");
			indexedFinal.createOrReplaceTempView("indexedfinal");

			/*
			 * Join input dataframe with userClusters on userIndex and id in the inner query
			 * and obtain UserID, SongID and prediction as ClusterID. Join this output with
			 * newmetadata on SongID and obtain UserID, ClusterID, SongID and ArtistID.
			 */
			Dataset<Row> clusteredData = sc
					.sql("select cluster.UserID, cluster.ClusterID, cluster.SongID, n.ArtistID from"
							+ " (select i.UserID, i.SongID, u.prediction as ClusterID from"
							+ " indexedfinal i, userclusters u where i.userIndex = u.id) cluster,"
							+ " newmetadata n where cluster.SongID = n.SongID");

			// Create temporary view on clusteredData
			clusteredData.createOrReplaceTempView("clustereddata");

			/*
			 * Join clustereddata and notificationartistdata on ArtistID to get
			 * distinct UserID, ClusterID, ArtistID and NotificationID for given
			 * notification ids as per problem statement. This is our
			 * NotificationNumber1 data which has original ClusterID and
			 * ArtistID for given NotificationIDs along with UserID and
			 * NotificationID.
			 */
			Dataset<Row> notificationNumber1 = sc.sql(
					"select distinct a.UserID, a.ClusterID, a.ArtistID, b.NotificationID from clustereddata a,"
					+ " notificationartistdata b where a.ArtistID = b.ArtistID and"
					+ " b.NotificationID in (9553, 9660, 9690, 9703, 9551)");

			// Save NotificationNumber1 data to a file
			System.out.println(
					"\nSaving NotificationNumber1 data for specific notification ids as per problem statement, to a file\n");

			notificationNumber1.write().mode(SaveMode.Overwrite).format("csv").option("header", "true")
					.save(args[5] + "/" + "NotificationNumber1");

			/*
			 * Select ClusterID and ArtistID from clustereddata and compute count by
			 * grouping on ClusterID and ArtitID as cnt_artists in innermost query. Then in
			 * immediate outer query, select ClusterID, ArtistID and use analytic function
			 * row_number() over partition by clause to partition data on ClusterID order by
			 * cnt_artists in descending order so ArtistID with highest count gets 1 as row
			 * number. Then in immediate outer query, select ClusterID, ArtistID and use
			 * analytic function first_value on ClusterID over partition by clause to
			 * partition on ArtistID order by ClusterID as CommonClusterID, from inner query
			 * output where row number = 1. Finally select ClusterID, form new ClusterID
			 * using CommnClusterID suffixed with '_common', ArtistID as PopularArtistID.
			 * 
			 * After this step, we will have formed new common clusters with popular artists
			 * for each cluster and 1 artist is associated with only 1 new common cluster.
			 */
			Dataset<Row> popularArtistWithNewCluster = sc.sql(
					"select ClusterID, CommonClusterID||'_common' as NewCommonClusterID, ArtistID as PopularArtistID from"
							+ " (select ClusterID, ArtistID, first_value(ClusterID) over"
							+ " (partition by ArtistID order by ClusterID) as CommonClusterID from"
							+ " (select ClusterID, ArtistID, row_number() over"
							+ " (partition by ClusterID order by cnt_artists desc) rn"
							+ " from (select ClusterID, ArtistID, count(*) as cnt_artists from clustereddata"
							+ " group by ClusterID, ArtistID)) where rn = 1)");

			// Create temporary view on popularArtistWithNewCluster
			popularArtistWithNewCluster.createOrReplaceTempView("popularartistwithnewcluster");

			/*
			 * Join popularartistwithnewcluster with clustereddata on ClusterID
			 * to get distinct UserID, NewCommonCluserID and PopularArtistID.
			 * This is our UserClusterArtist data.
			 */
			Dataset<Row> UserClusterArtist = sc.sql(
					"select distinct a.UserID, b.NewCommonClusterID, b.PopularArtistID from clustereddata a,"
					+ " popularartistwithnewcluster b where a.ClusterID = b.ClusterID");

			// Save UserClusterArtist data to a file
			System.out.println("\nSaving UserClusterArtist data to a file\n");

			UserClusterArtist.write().mode(SaveMode.Overwrite).format("csv").option("header", "true")
					.save(args[5] + "/" + "UserClusterArtist");

			// Create temporary view on UserClusterArtist
			UserClusterArtist.createOrReplaceTempView("userclusterartist");

			/*
			 * Join popularartistwithnewcluster with notificationartistdata on
			 * PopularArtistID and ArtistID to get distinct NewCommonCluserID,
			 * PopularArtistID and NotificaitonID.
			 */
			Dataset<Row> popularArtistNotification = sc.sql(
					"select distinct a.NewCommonClusterID, a.PopularArtistID, b.NotificationID from"
					+ " popularartistwithnewcluster a, notificationartistdata b"
					+ " where a.PopularArtistID = b.ArtistID");

			// Create temporary view on popularArtistNotification
			popularArtistNotification.createOrReplaceTempView("popularartistnotification");

			/*
			 * Join userclusterartist with popularartistnotification on
			 * PopularArtistID and NewCommonClusterID to get distinct UserID,
			 * NewCommonClusterID, PopularArtistID and NotificationID for given
			 * notification ids as per problem statement. This is our
			 * NotificationNumber2 data for given NotificationIDs.
			 */
			Dataset<Row> notificationNumber2 = sc.sql(
					"select distinct a.UserID, a.NewCommonClusterID, a.PopularArtistID, b.NotificationID from userclusterartist a,"
					+ " popularartistnotification b where a.NewCommonClusterID = b.NewCommonClusterID and "
					+ " a.PopularArtistID = b.PopularArtistID and b.NotificationID in (9553, 9660, 9690, 9703, 9551)");

			// Save NotificationNumber2 data to a file
			System.out.println(
					"\nSaving NotificationNumber2 data for specific notification ids as per problem statement, to a file\n");

			notificationNumber2.write().mode(SaveMode.Overwrite).format("csv").option("header", "true")
					.save(args[5] + "/" + "NotificationNumber2");

			/*
			 * Push notifications now. Get distinct UserID and NotificationID by
			 * joining userclusterartist and popularartistnotification on
			 * PopularArtistID and NewCommonClusterID. By doing so, if a
			 * notification is for multiple artists and those artists are
			 * popular in different clusters, then all users for all such
			 * clusters will be picked up for that notification and we will be
			 * able to report common CTR for that notification.
			 */
			Dataset<Row> userNotificationData = sc.sql(
					"select distinct a.UserID, b.NotificationID from userclusterartist a, popularartistnotification b"
							+ " where a.PopularArtistID = b.PopularArtistID and a.NewCommonClusterID = b.NewCommonClusterID");

			// Create temporary view on userNotificationData
			userNotificationData.createOrReplaceTempView("usernotificationdata");

			/*
			 * Get NotificationID and count from usernotificationdata data group by
			 * NotificationID. This will give us count of users where notifications were
			 * pushed.
			 */
			Dataset<Row> pushedUserCount = sc.sql(
					"select NotificationID, count(*) as pushedCount from usernotificationdata group by NotificationID");

			// Create temporary view on pushedUserCount
			pushedUserCount.createOrReplaceTempView("pushedusercount");

			/*
			 * Join usernotificationdata with notificationclickdata on UserID and
			 * NotificationID and get count of NotificationIDs group by NotitificationID.
			 * This will give us count of users who clicked pushed notifications.
			 */
			Dataset<Row> clickedUserCount = sc.sql(
					"select NotificationID, count(*) as clickedCount from (select distinct b.NotificationID, b.UserID from "
							+ " usernotificationdata a, notificationclickdata b"
							+ " where a.UserID = b.UserID and a.NotificationId = b.NotificationID) group by NotificationID");

			// Create temporary view on clickedUserCount
			clickedUserCount.createOrReplaceTempView("clickedusercount");

			/*
			 * Compute CTR - (count of users who clicked pushed notifications / count of
			 * users receiving pushed notifications) multiply by 100, by joining
			 * pushedusercount and clickedusercount on NotificationID.
			 */
			Dataset<Row> ctrData = sc.sql(
					"select a.NotificationID, (b.clickedCount/a.pushedCount) * 100 as CTR from pushedusercount a,"
					+ " clickedusercount b where a.NotificationID = b.NotificationID");

			// Save CTR data in a file
			System.out.println("\nSaving CTR data to a file\n");

			ctrData.coalesce(1).write().mode(SaveMode.Overwrite).format("csv").option("header", "true")
					.save(args[5] + "/" + "CTRData");

			// Create temporary view on ctrData
			ctrData.createOrReplaceTempView("ctrdata");

			// Get CTR for specific notification ids mentioned in the problem statement
			Dataset<Row> ctrDataSpecific = sc.sql(
					"select NotificationID, CTR from ctrdata where NotificationID in (9553, 9660, 9690, 9703, 9551)");

			// Save CTR data for specific notification ids in a file
			System.out.println("\nSaving CTR data for specific notification ids as per problem statement, to a file\n");

			ctrDataSpecific.coalesce(1).write().mode(SaveMode.Overwrite).format("csv").option("header", "true")
					.save(args[5] + "/" + "CTRSpecificNotificationData");

			// Drop all temporary views created so far
			spark.catalog().dropTempView("useractivitydata");
			spark.catalog().dropTempView("newmetadata");
			spark.catalog().dropTempView("notificationclickdata");
			spark.catalog().dropTempView("notificationartistdata");
			spark.catalog().dropTempView("userclusters");
			spark.catalog().dropTempView("indexedfinal");
			spark.catalog().dropTempView("clustereddata");
			spark.catalog().dropTempView("popularartistwithnewcluster");
			spark.catalog().dropTempView("userclusterartist");
			spark.catalog().dropTempView("popularartistnotification");
			spark.catalog().dropTempView("usernotificationdata");
			spark.catalog().dropTempView("pushedusercount");
			spark.catalog().dropTempView("clickedusercount");
			spark.catalog().dropTempView("ctrdata");

			// Print End time
			System.out.println("\nEnd Time : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()) + "\n");

		}
	}

}
