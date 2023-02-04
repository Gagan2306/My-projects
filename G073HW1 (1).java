import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

public class G073HW1 {

    public static void main(String[] args) {

        // CHECKING NUMBER OF CMD LINE PARAMETERS

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions H_popularity S_country file_path");
        }

        // SPARK SETUP

        SparkSession conf = SparkSession
                .builder()
                .appName("Homework 1")
                .config("spark.master", "local")
                .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(conf.sparkContext());
        sc.setLogLevel("WARN");

        //Part 1

        // INPUT READING

        int K = Integer.parseInt(args[0]);
        int H = Integer.parseInt(args[1]);
        String S = args[2];

        JavaRDD<String> rawDataset = sc.textFile(args[3]).repartition(K).cache();

        // DISPLAY INPUT


        long numberOfRowsInTheRawDataset = rawDataset.count();
        System.out.println("Number of Rows = " + numberOfRowsInTheRawDataset);

        //Part 2
        JavaPairRDD<String, Integer> productCustomer = rawDataset.mapPartitionsToPair((document) -> {
            Set<Tuple2<String, Integer>> productCustomerList = new HashSet<>();
            while (document.hasNext()) {
                String currentDocument = document.next();
                String[] tokens = currentDocument.split(",");
                double productQuantity = Double.parseDouble(tokens[3]);
                String productId = tokens[1];
                String countryName = tokens[7];
                Integer customerId = Integer.parseInt(tokens[6]);

                if (productQuantity > 0) {
                    if (S.equals("all")) {
                        productCustomerList.add(new Tuple2<>(productId, customerId));
                    } else {
                        if (countryName.equals(S)) {
                            productCustomerList.add(new Tuple2<>(productId, customerId));
                        }
                    }
                }
            }
            return productCustomerList.iterator();
        });


        System.out.println("Product-Customer Pairs = " + productCustomer.distinct().collect().size());

        //Part 3

        JavaPairRDD<String, Integer> productPopularity1 =
                productCustomer.distinct().mapPartitionsToPair(input -> {
                            ArrayList<Tuple2<String, Integer>> productPopularity1List = new ArrayList<>();
                            HashMap<String, Integer> counts = new HashMap<>();
                            while (input.hasNext()) {
                                Tuple2<String, Integer> token = input.next();
                                counts.put(token._1(), 1 + counts.getOrDefault(token._1(), 0));
                            }
                            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                                productPopularity1List.add(new Tuple2<>(e.getKey(), e.getValue()));
                            }
                            return productPopularity1List.iterator();
                        })
                        .groupByKey()
                        .mapValues((element) -> {
                            int popularityCount = 0;

                            for (int popularity : element) {
                                popularityCount += popularity;
                            }
                            return popularityCount;
                        });


        //Part 4

        JavaPairRDD<String, Integer> productPopularity2 = productCustomer.distinct().repartition(K).mapToPair(input -> (new Tuple2<>(input._1(), 1)))
                .reduceByKey(Integer::sum, K)
                .reduceByKey(Integer::sum);


        //Part 5

        if (H > 0) {
            List<Tuple2<Integer, String>> listTopHPopularity = productPopularity1
                    .mapToPair(Tuple2::swap)
                    .sortByKey(false)
                    .take(H);

            System.out.println("Top " + H + " Products and their Popularities");
            for (Tuple2<Integer, String> res : listTopHPopularity)
                System.out.print("Product " + res._2() + " Popularity " + res._1() + "; ");
        }

        //Part 6
        else {
            System.out.println("productPopularity1:");
            displayPopularities(productPopularity1.distinct().sortByKey(true).collect());
            System.out.println("\nproductPopularity2:");
            displayPopularities(productPopularity2.distinct().sortByKey(true).collect());
        }

    }

    static void displayPopularities(List<Tuple2<String, Integer>> stringIntegerJavaPairRDD) {
        for (Tuple2<String, Integer> res : stringIntegerJavaPairRDD)
            System.out.print("Product: " + res._1() + " Popularity: " + res._2() + "; ");
    }


}
