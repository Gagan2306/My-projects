import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class G073HW2 {

    private static double initialGuessR;
    private static double finalGuessR;
    private static int numberOfGuesses = 1;

    public static void main(String[] args) {

        // fetching file path from CLI arguments
        String filePath = args[0];

        // the number of centers
        int k = Integer.parseInt(args[1]);

        // the number of outliers
        int z = Integer.parseInt(args[2]);

        ArrayList<Vector> inputPoints;
        try {
            inputPoints = readVectorsSeq(filePath);
        } catch (IOException e) {
            throw new IllegalArgumentException("Incorrect file path");
        }

        ArrayList<Long> weights = new ArrayList<>();

        // initialize with array list of weights with all 1's
        for (int i = 0; i < inputPoints.size(); i++) {
            weights.add(1L);
        }

        long startingTime = System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long endingTime = System.currentTimeMillis();
        long elapsedTime = endingTime - startingTime;
        double objective = ComputeObjective(inputPoints, solution, z);

        // output of HW2
        System.out.println("Input size n = " + inputPoints.size() + "\nNumber of centers k = " + k + "\nNumber of outliers z = " + z
                + "\nInitial guess = " + initialGuessR + "\nFinal guess = " + finalGuessR + "\nNumber of guesses = " + numberOfGuesses +
                "\nObjective function = " + objective + "\nTime of SeqWeightedOutliers = " + elapsedTime);
    }

    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, double alpha) {

        ArrayList<Vector> listOfCenters = new ArrayList<>();
        double r = Math.sqrt(Vectors.sqdist(P.get(0), P.get(1)));

        for (int i = 0; i < (k + z + 1); i++) {
            for (int j = 0; j < (k + z + 1); j++) {
                if (j != i) {
                    double distance = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
                    if (distance < r) {
                        r = distance;
                    }
                }
            }
        }

        r = r / 2;
        initialGuessR = r;

        while (true) {

            ArrayList<Vector> Z = new ArrayList<>(P);

            Set<Vector> S = new HashSet<>();

            double innerRadius = (1 + 2 * alpha) * r;

            double outerRadius = (3 + 4 * alpha) * r;

            ArrayList<Vector> bZ;

            HashMap<Vector, Long> bZToWeightHashMapPairs;

            long wZ = 0;
            for (int j = 0; j < P.size(); j++) {
                wZ += W.get(j);
            }

            Vector newCenter = null;

            while ((S.size() < k) && (wZ > 0)) {
                long max = 0;

                for (Vector x : P) {
                    long ballWeight = 0;
                    int i = 0;
                    for (Vector y : Z) {
                        double distance = Math.sqrt(Vectors.sqdist(x, y));
                        if (distance <= innerRadius) {
                            ballWeight += W.get(i);
                        }
                        i++;
                    }

                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenter = x;
                    }
                }

                S.add(newCenter);

                bZ = new ArrayList<>();
                bZToWeightHashMapPairs = new HashMap<>();

                int t = 0;

                for (Vector v : Z) {

                    double distance = 0;
                    if (newCenter != null) {
                        distance = Math.sqrt(Vectors.sqdist(newCenter, v));
                    }

                    if (distance <= outerRadius) {
                        bZToWeightHashMapPairs.put(v, W.get(t));
                        bZ.add(v);
                    }

                    t++;
                }

                for (Vector y : bZ) {
                    Z.remove(y);
                    wZ -= bZToWeightHashMapPairs.get(y);
                }
            }

            if (wZ <= z) {
                finalGuessR = r;
                listOfCenters.addAll(S);
                return listOfCenters;
            } else {
                r = 2 * r;
                numberOfGuesses++;
            }
        }
    }

    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z) {

        double maximumDistanceFromCenterToPoint = 0;
        ArrayList<Double> listOfDistances = new ArrayList<>();

        for (Vector x : P) {
            double minimumDistance = Math.sqrt(Vectors.sqdist(x, S.get(0)));
            for (Vector s : S) {
                double currentDistance = Math.sqrt(Vectors.sqdist(x, s));
                if (currentDistance < minimumDistance) {
                    minimumDistance = currentDistance;
                }
            }
            listOfDistances.add(minimumDistance);
        }

        Collections.sort(listOfDistances);

        maximumDistanceFromCenterToPoint = listOfDistances.get(listOfDistances.size() - z - 1);
        return maximumDistanceFromCenterToPoint;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }
}