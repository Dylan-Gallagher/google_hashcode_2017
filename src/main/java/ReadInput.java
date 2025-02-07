package main.java;

import java.io.*;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.*;
import java.util.ArrayList;
import java.util.List;


public class ReadInput {
    static class Endpoint {
        int id;
        int latencyToDataCenter;
        List<Connection> connections = new ArrayList<>();

        Endpoint(int id, int latencyToDataCenter) {
            this.id = id;
            this.latencyToDataCenter = latencyToDataCenter;
        }

        void addConnection(int cacheServerId, int latency) {
            connections.add(new Connection(cacheServerId, latency));
        }
    }

    static class Connection {
        int cacheServerId;
        int latency;

        Connection(int cacheServerId, int latency) {
            this.cacheServerId = cacheServerId;
            this.latency = latency;
        }
    }

    static class Request {
        int videoId;
        int endpointId;
        int numRequests;

        Request(int videoId, int endpointId, int numRequests) {
            this.videoId = videoId;
            this.endpointId = endpointId;
            this.numRequests = numRequests;
        }
    }

    List<Endpoint> endpoints = new ArrayList<>();
    List<Request> requests = new ArrayList<>();


    public Map<String, Object> data;

    // This function implements first choice hill climbing. It chooses the first neighbour it finds that is better than the current solution.
    public int[][] FirstChoiceHillClimbing(int[][] solution, ReadInput ri, int maxAttempts) {
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int bestScore = ri.fitness(solution, ri);
        Random random = new Random();

        boolean improvement = true;
        while (improvement) {
            improvement = false;
            int attempts = 0;

            while (attempts < maxAttempts) {
                int[][] newSolution = Arrays.stream(solution).map(int[]::clone).toArray(int[][]::new); // Deep copy of solution

                // Generate a random neighbor by flipping a random bit
                int i = random.nextInt(numCaches);
                int j = random.nextInt(numVideos);
                newSolution[i][j] ^= 1; // Flip the bit

                int currentScore = ri.fitness(newSolution, ri);
                if (currentScore > bestScore) {
                    bestScore = currentScore;
                    solution[i][j] = newSolution[i][j]; // Accept the new solution
                    improvement = true;
                    break; // First choice - exit after finding the first improvement
                }
                attempts++;
            }

            if (attempts >= maxAttempts) {
                // No improvement found after maxAttempts, stop the algorithm
                break;
            }
        }
        return solution;
    }

    public int[][] HillClimbing(int[][] solution, ReadInput ri) {
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int best_score = ri.fitness(solution, ri);
        boolean improvement = true;

        while (improvement) {
            improvement = false;
            int[] best_move = new int[]{-1, -1}; // Initialize with invalid indices
            boolean addVideo = true; // True if adding a video improves the solution, false if removing

            for (int i = 0; i < numCaches; i++) {
                for (int j = 0; j < numVideos; j++) {
                    int originalValue = solution[i][j]; // Save original value to revert if needed
                    solution[i][j] = 1 - originalValue; // Flip the bit to explore neighbor

                    int current_score = ri.fitness(solution, ri);
                    if (current_score > best_score) {
                        best_score = current_score;
                        best_move[0] = i;
                        best_move[1] = j;
                        addVideo = solution[i][j] == 1; // Update based on the successful flip
                        improvement = true;
                    } else {
                        solution[i][j] = originalValue; // Revert change if no improvement
                    }
                }
            }

            // Apply the best move found in this iteration
            if (improvement) {
                solution[best_move[0]][best_move[1]] = addVideo ? 1 : 0;
            }
        }
        return solution;
    }


    public int[][][] GenerateInitialPopulationRandom(int populationSize, ReadInput ri, double sensitivity) {
        // sensitivity controls how likely it is to create solutions with 1s in them. The higher the sensitivity, the
        //  more likely it is that "solutions" are not valid.
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int[][][] population = new int[populationSize][numCaches][numVideos];

        // Fill in each bit with a probability of sensitivity
        for (int i = 0; i < populationSize; i++) {
            do {
                for (int j = 0; j < numCaches; j++) {
                    for (int k = 0; k < numVideos; k++) {
                        population[i][j][k] = Math.random() < sensitivity ? 1 : 0;
                    }
                }
            } while (!isValidSolution(population[i], ri));
        }

        return population;
    }


    public int[][][] TournamentSelection(int[][][] population, int numParents, ReadInput ri) {
        Random rand = new Random();
        int populationSize = population.length;
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int[][][] parents = new int[numParents][numCaches][numVideos];
        Set<Integer> individualsConsidered = new HashSet<>();

        for (int i = 0; i < numParents; i++) {
            // Select two random individuals from the population
            int challengerOneIndex = rand.nextInt(populationSize);
            while (individualsConsidered.contains(challengerOneIndex)) {
                challengerOneIndex = rand.nextInt(populationSize);
            }
            int challengerTwoIndex = rand.nextInt(populationSize);
            while (challengerOneIndex == challengerTwoIndex || individualsConsidered.contains(challengerTwoIndex)) {
                challengerTwoIndex = rand.nextInt(populationSize);
            }
            individualsConsidered.add(challengerOneIndex);
            individualsConsidered.add(challengerTwoIndex);

            // Copy the data of the winning individual into the parents array
            int[][] winningIndividual = (fitness(population[challengerOneIndex], ri) > fitness(population[challengerTwoIndex], ri)) ? population[challengerOneIndex] : population[challengerTwoIndex];
            for (int j = 0; j < numCaches; j++) {
                System.arraycopy(winningIndividual[j], 0, parents[i][j], 0, numVideos);
            }
        }
        return parents;
    }


    private static int[][] deepCopy2DArray(int[][] original) {
        int[][] copy = new int[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = original[i].clone();
        }
        return copy;
    }

    public void calculateAndPrintAverageFitness(int[][][] population, ReadInput ri, int generation, String fileName) throws IOException {
        int totalFitness = 0;
        for (int[][] individual : population) {
            totalFitness += fitness(individual, ri);
        }
        int averageFitness = totalFitness / population.length;

        System.out.println("Generation " + generation + " average fitness: " + averageFitness);

        // Write to CSV
        FileWriter fw = new FileWriter(fileName + ".csv", true);
        PrintWriter out = new PrintWriter(fw);

        out.println(generation + "," + averageFitness);
        out.flush();
        out.close();
    }

    public int[][][] getEliteIndividuals(int[][][] population, int elitismCount, ReadInput ri) {
        // make a copy of the population
        int[][][] eliteIndividuals = new int[elitismCount][population[0].length][population[0][0].length];
        for (int i = 0; i < elitismCount; i++) {
            for (int j = 0; j < population[0].length; j++) {
                System.arraycopy(population[i][j], 0, eliteIndividuals[i][j], 0, population[0][0].length);
            }
        }
        // sort the elite by fitness
        Arrays.sort(eliteIndividuals, (a, b) -> fitness(b, ri) - fitness(a, ri));
        return eliteIndividuals;
    }
    public int[][] GeneticAlgorithm(int[][][] population, ReadInput ri, int generations, double mutationRate, double crossoverRate, int numParents, int elitismCount) {
        for (int i = 0; i < population.length; i++) {
            int fitness = fitness(population[i], ri);
            System.out.println("Individual " + i + " fitness: " + fitness);
            if (fitness < -1) {
                // print solution
                for (int j = 0; j < population[i].length; j++) {
                    for (int k = 0; k < population[i][0].length; k++) {
                        System.out.print(population[i][j][k] + " ");
                    }
                    System.out.println();
                }
            }
        }
        for (int generation = 0; generation < generations; generation++) {
            int[][][] newPopulation = new int[population.length][][];

            // Elitism: Keep the best individuals
            int[][][] eliteIndividuals = getEliteIndividuals(population, elitismCount, ri);

            if (elitismCount >= 0) System.arraycopy(eliteIndividuals, 0, newPopulation, 0, elitismCount);

            // Fill the rest with the next generation
            for (int i = elitismCount; i < population.length; i++) {
                int[][] parent1 = select(population, ri, numParents);
                int[][] parent2 = select(population, ri, numParents);
                int[][] child = crossover(parent1, parent2, crossoverRate);
                mutate(child, mutationRate);
                if (!isValidSolution(child, ri)) {
                    newPopulation[i] = population[i];
                } else {
                    newPopulation[i] = child;
                }
            }

            population = newPopulation;
            System.out.println("Generation " + generation);

        }
        // Sort the population by fitness
        Arrays.sort(population, (a, b) -> fitness(b, ri) - fitness(a, ri));

//        for (int i = 0; i < population.length; i++) {
//            System.out.println("Individual " + i + " fitness: " + fitness(population[i], ri));
//        }


        // Return the best individual
        return population[0];
    }

    private void mutate(int[][] individual, double mutationRate) {
        int numCaches = individual.length;
        int numVideos = individual[0].length;
        Random random = new Random();

        for (int i = 0; i < numCaches; i++) {
            for (int j = 0; j < numVideos; j++) {
                if (random.nextDouble() < mutationRate) {
                    individual[i][j] = individual[i][j] == 0 ? 1 : 0; // Flip bit
                }
            }
        }
    }

    public int[][] crossover(int[][] parent1, int[][] parent2, double crossoverRate) {
        // deep copy of parents
        int[][] parent1copy = Arrays.stream(parent1).map(int[]::clone).toArray(int[][]::new);
        int[][] parent2copy = Arrays.stream(parent2).map(int[]::clone).toArray(int[][]::new);

        int numCaches = parent1.length;
        int numVideos = parent1[0].length;
        Random random = new Random();
        if (random.nextDouble() < crossoverRate) {
            int crossoverPoint = random.nextInt(numVideos);
            for (int i = 0; i < numCaches; i++) {
                for (int j = crossoverPoint; j < numVideos; j++) {
                    int temp = parent1[i][j];
                    parent1copy[i][j] = parent2copy[i][j];
                    parent2copy[i][j] = temp;
                }
            }
        }
        return random.nextBoolean() ? parent1copy : parent2copy; // Randomly returns one of the offspring
    }

    public int[][] select(int[][][] population, ReadInput ri, int numParents) {
        int tournamentSize = numParents * 2;
        int[][][] tournament = new int[tournamentSize][][];
        Random rand = new Random();
        for (int i = 0; i < tournamentSize; i++) {
            int randomIndex = rand.nextInt(population.length);
            tournament[i] = population[randomIndex];
        }
        return Arrays.stream(tournament)
                .max((individual1, individual2) -> Integer.compare(fitness(individual1, ri), fitness(individual2, ri)))
                .get();
    }



    public ReadInput() {
        data = new HashMap<String, Object>();
    }

    public int fitness(int[][] solution, ReadInput ri) {
        // Check if the solution is valid; if not, return -1
        if (!isValidSolution(solution, ri)) {
            return -1;
        }

        // Initialize variables for score calculation
        long totalSavedTime = 0; // Total time saved by serving from cache servers
        long totalRequests = 0; // Total number of requests across all videos and endpoints

        // Extract necessary data from ReadInput object
        HashMap<String, String> requestsMap = (HashMap<String, String>) ri.data.get("video_ed_request");
        List<List<Integer>> epToCacheLatency = (List<List<Integer>>) ri.data.get("ep_to_cache_latency");
        List<Integer> epToDcLatency = (List<Integer>) ri.data.get("ep_to_dc_latency");
        List<List<Integer>> edCacheList = (List<List<Integer>>) ri.data.get("ed_cache_list");
        int[] videoSizes = (int[]) ri.data.get("video_size_desc");


        // For each element in video_ed_request
        for (Map.Entry<String, String> requestEntry : requestsMap.entrySet()) {
            /*
            If the file isn't in a cache connected to the endpoint, then no time is saved
            else, the time saved is equal to the cost of a data center request minus the cost of a cache request
             */

            String[] requestKeyParts = requestEntry.getKey().split(",");
            int videoID = Integer.parseInt(requestKeyParts[0]);
            int endpointID = Integer.parseInt(requestKeyParts[1]);
            int numberOfRequests = Integer.parseInt(requestEntry.getValue());

            // Add the number of requests to total requests
            totalRequests += numberOfRequests;

            // Calculate the baseline cost of serving from the data center
            int dataCenterLatency = epToDcLatency.get(endpointID);

            // Find the best cache server latency for this request
            int bestLatency = dataCenterLatency; // Start with data center latency as worst case
            for (int cacheServerID = 0; cacheServerID < solution.length; cacheServerID++) {
                // if cache is connected to the endpoint
                if (edCacheList.get(endpointID).contains(cacheServerID)) {
                    if (solution[cacheServerID][videoID] == 1) { // If video is in cache server
                        int cacheServerLatency = epToCacheLatency.get(endpointID).get(cacheServerID);
                        bestLatency = Math.min(bestLatency, cacheServerLatency);
                    }
                }
            }

            // Calculate saved time for this request and add it to the total saved time
            int savedTimeForRequest = (dataCenterLatency - bestLatency) * numberOfRequests;
            totalSavedTime += savedTimeForRequest;
        }

        // Calculate and return the average saved time per request
        if (totalRequests > 0) {
            return (int) (totalSavedTime / totalRequests) * 1000;
        } else {
            return 0; // Return 0 if there are no requests
        }
    }


    public void readGoogle(String filename) throws IOException {

        BufferedReader fin = new BufferedReader(new FileReader(filename));

        String system_desc = fin.readLine();
        String[] system_desc_arr = system_desc.split(" ");
        int number_of_videos = Integer.parseInt(system_desc_arr[0]);
        int number_of_endpoints = Integer.parseInt(system_desc_arr[1]);
        int number_of_requests = Integer.parseInt(system_desc_arr[2]);
        int number_of_caches = Integer.parseInt(system_desc_arr[3]);
        int cache_size = Integer.parseInt(system_desc_arr[4]);

        Map<String, String> video_ed_request = new HashMap<String, String>();
        String video_size_desc_str = fin.readLine();
        String[] video_size_desc_arr = video_size_desc_str.split(" ");
        int[] video_size_desc = new int[video_size_desc_arr.length];
        for (int i = 0; i < video_size_desc_arr.length; i++) {
            video_size_desc[i] = Integer.parseInt(video_size_desc_arr[i]);
        }

        List<List<Integer>> ed_cache_list = new ArrayList<List<Integer>>();
        List<Integer> ep_to_dc_latency = new ArrayList<Integer>();
        List<List<Integer>> ep_to_cache_latency = new ArrayList<List<Integer>>();
        for (int i = 0; i < number_of_endpoints; i++) {
            ep_to_dc_latency.add(0);
            ep_to_cache_latency.add(new ArrayList<Integer>());

            String[] endpoint_desc_arr = fin.readLine().split(" ");
            int dc_latency = Integer.parseInt(endpoint_desc_arr[0]);
            int number_of_cache_i = Integer.parseInt(endpoint_desc_arr[1]);
            ep_to_dc_latency.set(i, dc_latency);

            for (int j = 0; j < number_of_caches; j++) {
                ep_to_cache_latency.get(i).add(ep_to_dc_latency.get(i) + 1);
            }

            List<Integer> cache_list = new ArrayList<Integer>();
            for (int j = 0; j < number_of_cache_i; j++) {
                String[] cache_desc_arr = fin.readLine().split(" ");
                int cache_id = Integer.parseInt(cache_desc_arr[0]);
                int latency = Integer.parseInt(cache_desc_arr[1]);
                cache_list.add(cache_id);
                ep_to_cache_latency.get(i).set(cache_id, latency);
            }
            ed_cache_list.add(cache_list);
        }

        for (int i = 0; i < number_of_requests; i++) {
            String[] request_desc_arr = fin.readLine().split(" ");
            String video_id = request_desc_arr[0];
            String ed_id = request_desc_arr[1];
            String requests = request_desc_arr[2];
            video_ed_request.put(video_id + "," + ed_id, requests);
        }

        data.put("number_of_videos", number_of_videos);
        data.put("number_of_endpoints", number_of_endpoints);
        data.put("number_of_requests", number_of_requests);
        data.put("number_of_caches", number_of_caches);
        data.put("cache_size", cache_size);
        data.put("video_size_desc", video_size_desc);
        data.put("ep_to_dc_latency", ep_to_dc_latency);
        data.put("ep_to_cache_latency", ep_to_cache_latency);
        data.put("ed_cache_list", ed_cache_list);
        data.put("video_ed_request", video_ed_request);

        fin.close();
     }

     public String toString() {
        String result = "";

        //for each endpoint:
        for(int i = 0; i < (Integer) data.get("number_of_endpoints"); i++) {
            result += "enpoint number " + i + "\n";
            //latendcy to DC
            int latency_dc = ((List<Integer>) data.get("ep_to_dc_latency")).get(i);
            result += "latency to dc " + latency_dc + "\n";
            //for each cache
            for(int j = 0; j < ((List<List<Integer>>) data.get("ep_to_cache_latency")).get(i).size(); j++) {
                int latency_c = ((List<List<Integer>>) data.get("ep_to_cache_latency")).get(i).get(j);
                result += "latency to cache number " + j + " = " + latency_c + "\n";
            }
        }

        return result;
    }

    public int[][] createNaiveSolution(int numCaches, int numFiles) {
        int[][] solution = new int[numCaches][numFiles];

        // Initialise matrix to zero
        for (int i = 0; i < numCaches; i++) {
            for (int j = 0; j < numFiles; j++) {
                solution[i][j] = 0;
            }
        }

        return solution;
    }

    public int[][] createRandomSolution(int numCaches, int numFiles, double probabilityOfOne, Random random) {
        int[][] solution = new int[numCaches][numFiles];

        for (int i = 0; i < numCaches; i++) {
            for (int j = 0; j < numFiles; j++) {
                solution[i][j] = random.nextDouble() < probabilityOfOne ? 1 : 0;
            }
        }

        return solution;
    }

    public boolean isValidSolution(int[][] solution, ReadInput ri) {
        int cacheServerMaxCapacity = (int) ri.data.get("cache_size");
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");

        for (int i = 0; i < numCaches; i++) {
            int runningTotalFileSize = 0;
            for (int j = 0; j < numVideos; j++) {
                if (solution[i][j] == 1) {
                    runningTotalFileSize += ((int[]) ri.data.get("video_size_desc"))[j];
                }
                if (runningTotalFileSize > cacheServerMaxCapacity) {
                    return false;
                }
            }
        }

        return true;
    }


    public int[][] generateGreedyInitialSolution(ReadInput ri) {
        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int cacheSize = (int) ri.data.get("cache_size");
        int[][] solution = new int[numCaches][numVideos];

        // Video sizes and requests
        int[] videoSizes = (int[]) ri.data.get("video_size_desc");
        Map<String, String> videoRequests = (Map<String, String>) ri.data.get("video_ed_request");

        // Endpoint to cache latencies
        List<List<Integer>> epToCacheLatency = (List<List<Integer>>) ri.data.get("ep_to_cache_latency");

        // Convert video requests to cache server requests
        Map<Integer, Map<Integer, Integer>> cacheVideoRequests = new HashMap<>();
        for (String videoEndpoint : videoRequests.keySet()) {
            String[] parts = videoEndpoint.split(",");
            int videoId = Integer.parseInt(parts[0]);
            int endpointId = Integer.parseInt(parts[1]);
            int requestCount = Integer.parseInt(videoRequests.get(videoEndpoint));

            List<Integer> caches = epToCacheLatency.get(endpointId);
            for (int cacheId = 0; cacheId < caches.size(); cacheId++) {
                int latency = caches.get(cacheId);
                if (latency < ((List<Integer>) data.get("ep_to_dc_latency")).get(endpointId)) { // Assuming MAX_VALUE is used for non-connected caches
                    cacheVideoRequests.putIfAbsent(cacheId, new HashMap<>());
                    Map<Integer, Integer> videoRequestCounts = cacheVideoRequests.get(cacheId);
                    videoRequestCounts.put(videoId, videoRequestCounts.getOrDefault(videoId, 0) + requestCount);
                }
            }
        }

        // Greedily fill cache servers
        for (int cacheId = 0; cacheId < numCaches; cacheId++) {
            int currentCapacity = cacheSize;
            Map<Integer, Integer> videoRequestCounts = cacheVideoRequests.getOrDefault(cacheId, new HashMap<>());

            // Sort videos by request counts for this cache
            List<Map.Entry<Integer, Integer>> sortedVideos = new ArrayList<>(videoRequestCounts.entrySet());
            sortedVideos.sort((a, b) -> b.getValue().compareTo(a.getValue()));

            for (Map.Entry<Integer, Integer> entry : sortedVideos) {
                int videoId = entry.getKey();
                int videoSize = videoSizes[videoId];
                if (currentCapacity >= videoSize) {
                    solution[cacheId][videoId] = 1;
                    currentCapacity -= videoSize;
                }
            }
        }

        return solution;
    }


    public int[][] runGeneticAlgorithm (ReadInput ri, int populationSize, int generations, double mutationRate, double crossoverRate, int numParents, int elitismCount) {
        int[][][] population = GenerateInitialPopulationRandom(populationSize, ri, 0.03);
        return GeneticAlgorithm(population, ri, generations, mutationRate, crossoverRate, numParents, elitismCount);
    }



    public static void main(String[] args) throws IOException {
        ReadInput ri = new ReadInput();
        ri.readGoogle("input/me_at_the_zoo.in");

        int numCaches = (int) ri.data.get("number_of_caches");
        int numVideos = (int) ri.data.get("number_of_videos");
        int numSolutions = 6;

        int[][][] solutions = new int[numSolutions][numCaches][numVideos];
        int[] scores = new int[numSolutions];

        // Run the Island Model Genetic Algorithm, with each island running a genetic algorithm in parallel
        ExecutorService executor = Executors.newFixedThreadPool(numSolutions);
        for (int i = 0; i < numSolutions; i++) {
            int finalI = i;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    int[][][] population = ri.GenerateInitialPopulationRandom(250, ri, 0.03);
                    solutions[finalI] = ri.runGeneticAlgorithm(ri, 250, 1000, 0.001, 0.9, 25, 25);
                    scores[finalI] = ri.fitness(solutions[finalI], ri);
                }
            });
        }
        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < numSolutions; i++) {
            System.out.println("Solution " + i + " score: " + scores[i]);
        }

        // print the best solution
        int bestScore = Integer.MIN_VALUE;
        int bestSolutionIndex = -1;
        for (int i = 0; i < numSolutions; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestSolutionIndex = i;
            }
        }

        int fitnessBest = ri.fitness(solutions[bestSolutionIndex], ri);
        System.out.println("Best solution: " + fitnessBest);



    }
}
