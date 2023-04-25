/* 
 * PRL 2023 - Projekt 2
 * Author: Vojtech Fiala <xfiala61>
*/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <limits>
//#include <iomanip>

#define n_of_clusters 4
#define endless_cycle_prevention 30000

using namespace std;


/* Read the 1 byte large numbers from the input file `numbers` */
vector<uint8_t> readNumbers() {
    ifstream num_file ("numbers", ios::binary);
    vector<uint8_t> nums;
    if (!num_file) MPI_Abort(MPI_COMM_WORLD, 1); // error opening file

    uint8_t i;
    while (num_file.read((char*)&i, sizeof(uint8_t)))
	    nums.push_back(i);

    return nums;
}

/* Load numbers from file and check if there's a correct value of them */
vector<uint8_t> loadNumbers(int size) {

    vector<uint8_t> numbers = readNumbers();

    if (numbers.size() > size) { // If there's more numbers than the processes, cut them off
        numbers = {numbers.begin(), numbers.begin()+size};
    }
    else if (numbers.size() < size) { // If there's less numbers than processes, error
        fprintf(stderr, "There is less values than the number of processes!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    else if (numbers.size() < 4 || numbers.size() > 32) {
        fprintf(stderr, "Unsupported number of values!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return numbers;
}

/* Get distance between 2 points (point & a center) */
float calculateDistance(int n1, float n2) {
    return abs((float) n1 - n2);
}

/* Find the index of the closest centroid to a point in a vector of centers */
int closestIndex(int givenNumber, vector<float> centers) {

    int tmp_dist = INT32_MAX; // high initial value
    int closest = 0;

    for (int i = 0; i < n_of_clusters; i++) {
        int distance = calculateDistance(givenNumber, centers[i]);
        if (distance < tmp_dist) {
            tmp_dist = distance;
            closest = i;
        }
    }
    return closest;
}

/* Calculate number of values in each cluster */
pair<vector<int>, vector<int>> assignToCluster(int givenNumber, vector<float> centers) {

    int closest = closestIndex(givenNumber, centers);

    vector<int> points;
    points.resize(n_of_clusters);

    vector<int> values_in_cluster = {0,0,0,0};
    values_in_cluster[closest] += givenNumber;

    for (int i = 0; i < n_of_clusters; i++) {
        // set the value to 1 where the point belongs
        if (i == closest) {
            points[i] = 1;
        } // everywhere else, there is no point, so 0.
        else {
            points[i] = 0;
        }
    }
    return make_pair(points, values_in_cluster);
}

/* Calculate new center points as the average value of points in the current center */
vector<float> calculateNewCenters(vector<float> centers, vector<int> val_sum, vector<int> n_of_vals) {

    // go through each value and divide it by the number of numbers the value was made of 
    for (int i = 0; i < n_of_clusters; i++) {
        if (n_of_vals[i] == 0) { // if there is no value in this centroid, keep it as is
            continue;
        }
        float center_val = (float) val_sum[i] / (float) n_of_vals[i];
        centers[i] = center_val;
    }
    return centers;

}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<uint8_t> numbers;
    vector<float> centers; 

    if (rank == 0) {
        numbers = loadNumbers(size);

        // initialize the center points
        for (int i = 0; i < n_of_clusters; i++) {
            centers.push_back(numbers[i]);
        }
    }

    centers.resize(n_of_clusters); // make space for the center points
    MPI_Bcast(centers.data(), n_of_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD); // broadcast the first 4 center points (the first 4 numbers)

    uint8_t givenNumber; // each process will have its number here
    MPI_Scatter(numbers.data(), 1, MPI_UINT8_T, &givenNumber, 1, MPI_UINT8_T, 0, MPI_COMM_WORLD); // root sends each process its (only one) number

    vector<float> prev_centers = {-1.0, -1.0, -1.0, -1.0};

    int steps = 0;

    // prepare space for reduce operation results
    vector<int> global_cluster_members, total_value_for_cluster;
    if (rank == 0) {
        global_cluster_members.resize(n_of_clusters);
        total_value_for_cluster.resize(n_of_clusters);
    }

    vector<uint8_t> c1,c2,c3,c4; // 4 center vectors
    vector<vector<uint8_t>> clusters =  {c1,c2,c3,c4};

    // break the loop if the centers dont change anymore or if its over 30000 steps - in that case, something probably went wrong and this is a failsafe
    while ((prev_centers != centers) || (steps > endless_cycle_prevention)) {

        prev_centers = centers;
        // calculate which cluster the value will be assigned to and save it in a support vector that contains number of values in each cluster
        pair<vector<int>, vector<int>> values = assignToCluster(givenNumber, centers);

        // Calculate new centroids and total value of numbers in each centroid 
        // This could also be done with Allreduce and then each process would calculate the new centroids by itself.. 
        // I've decided to do it this way because I don't want all processes to do the same thing pointlessly when only 1 can do this
        MPI_Reduce(values.first.data(), global_cluster_members.data(), n_of_clusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(values.second.data(), total_value_for_cluster.data(), n_of_clusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) { // only root calculates the new centers
            centers = calculateNewCenters(centers, total_value_for_cluster, global_cluster_members);
        }
        MPI_Bcast(centers.data(), n_of_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD); // root broadcasts new centers
        steps++;
    }

    // final printing
    if (rank == 0) {
        vector<uint8_t> c1,c2,c3,c4; // 4 center vectors
        vector<vector<uint8_t>> clusters =  {c1,c2,c3,c4};
        // rank 0 already has all the numbers, so he will just assign them based on the centers
        for (auto &number : numbers) {
            int closest = closestIndex(number, centers);
            clusters[closest].push_back(number);
        }

        int i = 0;
        for (auto& cluster : clusters) {
            // cout << std::fixed << setprecision(4) << "[" << centers[i] << "] ";
            cout << "[" << centers[i] << "] ";
            auto lim = cluster.size();
            for (int k = 0; k < lim; k++) {
                // last one doesnt print the final comma
                if (k == (lim-1)) {
                    cout << static_cast<int>(cluster[k]);
                }
                else {
                    cout << static_cast<int>(cluster[k]) << ", ";
                }
            }
            cout << endl;
            i++;
        }
    }

    MPI_Finalize();

}
