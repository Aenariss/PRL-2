from sklearn.cluster import KMeans 
import numpy as np
import sys
import subprocess

# Credits for this go to Monnte


filename = "numbers"
n_clusters = 4
num_of_tests = 100
num_of_numbers = 32
file_name_to_check = "output.txt"
file_name_to_check_with = "output_with.txt"
fail = 0
if __name__ == "__main__":

    for i in range(num_of_tests):
        print(f"Test {i+1}/{num_of_tests}...")
        subprocess.run(["dd", "if=/dev/random", "bs=1", f"count={num_of_numbers}", f"of={filename}", "status=none"])
        data = np.fromfile(filename,dtype=np.dtype('B'))

        centroids = np.array([data[0], data[1], data[2], data[3]]).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(data.reshape(-1, 1))
        
        with open(file_name_to_check, "w", newline="\r") as f:
            for i in range(n_clusters):
                centroid = kmeans.cluster_centers_[i][0]
                indices = np.where(kmeans.labels_ == i)[0]
                f.write(f"[{centroid:.4f}]")
                for j in range(len(indices)):
                    f.write(f" {data[indices[j]]}")
                f.write("\n")

        with open(file_name_to_check_with, "w", newline="\r") as f:
            subprocess.run(["mpirun", "--prefix", "/usr/local/share/OpenMPI", "-oversubscribe", "-np", f"{num_of_numbers}", "parkmeans"], stdout=f)

        with open(file_name_to_check, "r") as f1, open(file_name_to_check_with, "r") as f2:
            for line1, line2 in zip(f1, f2):
                line2 = line2.replace(',', '')
                print(line1, line2)
                if line1.strip() != line2.strip():
                    print("Test failed")
                    fail += 1
        print("Test passed")

print("Passed:", num_of_tests-fail, "Fail: ", fail)