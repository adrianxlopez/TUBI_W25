import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from kneed import KneeLocator  # For the Elbow Method
import time
import csv

class ClusteringPipeline:
    def __init__(self, metadata_file, coordinates_file, max_iter=300, tolerance=1e-4, n_init = 10):
        """Initialize the clustering pipeline with dataset paths and clustering parameters."""
        # Load Data
        self.metadata = pd.read_csv(metadata_file, delimiter="\t")
        self.xyCoord = pd.read_csv(coordinates_file, delimiter="\t")
    
        self.coordinateFile = coordinates_file
        # Ensure X is a 2D array with x, y coordinates
        self.X = self.xyCoord[['x', 'y']].values

        # Define possible cluster range (avoid zero and excessive values)
        self.max_k = len(self.metadata['disease'].unique()) # Limit max_k to available unique diseases
        self.range_n_cluster = list(range(2, self.max_k))  # Range from 2 to max_k

        #parameters for KMeans params 
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_init = n_init

        # Store results
        self.cluster_results = {}
        self.top_clusters = []
        self.elbow_k = None

    def elbow_analysis(self, displayPlot = False):
        """Finds the optimal k using the Elbow Method."""
        wcss = []

        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=self.n_init, max_iter=self.max_iter, tol=self.tolerance)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)

        # Detect the elbow point automatically
        kl = KneeLocator(range(1, self.max_k + 1), wcss, curve="convex", direction="decreasing")
        self.elbow_k = kl.elbow
        if self.elbow_k is None:
            print("Elbow point could not be found ...")
            return
        print(f"Optimal k (Elbow Method): {self.elbow_k}")

        if displayPlot:
            # Plot WCSS to visualize the elbow
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, self.max_k + 1), wcss, marker="o", linestyle="--")
            plt.axvline(x=self.elbow_k, color="r", linestyle="--", label=f"Optimal k = {self.elbow_k}")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
            plt.title("Elbow Method for Optimal k")
            plt.legend()
            plt.show()

    def silhouette_analysis(self):
        """Finds the best number of clusters using silhouette analysis."""
        silhouette_averages = []

        for n_clusters in self.range_n_cluster:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=self.n_init, max_iter=self.max_iter, tol=self.tolerance)
            cluster_labels = kmeans.fit_predict(self.X)

            # Compute silhouette score
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            if silhouette_avg < 0.60:
                continue
            silhouette_averages.append((silhouette_avg, n_clusters))

        # Sort silhouette scores and store the top 5 best choices
        self.top_clusters = sorted(silhouette_averages, reverse=True, key=lambda x: x[0])[:5]

        print("\nTop 5 cluster choices based on Silhouette Scores and Elbow Plots:")
        for rank, (score, k) in enumerate(self.top_clusters, 1):
            print(f"   {rank}. k = {k}, Silhouette Score = {score:.3f}")

    def cluster(self, num_clusters):
        """Performs KMeans clustering and stores the result."""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=self.n_init, max_iter=self.max_iter, tol=self.tolerance)
        self.xyCoord['cluster'] = kmeans.fit_predict(self.X)
        self.cluster_results[num_clusters] = self.xyCoord.copy()  # Store clustered data
        return kmeans.cluster_centers_

    def plot(self, num_clusters, centroids, rank):
        """Plots the clustered data with centroids."""
        data = self.cluster_results[num_clusters]

        plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
        plt.xlabel("x")
        plt.ylabel("y")

        # Annotate centroids
        for i, (x, y) in enumerate(centroids):
            plt.text(x, y+10, f'Cluster {i+1}', fontsize=12, ha='center', va='center', color='red', alpha=1)

        plt.title(f"{rank} Best Silhouette Analysis for KMeans with n_clusters = {num_clusters}")
        plt.show()

    def export_clusters(self):
        """Exports the cluster labels along with x, y coordinates to a tab-separated file, allowing user selection of cluster count."""
        
        if not self.cluster_results:
            print("No clustered data available. Run clustering first!")
            return
        
        # Get available cluster counts
        available_clusters = sorted(self.cluster_results.keys())
        
        # Ask user for the cluster count to use
        print(f"\nAvailable cluster options: {available_clusters}")
        while True:
            try:
                num_clusters = int(input(f"Enter the number of clusters you want to export (from above options): "))
                if num_clusters in available_clusters:
                    break
                elif num_clusters == 0:
                    return
                else:
                    print("Invalid selection. Please enter a valid cluster count from the list.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

        # Extract the base name of the coordinates file (without extension)
        base_filename = os.path.splitext(os.path.basename(self.coordinateFile))[0]
        output_filename = f"{base_filename}_clusters_{num_clusters}.tab"

        # Create directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, output_filename)

        # Get the selected cluster results
        final_df = self.cluster_results[num_clusters][["#ID", "cluster"]]
        # Increment all cluster values by 1 and convert to string
        final_df["cluster"] = "cluster" + (final_df["cluster"] + 1).astype(str)
        
        # Export to a .tab file
        final_df.to_csv(output_path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
        print("\nSucess!\n")
        print(f"Cluster results (k={num_clusters}) saved to: {output_path}")

    
    def run_pipeline(self, method="both", exportFile = False, displayPlots = False):
        """Executes the full clustering pipeline based on the chosen method."""
        # Run selected method(s)
        if method in ["elbow", "both"]:
            print("\nRunning Elbow Method...")
            self.elbow_analysis()

        if method in ["silhouette", "both"]:
            print("\nRunning Silhouette Analysis...")
            self.silhouette_analysis()

        # Choose clustering options based on method
        cluster_choices = []  # Use a list instead of a set to maintain order

        if method in ["silhouette", "both"]:
            silhouette_k_values = [k for _, k in self.top_clusters]  # Top 5 silhouette scores
            cluster_choices.extend(silhouette_k_values)  # Keep them in ranked order

        if method in ["elbow", "both"]:
            if self.elbow_k is not None and self.elbow_k not in cluster_choices:
                cluster_choices.append(self.elbow_k)  # Append Elbow k at the end

        # Perform clustering for selected clusters
        for rank_idx, n_clusters in enumerate(cluster_choices, 1):
            if n_clusters is None:
                continue  # Skip if no valid k

            # Determine ranking suffix
            rank = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank_idx, f"{rank_idx}th")

            centroids = self.cluster(n_clusters)  # Perform clustering
            if displayPlots:
                self.plot(n_clusters, centroids, rank)  # Plot clustered results
        if exportFile:
            self.export_clusters()

# ================== Command Line Interface ==================

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="KMeans Clustering Pipeline with Elbow and Silhouette Methods")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file (TSV format)")
    parser.add_argument("--coordinates", type=str, required=True, help="Path to xy-coordinates file (TSV format)")
    parser.add_argument("--method", type=str, choices=["silhouette", "elbow", "both"], default="both",
                        help="Choose clustering method: 'silhouette', 'elbow' or (deafult) 'both'")
    parser.add_argument("--export", action="store_true",
                        help="Export the cluster labels? (default) False")
    parser.add_argument("--plots", action="store_true",
                        help="Display plots? (default) False")
    parser.add_argument("--max_iter", type=int, default=300, 
                        help="Maximum number of iterations for K-Means. Default is 300.")
    parser.add_argument("--tol", type=float, default=1e-4,
        help="Relative tolerance to declare convergence. Default is 1e-4.")
    parser.add_argument("--n_init", type=int, default=10,
        help="Number of times the K-Means algorithm will be run with "
             "different centroid seeds. Default is 10.")

    args = parser.parse_args()

    # Initialize pipeline with input files
    pipeline = ClusteringPipeline(args.metadata, args.coordinates, args.max_iter, args.tol, args.n_init)

    # Determine which method to use
    pipeline.run_pipeline(method=args.method, exportFile = args.export, displayPlots = args.plots)
    end_time = time.time()
    print(f"\nExecution Time: {end_time - start:.2f} seconds")
if __name__ == "__main__":
    main()
