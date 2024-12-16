import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull

def calculate_radius_and_chains(pdb_file):
    all_coords = []
    current_chain = None
    chain_count = 0
    
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                chain_id = line[21]
                if chain_id != current_chain:
                    chain_count += 1
                    current_chain = chain_id
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                all_coords.append([x, y])  # Only consider x and y coordinates
    
    all_coords = np.array(all_coords)
    
    # Calculate convex hull
    hull = ConvexHull(all_coords)
    hull_points = all_coords[hull.vertices]
    
    # Calculate centroid of all points
    centroid = np.mean(all_coords, axis=0)
    
    # Calculate distances from centroid to hull points
    distances = np.linalg.norm(hull_points - centroid, axis=1)
    
    # Calculate radius as the 95th percentile of distances
    radius = np.percentile(distances, 95)
    
    # Calculate interquartile range (IQR) for error estimation
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    
    # Use IQR as a measure of spread (error)
    radius_error = iqr / 2
    
    return radius, radius_error, chain_count

def analyze_pdb_files(root_dir):
    results = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.startswith(os.path.join(root_dir, "distance_")):
            try:
                contact_distance = float(dirpath.split("_")[-1])
            except ValueError:
                print(f"Skipping directory {dirpath}: Unable to parse contact distance")
                continue
            
            for filename in filenames:
                if filename.endswith(".pdb"):
                    pdb_file = os.path.join(dirpath, filename)
                    try:
                        radius, radius_error, num_chains = calculate_radius_and_chains(pdb_file)
                        results.append((contact_distance, radius, radius_error, num_chains, filename))
                    except Exception as e:
                        print(f"Error processing {pdb_file}: {str(e)}")
    
    return results

def save_results_to_csv(results, filename="fibril_analysis_results.csv"):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Contact Distance', 'Fibril Radius (Å)', 'Radius Error', 'Number of Chains', 'Filename'])
        for result in sorted(results):
            csvwriter.writerow(result)
    print(f"Results saved to {filename}")

def plot_results(results):
    if not results:
        print("No valid results to plot.")
        return

    contact_distances, radii, radius_errors, num_chains, _ = zip(*sorted(results))
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Plot Radius vs Contact Distance
    plt.figure(figsize=(10, 8))
    plt.errorbar(contact_distances, radii, yerr=radius_errors, fmt='o-', capsize=5, ecolor='gray', markersize=8)
    plt.xlabel("Contact Distance")
    plt.ylabel("Fibril Radius (Å)")
    plt.title("Contact Distance vs Fibril Radius")
    
    z = np.polyfit(contact_distances, radii, 1)
    p = np.poly1d(z)
    plt.plot(contact_distances, p(contact_distances), "r--", alpha=0.8)
    
    r_squared = stats.pearsonr(contact_distances, radii)[0]**2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("fibril_radius_vs_contact_distance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Number of Chains vs Contact Distance
    plt.figure(figsize=(10, 8))
    plt.plot(contact_distances, num_chains, 'o-', markersize=8)
    plt.xlabel("Contact Distance")
    plt.ylabel("Number of Chains")
    plt.title("Contact Distance vs Number of Chains")
    
    z = np.polyfit(contact_distances, num_chains, 1)
    p = np.poly1d(z)
    plt.plot(contact_distances, p(contact_distances), "r--", alpha=0.8)
    
    r_squared = stats.pearsonr(contact_distances, num_chains)[0]**2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("num_chains_vs_contact_distance.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    root_dir = "/hits/fast/mbm/monegod/1-collagen_models/colbuilder-paper-data/colbuilder/colbuilder_paper_output/dc"
    #root_dir = "./dc_test"
    results = analyze_pdb_files(root_dir)
    save_results_to_csv(results)
    plot_results(results)
    print("Analysis complete. Results plotted and saved as separate PNG files and saved in 'fibril_analysis_results.csv'")
