import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_lateral_spacings(base_dir, max_threshold=23):
    all_spacings = []

    for dir_path in glob.glob(os.path.join(base_dir, "distance_*")):
        csv_file = os.path.join(dir_path, "collagen_rat_lateral_spacings.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, comment='#', header=None, names=['lateral_spacing_angstrom'])
            spacings = df['lateral_spacing_angstrom'].values
            
            # Filter out zeros and values greater than the threshold - same helix or not next neighbors
            filtered_spacings = spacings[(spacings > 1e-10) & (spacings <= max_threshold)]
            
            all_spacings.extend(filtered_spacings)

    if not all_spacings:
        print("No valid spacing data found.")
        return

    all_spacings = np.array(all_spacings)

    average = np.mean(all_spacings)
    std_dev = np.std(all_spacings)
    median = np.median(all_spacings)

    print(f"Number of valid spacings: {len(all_spacings)}")
    print(f"Average lateral spacing: {average:.2f} Å")
    print(f"Median lateral spacing: {median:.2f} Å")
    print(f"Standard deviation: {std_dev:.2f} Å")

    plt.figure(figsize=(10, 6))
    plt.hist(all_spacings, bins=50, edgecolor='black')
    plt.title(f"Distribution of Lateral Spacings (0 < x ≤ {max_threshold} Å)")
    plt.xlabel("Lateral Spacing (Å)")
    plt.ylabel("Frequency")
    plt.axvline(average, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({average:.2f} Å)')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median ({median:.2f} Å)')
    plt.legend()
    plt.savefig(os.path.join(base_dir, "combined_lateral_spacing_histogram.png"))
    plt.close()

    np.savetxt(os.path.join(base_dir, "filtered_combined_lateral_spacings.csv"), 
               all_spacings, delimiter=',', header='lateral_spacing_angstrom', comments='')

base_directory = "./"

analyze_lateral_spacings(base_directory)