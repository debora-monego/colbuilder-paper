import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
import glob

def identify_molecules(pdb_file):
    molecule_coords = []
    current_molecule = []
    current_chain = None
    chain_sequence = ''

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                chain_id = line[21]
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                
                if chain_id != current_chain:
                    if current_chain is not None:
                        chain_sequence += current_chain
                        if chain_sequence == 'ABC':
                            if current_molecule:
                                molecule_coords.append(np.array(current_molecule))
                            current_molecule = []
                            chain_sequence = ''
                    current_chain = chain_id
                
                current_molecule.append([x, y, z])

    if current_molecule:
        molecule_coords.append(np.array(current_molecule))

    return molecule_coords

def calculate_helix_intersections(molecule_coords, z_slice, slice_thickness):
    intersections = []
    for molecule in molecule_coords:
        mask = (molecule[:, 2] >= z_slice) & (molecule[:, 2] < z_slice + slice_thickness)
        if np.any(mask):
            intersection = np.mean(molecule[mask], axis=0)
            intersections.append(intersection[:2])  # Only x and y coordinates
    return np.array(intersections)

def calculate_lateral_spacing(intersections, max_distance=30):
    if len(intersections) < 2:
        return []
    
    tree = cKDTree(intersections)
    distances, _ = tree.query(intersections, k=7, distance_upper_bound=max_distance)
    
    valid_distances = distances[:, 1:][distances[:, 1:] < max_distance]
    return valid_distances.flatten()

def plot_cross_section(intersections, output_file):
    plt.figure(figsize=(10, 10))
    plt.scatter(intersections[:, 0], intersections[:, 1], alpha=0.5)
    plt.title("Cross-Section of Collagen Fibril")
    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.axis('equal')

    center = np.mean(intersections, axis=0)
    radius = np.max(np.linalg.norm(intersections - center, axis=1))
    circle = plt.Circle(center, radius, fill=False, color='r', linestyle='--', linewidth=2)
    plt.gca().add_artist(circle)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_lateral_spacing_histogram(spacings, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(spacings, bins=50, edgecolor='black')
    plt.xlabel('Lateral Spacing (Å)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lateral Spacings in Collagen Fibril')
    plt.savefig(output_file)
    plt.close()

def save_lateral_spacings(lateral_spacings, output_file):
    np.savetxt(output_file, lateral_spacings, delimiter=',', header='lateral_spacing_angstrom')

def process_pdb_file(pdb_file, output_dir, slice_thickness=10):
    print(f"Processing file: {pdb_file}")
    try:
        molecule_coords = identify_molecules(pdb_file)
        
        if not molecule_coords:
            print(f"Warning: No valid molecules found in {pdb_file}. Skipping this file.")
            return None, None, None

        # Calculate the middle z-coordinate for the slice
        all_z = np.concatenate([m[:, 2] for m in molecule_coords])
        z_mid = (np.min(all_z) + np.max(all_z)) / 2

        intersections = calculate_helix_intersections(molecule_coords, z_mid, slice_thickness)
        lateral_spacings = calculate_lateral_spacing(intersections)

        print(f"Number of molecules: {len(molecule_coords)}")
        print(f"Number of intersections in cross-section: {len(intersections)}")
        print(f"Number of lateral spacing measurements: {len(lateral_spacings)}")
        print(f"Mean lateral spacing: {np.mean(lateral_spacings):.2f} Å")
        print(f"Median lateral spacing: {np.median(lateral_spacings):.2f} Å")
        print(f"Std dev of lateral spacing: {np.std(lateral_spacings):.2f} Å")

        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        
        histogram_file = os.path.join(output_dir, f"{base_name}_lateral_spacing_histogram.png")
        plot_lateral_spacing_histogram(lateral_spacings, histogram_file)

        cross_section_file = os.path.join(output_dir, f"{base_name}_cross_section.png")
        plot_cross_section(intersections, cross_section_file)

        # Save lateral spacings to a data file
        data_file = os.path.join(output_dir, f"{base_name}_lateral_spacings.csv")
        save_lateral_spacings(lateral_spacings, data_file)

        return np.mean(lateral_spacings), np.std(lateral_spacings), intersections, lateral_spacings

    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        return None, None, None, None

def process_all_pdb_files(base_dir, output_dir):
    all_mean_spacings = []
    all_std_spacings = []
    all_intersections = []
    all_lateral_spacings = []
    processed_files = 0
    skipped_files = 0

    for dir_path in glob.glob(os.path.join(base_dir, "distance_*")):
        pdb_file = os.path.join(dir_path, "collagen_rat.pdb")
        if os.path.exists(pdb_file):
            dir_output = os.path.join(output_dir, os.path.basename(dir_path))
            os.makedirs(dir_output, exist_ok=True)
            
            result = process_pdb_file(pdb_file, dir_output)
            if result is not None and all(x is not None for x in result):
                mean_spacing, std_spacing, intersections, lateral_spacings = result
                all_mean_spacings.append(mean_spacing)
                all_std_spacings.append(std_spacing)
                all_intersections.extend(intersections)
                all_lateral_spacings.extend(lateral_spacings)
                processed_files += 1
                print(f"Successfully processed: {pdb_file}")
            else:
                skipped_files += 1
                print(f"Skipped due to invalid results: {pdb_file}")
            print("--------------------")

    print(f"\nProcessing complete. Files processed: {processed_files}, Files skipped: {skipped_files}")

    if all_mean_spacings:
        overall_mean = np.mean(all_mean_spacings)
        overall_std = np.mean(all_std_spacings)
        print(f"\nOverall Mean Lateral Spacing: {overall_mean:.2f} Å")
        print(f"Average Standard Deviation: {overall_std:.2f} Å")

        plot_lateral_spacing_histogram(all_lateral_spacings, 
                                       os.path.join(output_dir, "overall_lateral_spacing_distribution.png"))

        # Plot average cross-section from all processed files
        if all_intersections:
            plot_cross_section(np.array(all_intersections), os.path.join(output_dir, "average_cross_section.png"))

        # Save all lateral spacings to a single file
        all_spacings_file = os.path.join(output_dir, "all_lateral_spacings.csv")
        save_lateral_spacings(all_lateral_spacings, all_spacings_file)
    else:
        print("No valid spacing data found in any of the processed files.")

base_dir = "/hits/fast/mbm/monegod/1-collagen_models/colbuilder-paper-data/colbuilder/colbuilder_paper_output/dc/"
output_dir = "lateral_spacing_analysis_results"
os.makedirs(output_dir, exist_ok=True)
process_all_pdb_files(base_dir, output_dir)
