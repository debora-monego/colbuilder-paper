import os
import json
import csv
from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import MDAnalysis as mda
from sklearn.cluster import KMeans
import logging
from matplotlib.ticker import FuncFormatter
import warnings
from itertools import cycle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD")

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']

marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

def bootstrap_error(data, n_bootstrap=1000, confidence=0.95):
    bootstrap_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) 
                                for _ in range(n_bootstrap)])
    return np.std(bootstrap_means)

def get_script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        raise

def generate_file_names(config: Dict, system_dir: str) -> List[str]:
    file_names = []
    patterns = config['file_patterns']
    prod_range = config.get('prod_range', [1, 30, 5])

    for pattern in patterns:
        if '{start}' in pattern and '{end}' in pattern:
            start, end, step = prod_range
            for i in range(start, end + 1, step):
                file_name = pattern.format(start=i, end=min(i + step - 1, end))
                if os.path.exists(os.path.join(system_dir, file_name)):
                    file_names.append(file_name)
        else:
            if os.path.exists(os.path.join(system_dir, pattern)):
                file_names.append(pattern)

    return file_names

def save_to_csv(filename: str, header: List[str], data: List[List[float]]) -> None:
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(zip(*data))
    except Exception as e:
        logging.error(f"Failed to save data to CSV: {e}")

def calculate_end_to_end_distance(ace_caps: mda.AtomGroup, nme_caps: mda.AtomGroup) -> float:
    return np.linalg.norm(np.mean(ace_caps.positions, axis=0) - np.mean(nme_caps.positions, axis=0)) / 10

def analyze_system(config: Dict, system_key: str) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    try:
        script_dir = get_script_dir()
        input_dir = os.path.join(config['input_path'], config['systems'][system_key])
        if system_key == "mix":
            input_dir = os.path.join(input_dir, "test_case1")
        output_dir = os.path.join(script_dir, config['output_path'], system_key)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(input_dir)

        file_names = generate_file_names(config, input_dir)
        if not file_names:
            logging.warning(f"No valid trajectory files found for system {system_key}. Skipping analysis.")
            return None

        topology_file = config['topology_file']
        if not os.path.exists(topology_file):
            logging.warning(f"Topology file {topology_file} not found for system {system_key}. Skipping analysis.")
            return None

        uAA = mda.Universe(topology_file, *file_names)
        ace_caps = uAA.select_atoms(f"resname {config['cap_resnames']['ace']}")
        nme_caps = uAA.select_atoms(f"resname {config['cap_resnames']['nme']}")

        cross_resname = config['cross_resnames'][system_key]
        selection = ' or '.join(f'resname {res}' for res in cross_resname) if isinstance(cross_resname, list) else f'resname {cross_resname}'
        selection += ' and name CA'
        cross = uAA.select_atoms(selection)

        cluster_centers, cluster_z = cluster_crosslinking_atoms(uAA, cross, config['kmeans_clusters'])

        if len(cluster_centers) == 0:
            logging.warning(f"No clusters found for system {system_key}. Skipping analysis.")
            return None

        e2e_distances, gap_distances, overlap_distances, dband_distances, dband_errors, times = process_trajectory(
            uAA, ace_caps, nme_caps, cluster_centers, cross, cluster_z, config, output_dir, system_key)

        overlap_extensions = np.where(overlap_distances != 0,
                                    (overlap_distances - overlap_distances[0]) / overlap_distances[0],
                                    np.nan)
        gap_extensions = np.where(gap_distances != 0,
                                (gap_distances - gap_distances[0]) / gap_distances[0],
                                np.nan)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_overlap_gap = np.where(
                (gap_extensions != 0) & (~np.isnan(gap_extensions)) & (~np.isnan(overlap_extensions)),
                overlap_extensions / gap_extensions,
                np.nan
            )

        plot_and_save_data(
            config, system_key, output_dir, times, e2e_distances, gap_distances, overlap_distances, dband_distances, dband_errors, ratio_overlap_gap
        )
        return e2e_distances, gap_distances, overlap_distances, dband_distances, dband_errors, ratio_overlap_gap, times

    except Exception as e:
        logging.error(f"Error analyzing system {system_key}: {e}")
        return None

def process_trajectory(uAA, ace_caps, nme_caps, cluster_centers, cross, cluster_z, config, output_dir, system_key):
    e2e_distances = []
    gap_distances = []
    overlap_distances = []
    dband_distances = []
    dband_frame_errors = []
    dband_bootstrap_errors = []
    times = []

    window_size = 10 
    dband_window = []

    for ts in uAA.trajectory:
        e2e_dist = calculate_end_to_end_distance(ace_caps, nme_caps)
        if np.isnan(e2e_dist):
            logging.warning(f"Frame {ts.frame}: End-to-end distance is NaN.")
            continue

        e2e_distances.append(e2e_dist)
        times.append(ts.frame * config['frame_interval'] / 1000)

        d_gap, d_overlap, d_gap_errors, d_overlap_errors = calculate_d_band_distances(cluster_centers, cross, cluster_z, config['distance_threshold'])
        gap_dist = np.nanmean(d_gap) if d_gap else np.nan
        overlap_dist = np.nanmean(d_overlap) if d_overlap else np.nan
        
        gap_distances.append(gap_dist)
        overlap_distances.append(overlap_dist)
        
        if not np.isnan(gap_dist) and not np.isnan(overlap_dist):
            dband_dist = gap_dist + overlap_dist
            dband_distances.append(dband_dist)
            dband_window.append(dband_dist)
            
            frame_error = np.sqrt(np.nanmean(d_gap_errors)**2 + np.nanmean(d_overlap_errors)**2)
            dband_frame_errors.append(frame_error)
        else:
            dband_distances.append(np.nan)
            dband_window.append(np.nan)
            dband_frame_errors.append(np.nan)

        if len(dband_window) > window_size:
            dband_window.pop(0)

        if len(dband_window) > 1:
            bootstrap_err = bootstrap_error(dband_window)
            dband_bootstrap_errors.append(bootstrap_err)
        else:
            dband_bootstrap_errors.append(np.nan)

    dband_combined_errors = np.sqrt(np.array(dband_frame_errors)**2 + np.array(dband_bootstrap_errors)**2)

    create_diagnostic_plots(times, dband_distances, dband_frame_errors, dband_bootstrap_errors, dband_combined_errors, output_dir, system_key)

    return (np.array(e2e_distances), np.array(gap_distances), np.array(overlap_distances),
            np.array(dband_distances), dband_combined_errors, np.array(times))

def cluster_crosslinking_atoms(universe: mda.Universe, cross_atoms: mda.AtomGroup, n_clusters: int) -> Tuple[np.ndarray, List[List[int]]]:
    if len(cross_atoms) == 0:
        logging.warning("No atoms found for clustering")
        return np.array([]), []

    ly_positions = np.sort(cross_atoms.positions[:, 2])

    if len(ly_positions) < n_clusters:
        logging.warning(f"Number of atoms ({len(ly_positions)}) is less than the number of clusters ({n_clusters})")
        n_clusters = max(min(len(ly_positions), 2), 2)

    kmeans = KMeans(n_clusters).fit(ly_positions.reshape(-1, 1))
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    cluster_z = [[] for _ in range(len(cluster_centers))]
    for i, atom in enumerate(cross_atoms):
        cluster_idx = np.argmin(np.abs(cluster_centers - atom.position[2]))
        cluster_z[cluster_idx].append(i)

    return cluster_centers, cluster_z

def calculate_d_band_distances(cluster_centers: np.ndarray, cross: mda.AtomGroup, cluster_z: List[List[int]], distance_threshold: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    d_gap = []
    d_overlap = []
    d_gap_errors = []
    d_overlap_errors = []
    for cnt in range(len(cluster_centers) - 1):
        atoms_0 = cross[cluster_z[cnt]]
        atoms_1 = cross[cluster_z[cnt + 1]]

        if len(atoms_0) == 0 or len(atoms_1) == 0:
            logging.warning(f"Empty atom group for cluster {cnt} or {cnt + 1}")
            continue

        pos_0 = atoms_0.positions[:, 2]  
        pos_1 = atoms_1.positions[:, 2]
        dist = np.abs(np.mean(pos_1) - np.mean(pos_0)) / 10 

        std_0 = np.std(pos_0)
        std_1 = np.std(pos_1)
        error = np.sqrt((std_0**2 / len(pos_0)) + (std_1**2 / len(pos_1))) / 10  # Error in nm

        if dist <= distance_threshold:
            d_overlap.append(dist)
            d_overlap_errors.append(error)
        else:
            d_gap.append(dist)
            d_gap_errors.append(error)

    return d_gap or [np.nan], d_overlap or [np.nan], d_gap_errors or [np.nan], d_overlap_errors or [np.nan]

def create_diagnostic_plots(times, dband_distances, dband_frame_errors, dband_bootstrap_errors, dband_combined_errors, output_dir, system_key):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(times, dband_distances)
    plt.title('Raw D-band Distances')
    plt.xlabel('Time (ns)')
    plt.ylabel('D-band Distance (nm)')
    
    window = 50 
    running_avg = np.convolve(dband_distances, np.ones(window)/window, mode='valid')
    plt.plot(times[window-1:], running_avg, color='red', linewidth=2)
    plt.legend(['Raw data', f'{window}-point running average'])
    
    plt.subplot(2, 2, 2)
    plt.plot(times, dband_frame_errors)
    plt.title('Frame-by-Frame Errors')
    plt.xlabel('Time (ns)')
    plt.ylabel('Error (nm)')
    
    plt.subplot(2, 2, 3)
    plt.plot(times, dband_bootstrap_errors)
    plt.title('Bootstrap Errors')
    plt.xlabel('Time (ns)')
    plt.ylabel('Error (nm)')
    
    plt.subplot(2, 2, 4)
    plt.plot(times, dband_combined_errors)
    plt.title('Combined Errors')
    plt.xlabel('Time (ns)')
    plt.ylabel('Error (nm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{system_key}_diagnostic_plots.png'))
    plt.close()

def plot_time_series_with_error(times: np.ndarray, data: np.ndarray, errors: np.ndarray, ylabel: str, filename: str, 
                                color: str, xlimits: Tuple[float, float], marker: str) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.rcParams.update({'font.size': 12})

    mask = ~np.isnan(data) & ~np.isnan(errors)
    plt.errorbar(times[mask], data[mask], yerr=errors[mask], color=color, linewidth=1.5, 
                 marker=marker, markersize=5, markeredgecolor='black', markeredgewidth=0.5,
                 markevery=int(len(times[mask])/10), ecolor=color, capsize=3, alpha=0.5)

    plt.xlabel('Time (ns)', labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    plt.xlim(xlimits)
    
    if np.any(mask):
        ymin, ymax = np.nanmin(data[mask]), np.nanmax(data[mask])
        y_range = ymax - ymin
        plt.ylim(ymin - 0.1*y_range, ymax + 0.1*y_range)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(direction='in', which='both', top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series(times: np.ndarray, data: np.ndarray, ylabel: str, filename: str, 
                     color: str, xlimits: Tuple[float, float], marker: str) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.rcParams.update({'font.size': 12})

    mask = ~np.isnan(data)
    plt.plot(times[mask], data[mask], color=color, linewidth=1.5, 
             marker=marker, markersize=5, markeredgecolor='black', markeredgewidth=0.5,
             markevery=int(len(times[mask])/10))

    plt.xlabel('Time (ns)', labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    plt.xlim(xlimits)
    
    if np.any(mask):
        ymin, ymax = np.nanmin(data[mask]), np.nanmax(data[mask])
        y_range = ymax - ymin
        plt.ylim(ymin - 0.1*y_range, ymax + 0.1*y_range)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(direction='in', which='both', top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_ratio_overlap_gap(data: Dict[str, np.ndarray], time_data: Dict[str, np.ndarray],
                                    error_data: Dict[str, np.ndarray], filename: str, colors: Dict[str, str],
                                    xlimits: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    main_ax = fig.add_subplot(111)
    max_ratio = max(np.nanmax(data[sys][:1800]) for sys in data)
    min_ratio = min(np.nanmin(data[sys][:1800]) for sys in data)
    y_max = max_ratio * 1.1
    y_min = min_ratio * 0.5

    num_systems = len(data)
    offset_step = 4

    for i, system in enumerate(data.keys()):
        mask = ~np.isnan(data[system])
        times = time_data[system][mask][:1800]
        ratio = data[system][mask][:1800]
        main_ax.plot(times, ratio, color=colors[system], label=system, linewidth=1.5)

        if error_data is not None and system in error_data:
            errors = error_data[system][mask][:1800]
            num_points = 6
            indices = np.linspace(5, len(times) - 1, num_points, dtype=int)
            offset = (i - (num_systems - 1) / 2) * offset_step
            main_ax.errorbar(times[indices] + offset, ratio[indices], yerr=errors[indices],
                             fmt='o', markersize=4, color=colors[system],
                             ecolor=colors[system], capsize=3, capthick=1,
                             elinewidth=1, markeredgecolor='black', markeredgewidth=0.5)

    main_ax.axhline(y=0.5, color="#e7298a", linestyle="--", linewidth=2, label='Experimental Ratio')
    main_ax.set_xlabel('Time (ns)', fontsize=18)
    main_ax.set_ylabel('Overlap/Gap Strain Ratio', fontsize=18)
    main_ax.set_xlim(xlimits)
    main_ax.set_ylim(y_min, y_max)

    main_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    main_ax.spines['top'].set_visible(True)
    main_ax.spines['right'].set_visible(True)
    main_ax.tick_params(direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_dband_with_insets(data: Dict[str, Dict[str, np.ndarray]], time_data: Dict[str, np.ndarray],
                                    error_data: Dict[str, np.ndarray], filename: str, colors: Dict[str, str],
                                    xlimits: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    main_ax = fig.add_subplot(111)
    max_dband = max(np.nanmax(data['dband'][sys][:1800]) for sys in data['dband'])
    min_dband = min(np.nanmin(data['dband'][sys][:1800]) for sys in data['dband'])
    y_max = max_dband * 1.02
    y_min = min_dband

    num_systems = len(data['dband'])
    offset_step = 2

    for i, system in enumerate(data['dband'].keys()):
        mask = ~np.isnan(data['dband'][system])
        times = time_data[system][mask][:1800]
        dband = data['dband'][system][mask][:1800]
        main_ax.plot(times, dband, color=colors[system], label=system, linewidth=1.5)

        if error_data is not None and system in error_data:
            errors = error_data[system][mask][:1800]
            num_points = 6
            indices = np.linspace(150, len(times) - 1, num_points, dtype=int)
            offset = (i - (num_systems - 1) / 2) * offset_step
            main_ax.errorbar(times[indices] + offset, dband[indices], yerr=errors[indices],
                             fmt='o', markersize=4, color=colors[system],
                             ecolor=colors[system], capsize=3, capthick=1,
                             elinewidth=1, markeredgecolor='black', markeredgewidth=0.5)

    main_ax.axhline(y=82.5, color="#e7298a", linestyle="--", linewidth=2, label='Experimental D-band')
    main_ax.set_xlabel('Time (ns)', fontsize=18)
    main_ax.set_ylabel('D-band Distance (nm)', fontsize=18)
    main_ax.set_xlim(xlimits)
    main_ax.set_ylim(y_min, y_max)

    max_overlap = 1.03*max(np.nanmax(data['overlap'][sys][~np.isnan(data['overlap'][sys])][:500]) for sys in data['overlap'])
    min_overlap = min(np.nanmin(data['overlap'][sys][~np.isnan(data['overlap'][sys])][:500]) for sys in data['overlap'])
    max_gap = 1.03*max(np.nanmax(data['gap'][sys][~np.isnan(data['gap'][sys])][:500]) for sys in data['gap'])
    min_gap = min(np.nanmin(data['gap'][sys][~np.isnan(data['gap'][sys])][:500]) for sys in data['gap'])

    overlap_range = max_overlap - min_overlap
    gap_range = max_gap - min_gap
    max_range = max(overlap_range, gap_range)
    tick_interval = max(1, int(max_range / 5)) 

    total_range = overlap_range + gap_range
    overlap_height = 0.6 * (overlap_range / total_range)
    gap_height = 0.6 * (gap_range / total_range)

    def setup_inset(ax, data_key, title, y_min, y_max):
        for i, system in enumerate(data[data_key].keys()):
            mask = ~np.isnan(data[data_key][system])
            times = time_data[system][mask][:500]
            values = data[data_key][system][mask][:500]
            ax.plot(times, values, color=colors[system])

            if error_data is not None and system in error_data:
                errors = error_data[system][mask][:500]
                num_points = 6
                indices = np.linspace(5, len(times) - 1, num_points, dtype=int)
                offset = (i - (num_systems - 1) / 2) * offset_step / 6  
                ax.errorbar(times[indices] + offset, values[indices], yerr=errors[indices],
                            fmt='o', markersize=3, color=colors[system],
                            ecolor=colors[system], capsize=2, capthick=0.5,
                            elinewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)

        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.set_title(title, fontsize=14)
        ax.tick_params(labelsize=12)

    axins1 = fig.add_axes([0.18, 0.25, 0.35, overlap_height])
    setup_inset(axins1, 'overlap', 'Early Overlap Distance', min_overlap, max_overlap)

    axins2 = fig.add_axes([0.6, 0.25, 0.35, gap_height])
    setup_inset(axins2, 'gap', 'Early Gap Distance', min_gap, max_gap)

    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_and_save_data(config: Dict, system_key: str, output_dir: str, times: np.ndarray, 
                       e2e_distances: np.ndarray, gap_distances: np.ndarray, 
                       overlap_distances: np.ndarray, dband_distances: np.ndarray, 
                       dband_errors: np.ndarray, ratio_overlap_gap: np.ndarray) -> None:
    color = config['colors'][system_key]
    xlimits = (0, 190)
    marker = marker_styles[list(config['systems'].keys()).index(system_key) % len(marker_styles)]

    plot_time_series(times, e2e_distances, 'End-to-End Distance (nm)', 
                     os.path.join(output_dir, f'{system_key}_e2e.png'), color, xlimits, marker)

    plot_time_series(times, gap_distances, 'Gap Distance (nm)', 
                     os.path.join(output_dir, f'{system_key}_gap.png'), color, xlimits, marker)

    plot_time_series(times, overlap_distances, 'Overlap Distance (nm)', 
                     os.path.join(output_dir, f'{system_key}_overlap.png'), color, xlimits, marker)

    plot_time_series_with_error(times, dband_distances, dband_errors, 'D-band Distance (nm)', 
                                os.path.join(output_dir, f'{system_key}_dband.png'), color, xlimits, marker)

    plot_time_series(times, ratio_overlap_gap, 'Ratio Overlap/Gap', 
                     os.path.join(output_dir, f'{system_key}_ratio_overlap_gap.png'), color, xlimits, marker)

    save_to_csv(os.path.join(output_dir, f"{system_key}_e2e.csv"), ['Time [ns]', 'End-to-End Distance [nm]'], [times, e2e_distances])
    save_to_csv(os.path.join(output_dir, f"{system_key}_gap.csv"), ['Time [ns]', 'Gap Distance [nm]'], [times, gap_distances])
    save_to_csv(os.path.join(output_dir, f"{system_key}_overlap.csv"), ['Time [ns]', 'Overlap Distance [nm]'], [times, overlap_distances])
    save_to_csv(os.path.join(output_dir, f"{system_key}_dband.csv"), ['Time [ns]', 'D-band Distance [nm]', 'D-band Error [nm]'], [times, dband_distances, dband_errors])
    save_to_csv(os.path.join(output_dir, f"{system_key}_ratio_overlap_gap.csv"), ['Time [ns]', 'Ratio Overlap/Gap [-]'], [times, ratio_overlap_gap])

def plot_combined_data(data: Dict[str, np.ndarray], time_data: Dict[str, np.ndarray],
                       error_data: Dict[str, np.ndarray], filename: str, colors: Dict[str, str], 
                       xlimits: Tuple[float, float], ylabel: str) -> None:
    plt.figure(figsize=(8, 5)) 
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    all_data = []
    marker_cycle = cycle(marker_styles)
    for system, values in data.items():
        mask = ~np.isnan(values)
        marker = next(marker_cycle)
        
        num_points = 8
        total_points = len(values[mask])
        indices = np.linspace(0, total_points - 1, num_points, dtype=int)
        
        plt.plot(time_data[system][mask], values[mask], color=colors[system], linewidth=1.5, label=system)
        
        plt.plot(time_data[system][mask][indices], values[mask][indices], 
                 color=colors[system], marker=marker, markersize=5, 
                 markeredgecolor='black', markeredgewidth=0.5, linestyle='')

        all_data.extend(values[mask])

        if error_data is not None and system in error_data:
            errors = error_data[system][mask]
            plt.errorbar(time_data[system][mask][indices], values[mask][indices], 
                         yerr=errors[indices], fmt='none', ecolor=colors[system], 
                         capsize=3, capthick=1, elinewidth=1)

    if 'dband' in filename:
        plt.axhline(y=82.5, color="#e7298a", linestyle="--", linewidth=2, label='Experimental')
    elif 'ratio_overlap_gap' in filename:
        plt.axhline(y=0.5, color="#e7298a", linestyle='--', linewidth=2, label='Experimental')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    plt.xlabel('Time (ns)', labelpad=14)
    plt.ylabel(ylabel, labelpad=14)
    plt.xlim(xlimits)

    if all_data:
        ymin, ymax = np.nanmin(all_data), np.nanmax(all_data)
        y_range = ymax - ymin
        plt.ylim(ymin - 0.1*y_range, ymax + 0.1*y_range)

    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(direction='in', which='both', top=True, right=True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def main() -> None:
    config_path = os.path.join(get_script_dir(), 'config.json')
    config = load_config(config_path)

    combined_data = {key: {} for key in ['e2e', 'gap', 'overlap', 'dband', 'dband_errors', 'ratio_overlap_gap', 'ratio_overlap_gap_errors', 'time']}

    for system in config['systems']:
        try:
            result = analyze_system(config, system)
            if result is not None:
                e2e_distances, gap_distances, overlap_distances, dband_distances, dband_errors, ratio_overlap_gap, times = result
                combined_data['e2e'][system] = e2e_distances
                combined_data['gap'][system] = gap_distances
                combined_data['overlap'][system] = overlap_distances
                combined_data['dband'][system] = dband_distances
                combined_data['dband_errors'][system] = dband_errors
                combined_data['ratio_overlap_gap'][system] = ratio_overlap_gap
                combined_data['time'][system] = times

                overlap_extensions = (overlap_distances - overlap_distances[0]) / overlap_distances[0]
                gap_extensions = (gap_distances - gap_distances[0]) / gap_distances[0]
                
                overlap_extension_errors = dband_errors / overlap_distances[0]
                gap_extension_errors = dband_errors / gap_distances[0]
                
                ratio_errors = np.abs(ratio_overlap_gap) * np.sqrt(
                    (overlap_extension_errors / overlap_extensions)**2 + 
                    (gap_extension_errors / gap_extensions)**2
                )
                combined_data['ratio_overlap_gap_errors'][system] = ratio_errors

        except Exception as e:
            logging.error(f"Error processing system {system}: {e}")

    if not any(combined_data['dband']):
        logging.error("No valid data available for plotting. Exiting.")
        return

    combined_output_dir = os.path.join(get_script_dir(), config['output_path'], 'combined')
    os.makedirs(combined_output_dir, exist_ok=True)

    xlimits = (0, 150)
    
    plot_labels = {
        'e2e': 'End-to-End Distance (nm)',
        'gap': 'Gap Distance (nm)',
        'overlap': 'Overlap Distance (nm)',
        'dband': 'D-band Distance (nm)',
        'ratio_overlap_gap': 'Overlap/Gap Strain Ratio'
    }

    for metric in ['e2e', 'gap', 'overlap', 'dband', 'ratio_overlap_gap']:
        if metric in config['plot_series'] and any(combined_data[metric]):
            error_data = None
            if metric == 'ratio_overlap_gap':
                error_data = combined_data['ratio_overlap_gap_errors']
            elif metric == 'dband':
                error_data = combined_data['dband_errors']
            
            plot_combined_data(
                data=combined_data[metric],
                time_data=combined_data['time'],
                error_data=error_data,
                filename=os.path.join(combined_output_dir, f'combined_{metric}.png'),
                colors=config['colors'],
                xlimits=xlimits,
                ylabel=plot_labels[metric]
            )
            
    plot_combined_ratio_overlap_gap(
        data=combined_data['ratio_overlap_gap'],
        time_data=combined_data['time'],
        error_data=combined_data['ratio_overlap_gap_errors'],
        filename=os.path.join(combined_output_dir, 'combined_ratio_overlap_gap.png'),
        colors=config['colors'],
        xlimits=(0, 150)
    )

    if any(combined_data['dband']):
        plot_combined_dband_with_insets(
            data=combined_data,
            time_data=combined_data['time'],
            error_data=combined_data['dband_errors'],
            filename=os.path.join(combined_output_dir, 'combined_dband_with_insets.png'),
            colors=config['colors'],
            xlimits=xlimits
        )
    else:
        logging.warning("No valid D-band data available for plotting insets.")

if __name__ == "__main__":
    main()