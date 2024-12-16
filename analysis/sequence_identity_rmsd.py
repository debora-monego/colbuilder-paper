import os
from Bio import SeqIO, AlignIO, Align
from Bio.PDB import PDBParser, Superimposer
import matplotlib.pyplot as plt
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import subprocess
import tempfile

# Suppress PDBConstructionWarnings
warnings.simplefilter('ignore', PDBConstructionWarning)

PURPLE_COLORS = {
    'A': '#238b45', 'B': '#00441b', 'C': '#238b45'
}
BLUE_COLORS = {
    'A': '#238b45', 'B': '#00441b', 'C': '#238b45'
}

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a dictionary with chain IDs as keys and sequences as values.
    
    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        dict: A dictionary mapping chain IDs to sequences.
    """
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        chain = record.id.split(":")[-1]
        sequences[chain] = str(record.seq)
    return sequences

def run_muscle(input_fasta, output_aln):
    """
    Runs MUSCLE to perform multiple sequence alignment on the input FASTA file using the correct syntax for MUSCLE 5.

    Args:
        input_fasta (str): Path to the input FASTA file.
        output_aln (str): Path to the output alignment file.
    """
    try:
        subprocess.run(
            ['muscle', '-align', input_fasta, '-output', output_aln],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running MUSCLE: {e}")
        raise

def calculate_atom_rmsd(ref_atoms, sample_atoms):
    """
    Calculates the RMSD between two sets of atoms.

    Args:
        ref_atoms (list): List of reference atoms.
        sample_atoms (list): List of sample atoms.

    Returns:
        float: The RMSD value.
    """
    if len(ref_atoms) != len(sample_atoms):
        return None
    sup = Superimposer()
    sup.set_atoms(ref_atoms, sample_atoms)
    return sup.rms

def get_matching_atoms(res1, res2):
    """
    Retrieves matching atoms between two residues.

    Args:
        res1 (Residue): The first residue.
        res2 (Residue): The second residue.

    Returns:
        tuple: Two lists of matching atoms.
    """
    atoms1 = []
    atoms2 = []
    for atom in res1:
        if atom.name in res2 and atom.name in res2:
            atoms1.append(atom)
            atoms2.append(res2[atom.name])
    return atoms1, atoms2

def calculate_single_rmsd(mol, template_res_list, template_aligned, mol_aligned, chain_id):
    """
    Calculates RMSD values between a template and a molecule structure.

    Args:
        mol (Structure): The molecule structure.
        template_res_list (list): List of template residues.
        template_aligned (str): Aligned sequence of the template.
        mol_aligned (str): Aligned sequence of the molecule.
        chain_id (str): Chain identifier.

    Returns:
        list: RMSD values.
    """
    mol_res_list = list(mol[0][chain_id].get_residues())
    rmsd_values = []
    template_idx, mol_idx = 0, 0

    for a, b in zip(template_aligned, mol_aligned):
        if a == '-' or b == '-':
            rmsd_values.append(None)
            if a != '-':
                template_idx += 1
            if b != '-':
                mol_idx += 1
        else:
            res_template = template_res_list[template_idx]
            res_mol = mol_res_list[mol_idx]
            atoms_template, atoms_mol = get_matching_atoms(res_template, res_mol)
            if atoms_template and atoms_mol:
                rmsd = calculate_atom_rmsd(atoms_template, atoms_mol)
                rmsd_values.append(rmsd)
            else:
                rmsd_values.append(None)
            template_idx += 1
            mol_idx += 1

    return rmsd_values

def calculate_rmsd(template_pdb, mol_pdb, template_fasta, mol_fasta, chain_id, calculate_error=False):
    """
    Calculates RMSD and sequence identity between a template and a molecule structure.

    Args:
        template_pdb (str): Path to the template PDB file.
        mol_pdb (str): Path to the molecule PDB file.
        template_fasta (str): Path to the template FASTA file.
        mol_fasta (str): Path to the molecule FASTA file.
        chain_id (str): Chain identifier.
        calculate_error (bool): Whether to calculate error (standard deviation).

    Returns:
        tuple: RMSD mean, RMSD max mean, RMSD max std, sequence identity, aligned sequences.
    """
    parser = PDBParser()
    template = parser.get_structure("template", template_pdb)
    template_seq = read_fasta(template_fasta)
    mol_seq = read_fasta(mol_fasta)

    print(f"Template sequence for chain {chain_id}: {template_seq.get(chain_id, 'Not found')}")
    print(f"Molecule sequence for chain {chain_id}: {mol_seq.get(chain_id, 'Not found')}")

    # Check if sequences exist for the given chain
    if chain_id not in template_seq or chain_id not in mol_seq:
        raise ValueError(f"Sequence for chain {chain_id} not found in template or molecule.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.fasta') as temp_fasta, tempfile.NamedTemporaryFile(delete=False, suffix='.aln') as temp_aln:
        temp_fasta.write(f">{chain_id}_template\n{template_seq[chain_id]}\n>{chain_id}_mol\n{mol_seq[chain_id]}\n".encode())
        temp_fasta.flush()  

        print(f"Temporary FASTA file created at: {temp_fasta.name}")
        with open(temp_fasta.name, 'r') as f:
            print(f.read())

        run_muscle(temp_fasta.name, temp_aln.name)
        alignment = AlignIO.read(temp_aln.name, "fasta")

    template_aligned, mol_aligned = alignment[0].seq, alignment[1].seq
    template_res_list = list(template[0][chain_id].get_residues())

    if isinstance(mol_pdb, list) and calculate_error:
        rmsd_values_list = []
        for pdb in mol_pdb:
            mol = parser.get_structure("mol", pdb)
            rmsd_values_list.append(calculate_single_rmsd(mol, template_res_list, template_aligned, mol_aligned, chain_id))
        rmsd_array = np.array(rmsd_values_list)
        rmsd_mean = np.nanmean(rmsd_array, axis=0)
        rmsd_max = np.nanmax(rmsd_array, axis=1)
        rmsd_max_mean = np.nanmean(rmsd_max)
        rmsd_max_std = np.nanstd(rmsd_max) if calculate_error else None
    else:
        mol = parser.get_structure("mol", mol_pdb if isinstance(mol_pdb, str) else mol_pdb[0])
        rmsd_mean = calculate_single_rmsd(mol, template_res_list, template_aligned, mol_aligned, chain_id)
        rmsd_mean_filtered = [r for r in rmsd_mean if r is not None]
        rmsd_max_mean = max(rmsd_mean_filtered) if rmsd_mean_filtered else 0
        rmsd_max_std = None

    identical_count = sum(1 for a, b in zip(template_aligned, mol_aligned) if a == b and a != '-' and b != '-')
    total_aligned = sum(1 for a, b in zip(template_aligned, mol_aligned) if a != '-' and b != '-')
    identity = identical_count / total_aligned if total_aligned > 0 else 0

    return rmsd_mean, rmsd_max_mean, rmsd_max_std, identity, template_aligned, mol_aligned

def plot_rmsd_with_identity(template_pdb, mol_pdb, template_fasta, mol_fasta, output_file, mol_name, color_scheme='purple', calculate_error=False, start_residues={'template': {'A': 152, 'B': 86, 'C': 152}, 'mol': {'A': 162, 'B': 80, 'C': 162}}):
    """
    Plots RMSD and sequence identity for comparison between a template and a molecule structure.

    Args:
        template_pdb (str): Path to the template PDB file.
        mol_pdb (str): Path to the molecule PDB file.
        template_fasta (str): Path to the template FASTA file.
        mol_fasta (str): Path to the molecule FASTA file.
        output_file (str): Path to save the output plot.
        mol_name (str): Name of the molecule.
        color_scheme (str): Color scheme for the plot ('purple' or 'blue').
        calculate_error (bool): Whether to calculate error (standard deviation).
        start_residues (dict): Starting residue numbers for labeling.
    """
    fig, ax = plt.subplots(figsize=(32, 10))

    colors = PURPLE_COLORS if color_scheme == 'purple' else BLUE_COLORS
    offsets = {'A': 0, 'B': 1.7, 'C': 3.4}

    max_x = 0
    for chain_id in ['A', 'B', 'C']:
        rmsd_mean, rmsd_max_mean, rmsd_max_std, identity, template_aligned, mol_aligned = calculate_rmsd(
            template_pdb, mol_pdb, template_fasta, mol_fasta, chain_id, calculate_error
        )

        x_values = range(len(template_aligned))

        start = None
        for i, (a, b) in enumerate(zip(template_aligned, mol_aligned)):
            if a == b and a != '-':
                if start is None:
                    start = i 
            else:
                if start is not None:
                    
                    ax.fill_between([start, i],
                                    [offsets[chain_id]] * 2,
                                    [offsets[chain_id] + 1] * 2,
                                    color=colors[chain_id], alpha=0.3)
                    start = None  

        if start is not None:
            ax.fill_between([start, len(template_aligned)],
                            [offsets[chain_id]] * 2,
                            [offsets[chain_id] + 1] * 2,
                            color=colors[chain_id], alpha=0.3)

        for i, (a, b) in enumerate(zip(template_aligned, mol_aligned)):
            if a != b and a != '-' and b != '-':
                ax.fill_between([i, i + 1],
                                [offsets[chain_id]] * 2,
                                [offsets[chain_id] + 1] * 2,
                                color="#f7fcf5", alpha=0.8)

        plotted_x = [i for i, rmsd in enumerate(rmsd_mean) if rmsd is not None]
        plotted_y = [rmsd + offsets[chain_id] for rmsd in rmsd_mean if rmsd is not None]
        ax.plot(plotted_x, plotted_y, color=colors[chain_id], linewidth=2)

        max_x = max(max_x, len(template_aligned))

        ax.plot([-10, -10], 
                [offsets[chain_id], offsets[chain_id] + rmsd_max_mean], 
                color=colors[chain_id], linewidth=5)
        
        rmsd_text = f'{rmsd_max_mean:.1f} Å'
        if calculate_error and rmsd_max_std is not None:
            rmsd_text += f' ± {rmsd_max_std:.1f} Å'
        ax.text(-55, offsets[chain_id] + rmsd_max_mean / 2, 
                rmsd_text, 
                color=colors[chain_id], va='center', ha='left', fontsize=24)

        ax.text(len(template_aligned) + 30, offsets[chain_id] + 0.5, 
                f'chain {chain_id}:\n{identity:.1%}', 
                color=colors[chain_id], va='center', ha='left', fontsize=24)

        mol_residue_number = start_residues['mol'][chain_id]
        for i, (template_aa, mol_aa) in enumerate(zip(template_aligned, mol_aligned)):
            if mol_aa != '-':
                if (i % 50 == 0 or i == 0 or i == len(mol_aligned) - 1):
                    if (i <= (len(mol_aligned) - 20) or i > (len(mol_aligned) - 2)):
                        ax.text(i, offsets[chain_id] - 0.05, f'{mol_aa}\n{mol_residue_number}', 
                                ha='center', va='top', fontsize=16, rotation=0, 
                                color=colors[chain_id])
                mol_residue_number += 1


    ax.set_ylim(bottom=-0.5, top=5)
    ax.set_xlim(left=-60, right=max_x + 120)
    #ax.set_title(f'RMSD and Sequence Identity by Residue - {mol_name}', fontsize=24, pad=20)
    ax.text(len(template_aligned) + 20, offsets['C'] + 1.5,
            f'{mol_name}', color=colors['B'], va='center', ha='left', fontsize=30)

    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    template_species = "rattusnovergicus"
    species1 = "homosapiens"
    species2 = "myotislucifugus"
    template_fasta = f'../sequences/{template_species}.fasta'
    mol1_fasta = f'../sequences/{species1}.fasta'
    mol2_fasta = f'../sequences/{species2}.fasta'
    template_pdb = f'../pdbs/{template_species}.pdb'
    mol1_pdb = f"../pdbs/{species1}.pdb"
    mol2_pdb = f"../pdbs/{species2}.pdb"

    plot_rmsd_with_identity(template_pdb, mol1_pdb, template_fasta, mol1_fasta, 
                            f"rmsd_identity_mol1_{species1}.png", species1, 
                            color_scheme='purple', calculate_error=True,
                            start_residues={'template': {'A': 152, 'B': 86, 'C': 152}, 
                                            'mol': {'A': 162, 'B': 80, 'C': 162}})

    plot_rmsd_with_identity(template_pdb, mol2_pdb, template_fasta, mol2_fasta, 
                            f"rmsd_identity_mol2_{species2}.png", species2, 
                            color_scheme='blue', calculate_error=False,
                            start_residues={'template': {'A': 152, 'B': 86, 'C': 152}, 
                                            'mol': {'A': 146, 'B': 73, 'C': 146}})

if __name__ == "__main__":
    main()
