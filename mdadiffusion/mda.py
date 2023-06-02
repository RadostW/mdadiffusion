import numpy as np
import scipy as sp
import scipy.linalg
import re

import pygrpy
import sarw_spheres

import tqdm


def _lapackinv(mat):
    # inversion of POSDEF matrices
    zz, _ = sp.linalg.lapack.dpotrf(mat, False, False)  # cholesky decompose
    inv_M, _ = sp.linalg.lapack.dpotri(zz)  # invert triangle
    inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T  # combine triangles
    return inv_M


def minimum_dissipation_approximation(average_trace_mobility):
    """
    Returns hydrodynamic size in minimum dissipation approximation
    For details see:
    Cichocki, B; Rubin, M.; Niedzwiecka A. & Szymczak P.
    Diffusion coefficients of elastic macromolecules.
    J. Fluid Mech. 878 R3 (2019)

    Parameters
    ----------
    average_mobility_matrix: np.array
        An ``N`` by ``N`` array. Ensemble average value of beadwise trace of the mobility matrix.

    Returns
    -------
    float
        Diffusive hydrodynamic size.
    """

    total = np.sum(_lapackinv(average_trace_mobility))
    return total / (2 * np.pi)


def kirkwood_riseman_approximation(average_pairwise_inverse_distance):
    """
    Returns hydrodynamic size in Kirkwood-Riesman approximation
    For details see:
    Kirkwood, J. G. & Riseman, J. The intrinsic viscosities and
    diffusion constants of flexible macromolecules in solution.
    J. Chem. Phys. 16, 565–573 (1948)

    Parameters
    ----------
    average_pariwise__inverse_distance: np.array
        An ``N`` by ``N`` array. Ensemble average value of 1 / r_ij. Zeros on the diagonal.

    Returns
    -------
    float
        Diffusive hydrodynamic size.
    """

    chain_length = len(average_pairwise_inverse_distance)

    return chain_length * (chain_length - 1) / np.sum(average_pairwise_inverse_distance)


def hydrodynamic_size(
    bead_steric_radii,
    bead_hydrodynamic_radii,
    ensemble_size,
    bootstrap_rounds=10,
    progress=False,
):
    """
    Returns hydrodynamic size estimate from description of the bead-structure of the molecule.

    Parameters
    ----------
    bead_steric_radii: np.array
        Vector of length ``N`` describing steric sizes of beads. Used for conformer generation.
    bead_hydrodynamic_radii: np.array
        Vector of length ``N`` describing hydrodynamic sizes of beads. Used for mobility tensor estimation.
    ensemble_size: int
        Number of conformers to generate
    bootstrap_rounds: int, default=10
        Number of bootstrap rounds for numerical error estimation
    progress: bool, default=False
        Show progress bar during ensemble generation

    Returns
    -------
    dict
        Dictionary (string,float) with the following keys:
        'rh_mda' -- hydrodynamic size in MDA approximation
        'rh_mda (se)' -- standard error of the above
        'rh_kr' -- hydrodynamic size in Kirkwood-Riseman approximation
        'rh_kr (se)' -- standard error of the above
    """
    bootstrap_vectors = np.random.choice(
        ensemble_size, (bootstrap_rounds, ensemble_size)
    )
    bootstrap_vectors[0] = np.arange(
        ensemble_size
    )  # zero'th round is simply normal (no randomization)
    bootstrap_weight_accum = np.zeros(
        bootstrap_rounds
    )  # accumulated weight in averaging

    chain_length = len(bead_steric_radii)

    average_trace_mobility = np.zeros((bootstrap_rounds, chain_length, chain_length))
    average_pairwise_inverse_distance = np.zeros(
        (bootstrap_rounds, chain_length, chain_length)
    )

    if progress:
        gen = tqdm.tqdm(range(ensemble_size))
    else:
        gen = range(ensemble_size)

    for conformer_id in gen:
        conformer = sarw_spheres.generateChain(np.array(bead_steric_radii))

        c_distances = np.sum(
            (conformer[:, np.newaxis, :] - conformer[np.newaxis, :, :]) ** 2, axis=-1
        ) ** (1 / 2)
        c_inverse_distance = (c_distances + np.eye(chain_length)) ** (-1) * (
            np.ones((chain_length, chain_length)) - np.eye(chain_length)
        )
        c_trace_mobility = pygrpy.grpy_tensors.muTT_trace(
            conformer, bead_hydrodynamic_radii
        )

        for j in range(bootstrap_rounds):
            weight = np.count_nonzero(bootstrap_vectors[j] == conformer_id)
            prev_weight = bootstrap_weight_accum[j]
            new_weight = prev_weight + weight
            if new_weight == 0:
                continue

            average_trace_mobility[j] = (
                prev_weight / (new_weight)
            ) * average_trace_mobility[j] + (weight / new_weight) * c_trace_mobility

            average_pairwise_inverse_distance[j] = (
                prev_weight / (new_weight)
            ) * average_pairwise_inverse_distance[j] + (
                weight / new_weight
            ) * c_inverse_distance

            bootstrap_weight_accum[j] = new_weight

    rh_mda = np.zeros(bootstrap_rounds)
    rh_kr = np.zeros(bootstrap_rounds)

    for j in range(bootstrap_rounds):
        rh_mda[j] = minimum_dissipation_approximation(average_trace_mobility[j])
        rh_kr[j] = kirkwood_riseman_approximation(
            average_pairwise_inverse_distance[j]
        )

    return {
        "rh_mda": rh_mda[0],
        "rh_mda (se)": np.std(rh_mda) / np.sqrt(bootstrap_rounds),
        "rh_kr": rh_kr[0],
        "rh_kr (se)": np.std(rh_kr) / np.sqrt(bootstrap_rounds),
    }


def bead_model_from_sequence(
    annotated_sequence,
    effective_density,
    hydration_thickness,
    disordered_radii,
    c_alpha_distance,
    aa_masses,
):
    """
    Returns bead model from annotated protein sequence

    Parameters
    ----------
    sequence: string
        String describing protein sequence with square brackets [] denoting start/end of rigid domains.
    effective_density: float
        Effective density of rigid domain cores. Units: [Da / Angstrom^3]
    hydration_thickness: float
        Thickness of hydration layer of rigid domains. Units: [Angstrom]
    disordered_radii: float
        Hydrodynamic radius of each of the spheres modelling disordered segments. Units: [Angstrom]
    c_alpha_distance: float
        C alpha distance used as steric size of the beads in disordered segments. Units: [Angstrom]
    aa_masses: dict
        Dictionary mapping single letter codes to masses. Units: [Da]

    Returns
    -------
    dict
        Dictionary with following key-value pairs:
        'bead_decription_compact' -- array with elements [r_h,r_s,repeat]
        'steric_radii' -- long list of steric radii
        'hydrodynamic_radii' -- long list of hydrodynamic radii
    """

    blocks = re.split(
        r"(\[[A-Z].*?\])", annotated_sequence
    )  # things inside braces are blocks
    bead_description_compact = []
    for block in blocks:
        if len(block) >= 2 and block[0] == "[":
            block_mass = sum(aa_masses[aa] for aa in block[1:-1])
            block_excluded_volume_radius = (
                block_mass * (3 / (4 * np.pi)) / effective_density
            ) ** (1 / 3)
            block_radius = block_excluded_volume_radius + hydration_thickness

            bead_description_compact.append([block_radius, block_radius, 1])
        elif len(block) > 0:
            bead_description_compact.append(
                [
                    disordered_radii,
                    c_alpha_distance / 2,
                    len(block),
                ]
            )

    hydrodynamic_radii = []
    steric_radii = []

    for rh, rs, rep in bead_description_compact:
        hydrodynamic_radii = hydrodynamic_radii + rep * [rh]
        steric_radii = steric_radii + rep * [rs]

    return {
        "bead_description_compact": np.array(bead_description_compact),
        "steric_radii": np.array(steric_radii),
        "hydrodynamic_radii": np.array(hydrodynamic_radii),
    }
