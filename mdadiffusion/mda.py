import numpy as np
import scipy as sp
import scipy.linalg

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
    Returns hydrodynamic size in Kirkwood-Riesman approximation
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
    bootstrap_vectors = np.random.choice(
        ensemble_size, (bootstrap_rounds, ensemble_size)
    )
    bootstrap_vectors[0] = np.arange(
        ensemble_size
    )  # zero'th round is simply normal (no randomization)
    bootstrap_weight_accum = np.zeros(
        bootstrap_rounds
    )  # accumulate weight in averaging

    chain_length = len(bead_steric_radii)

    average_trace_mobility = np.zeros((chain_length, chain_length))
    average_pairwise_inverse_distance = np.zeros((chain_length, chain_length))

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
            weight = bootstrap_vectors[j].count(conformer_id)
            prev_weight = bootstrap_weight_accum[j]
            new_weight = prev_weight + weight

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

    for j in bootstrap_rounds:
        rh_mda[j] = minimum_dissipation_approximation(average_trace_mobility[j])
        rh_kr[j] = minimum_dissipation_approximation(
            average_pairwise_inverse_distance[j]
        )

    return {
        "rh_mda": np.mean(rh_mda),
        "rh_mda (se)": np.std(rh_mda),

        "rh_kr": np.mean(rh_kr),
        "rh_kr (se)": np.std(rh_kr),
    }
