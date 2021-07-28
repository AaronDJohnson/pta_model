from mpmath import *
import numpy as np


def fourier_matrix(toas, num_freqs=30):
    time_span = max(toas) - min(toas)
    # freq_min = 1 / time_span
    # freq_max = num_freqs / time_span

    num_toas = len(toas)
    matrix = np.zeros((num_toas, 2 * num_freqs)) * mpf('1')
    div = np.vectorize(fdiv)
    freqs = div(arange(1, num_freqs + 1), time_span)
    omega = 2 * pi * freqs

    arg = np.outer(toas, omega)
    sine = np.vectorize(mp.sin)
    cosine = np.vectorize(mp.cos)
    matrix[:,::2] = sine(arg)
    matrix[:,1::2] = cosine(arg)
    return matrix, np.repeat(freqs, 2)


def powerlaw(freqs, log10_A, gamma, components=2):
    """
    default ref_freq is 1 / yr in hz
    components is the number of repetitions of freqs
    """
    ref_freq = mpf("1 / 31557600")
    df = np.diff(np.concatenate((np.array([0]), freqs[::components])))
    return (
        (10**log10_A)**2 / (12 * np.pi**2) * ref_freq ** (gamma - 3) * freqs ** (-gamma) * np.repeat(df, components)
    )


def phi_matrix(freqs, log10_A_gw, log10_A_rn, gamma_rn, gamma_gw=4.33):
    """
    NO CROSS CORRELATIONS
    """
    rho = powerlaw(freqs, log10_A_gw, gamma_gw)
    kappa = powerlaw(freqs, log10_A_rn, gamma_rn)
    phi = rho + kappa

    return phi



def phi_tot_matrix(phi, Mmat):
    """
    Note that this is somewhat different than shown in the literature.
    Phi = B with the infinities and phi parts swapped.
    """
    phim = np.ones(Mmat.shape[1]) * mpf("1e40")
    return np.concatenate((phi, phim))


def normed_timing_matrix(Mmat):
    """
    Normalize the timing design matrix
    """
    norm = np.sqrt(np.sum(Mmat ** 2, axis=0))

    nmat = Mmat / norm
    nmat[:, norm == 0] = 0

    return nmat


def get_T(fmat, norm_tmatrix):
    return np.hstack((fmat, norm_tmatrix))


def get_TNT(T, wnoise_mat_inv):
    return np.matmul(T.T, np.matmul(wnoise_mat_inv, T))


























