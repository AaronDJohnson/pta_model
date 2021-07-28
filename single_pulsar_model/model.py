# import autograd.numpy as np
import numpy as np


def fourier_matrix(toas, num_freqs=30):
    time_span = np.max(toas) - np.min(toas)
    # freq_min = 1 / time_span
    # freq_max = num_freqs / time_span

    num_toas = len(toas)
    matrix = np.zeros((num_toas, 2 * num_freqs))

    freqs = np.arange(1, num_freqs + 1) / time_span
    omega = 2 * np.pi * freqs

    arg = np.outer(toas, omega)
    matrix[:,::2] = np.sin(arg)
    matrix[:,1::2] = np.cos(arg)
    return matrix, np.repeat(freqs, 2)


def powerlaw(freqs, log10_A, gamma, ref_freq=3.168808781402895e-08, components=2):
    """
    default ref_freq is 1 / yr in hz
    components is the number of repetitions of freqs
    """
    df = np.diff(np.concatenate((np.array([0]), freqs[::components])))
    return (
        (10**log10_A)**2 / (12 * np.pi**2) * ref_freq ** (gamma - 3) * freqs ** (-gamma) * np.repeat(df, components)
    )


def phi_matrix(freqs, log10_A_gw, log10_A_rn, gamma_rn, gamma_gw=4.33):
    """
    NO CROSS CORRELATIONS
    """
    rho = powerlaw(freqs, log10_A_gw, gamma_gw)
    print(rho)
    kappa = powerlaw(freqs, log10_A_rn, gamma_rn)
    phi = rho + kappa

    return phi


def phi_tot_matrix(phi, Mmat):
    """
    Note that this is somewhat different than shown in the literature.
    Phi = B with the infinities and phi parts swapped.
    """
    phim = np.ones(Mmat.shape[1]) * 1e40
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


# normal way of doing things:
def get_TNT(T, wnoise_mat_inv):
    return np.matmul(T.T, np.matmul(np.diag(wnoise_mat_inv), T))


def make_sigma(TNT, phi):
    return np.diag(phi**(-1)) + TNT


def get_rNT(T, N_inv, res):
    return np.matmul(res.T, np.matmul(np.diag(N_inv), T))


def get_TNr(T, N, res):
    return np.matmul(T.T, np.matmul(np.diag(N), res))


def get_rNr(N_inv, res):
    return np.matmul(res.T, np.matmul(np.diag(N_inv), res))


def get_C_inv(N_inv, Sigma_inv, TNr):
    return np.diag(N_inv) - np.matmul(TNr.T, np.matmul(Sigma_inv, TNr))


def logdet_N(wnoise_matrix_inv):
    return -np.sum(np.log(wnoise_matrix_inv))


def logdet_C(phi, sigma):
    logdet_phi = np.sum(np.log(phi))
    logdet_Sigma = np.sum(np.log(np.diagonal(sigma)))

    return logdet_phi + logdet_Sigma


class PulsarLogLikelihood():
    def __init__(self, psr):
        # get the relevant pieces from TEMPO2 pulsar object
        self.toas = psr.toas.copy()
        self.res = psr.residuals.copy()
        self.toaerrs = psr.toaerrs.copy()
        self.timing_matrix = psr.Mmat.copy()

        # setup fourier design matrix
        self.fmat, self.freqs = fourier_matrix(self.toas)
        # print(self.fmat)

        # get the white noise signal (1d array)
        self.wnoise_mat = self.toaerrs**2
        self.wnoise_mat_inv = self.toaerrs**(-2)

        # normalize the timing matrix
        self.norm_tmatrix = normed_timing_matrix(self.timing_matrix)

        # make T
        self.tmat = np.hstack((self.fmat, self.norm_tmatrix))

        self.TNT = get_TNT(self.tmat, self.wnoise_mat_inv)
        self.TNr = get_TNr(self.tmat, self.wnoise_mat_inv, self.res)
        # self.rNT = get_rNT(self.tmat, self.toas**2, self.res)
        self.rNr = get_rNr(self.wnoise_mat_inv, self.res)

        self.N_det = logdet_N(self.wnoise_mat_inv)

        print(self.rNr)

        self.lnlike = -0.5 * self.rNr - 0.5 * self.N_det

        self.b = np.identity(self.tmat.shape[1])


    def __call__(self, x):
        # phi
        gamma_rn, log10_A_rn, log10_A_gw = x

        # 20 microseconds: --
        phi = phi_matrix(self.freqs, log10_A_gw, log10_A_rn, gamma_rn)
        # phi_total -> basically B but with infinities and phi swapped
        phi = phi_tot_matrix(phi, self.norm_tmatrix)
        # --
        print(phi)

        # 21 microseconds: --
        sigma = make_sigma(self.TNT, phi)
        # --

        # sigma_inv, cf = get_sigma_inv(sigma, self.b)
        # # inverse (the long part! ... can be made shorter with scipy.sparse)
        # 1.13 ms (BAD)
        cf = np.linalg.cholesky(sigma)
        c = np.linalg.solve(cf, self.b)
        sigma_inv = np.dot(c.T, c)
        # print(sigma_inv)
        # --
        
        # the rest of this takes a small fraction of the total time on average (0.17ms)
        logdet_sigma = np.sum(2 * np.log(np.diag(cf)))
        logdet_phi = np.sum(np.log(phi))

        # slow mat mul (speed up with scipy.sparse)
        lnlike = self.lnlike + (0.5 * np.dot(self.TNr, np.matmul(sigma_inv, self.TNr)) - 0.5 * (logdet_phi + logdet_sigma))
        return lnlike


# class LogPrior():
#     def __init__(self, log10_A_gw, log10_A_rn, gamma_rn):






