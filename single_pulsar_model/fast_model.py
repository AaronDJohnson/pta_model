import numpy as np


def normed_timing_matrix(Mmat):
    """
    Normalize the timing design matrix
    """
    norm = np.sqrt(np.sum(Mmat ** 2, axis=0))

    nmat = Mmat / norm
    nmat[:, norm == 0] = 0

    return nmat


def power_law(params, freqs, ref_freq=3.168808781402895e-08, repeats=2):
    """
    default ref_freq is 1 / yr in Hz
    repeats is the number of repetitions of freqs
    """
    gamma = params[0]
    log10_A = params[1]

    prefactor = ref_freq**(-3) / (12 * np.pi**2)

    df = np.diff(np.concatenate((np.array([0]), freqs[::repeats])))
    df = np.repeat(df, repeats)  # (Nfreq, )

    freq_ratio = np.power.outer((ref_freq / freqs), gamma)  # (Nfreq, Ngamma)

    amp = 10**(2 * log10_A)  # (Namp, )

    freq_term = freq_ratio * df[:, np.newaxis]  # (Nfreq, Ngamma)

    total = np.multiply(amp[:, np.newaxis], freq_term.T)

    return prefactor * total  # (Namp, Ngamma, Nfreqs)


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


def chi_matrix(params_gw, params_rn, freqs):
    """
    NO CROSS CORRELATIONS
    """
    rho = power_law(params_gw, freqs)
    kappa = power_law(params_rn, freqs)
    phi = rho + kappa

    return phi


def make_D_inv(N_inv, M):
    N_inv = np.diag(N_inv)
    A = np.matmul(N_inv, M)
    B = A.T
    S = np.matmul(M.T, np.matmul(N_inv, M))

    cf = np.linalg.cholesky(S)
    c = np.linalg.solve(cf, np.identity(S.shape[0]))
    S_inv = np.dot(c.T, c)
    logdet_S = np.sum(2 * np.log(np.diag(cf)))
    D_inv = N_inv - np.matmul(A, np.matmul(S_inv, B))
    return logdet_S, S, D_inv

def get_rDr(r, D_inv):
    return np.matmul(r.T, np.matmul(D_inv, r))

def get_FDF(F, D_inv):
    return np.matmul(F.T, np.matmul(D_inv, F))

def get_FDr(r, D_inv, F):
    return np.matmul(F.T, np.matmul(D_inv, r))


def make_new_sigma(chi, FDF):
    chi_inv = chi**(-1)
    temp_list = []
    for mat in chi_inv:
        temp_list.append(np.diag(mat))
    chi_inv = np.array(temp_list)
    return chi_inv + FDF[np.newaxis, :]


class FastLogLikelihood():
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

        self.logdet_S, self.S, self.D_inv = make_D_inv(self.wnoise_mat_inv, self.norm_tmatrix)
        self.FDF = get_FDF(self.fmat, self.D_inv)
        self.FDr = get_FDr(self.res, self.D_inv, self.fmat)

        self.b = np.identity(60)
        rDr = get_rDr(self.res, self.D_inv)
        self.lnlike0 = -0.5 * rDr
        logdet_E = self.norm_tmatrix.shape[1] * np.log(1e40)
        self.logdet_D = self.logdet_S + np.sum(np.log(self.wnoise_mat)) + logdet_E


    def __call__(self, params_gw, params_rn):

        # 20 microseconds: --
        chi = chi_matrix(params_gw, params_rn, self.freqs)
        # --

        # 21 microseconds: --
        sigma = make_new_sigma(chi, self.FDF)
        # --

        # inverse (the long part! ... can be made shorter with scipy.sparse cholesky)
        # 700 microseconds: --
        # cf = np.linalg.cholesky(sigma)
        # print(cf)
        ident = np.zeros((params_gw.shape[1], 60, 60))
        A = np.einsum('...ii->...i', ident)
        A[:] = 1
        sigma_inv = np.linalg.solve(sigma, ident)
        # sigma_inv = np.dot(c.T, c)
        # --

        logdet_sigma = -np.linalg.slogdet(sigma_inv)[1]
        logdet_chi = np.sum(np.log(chi), axis=1)
        # slow mat mul (speed up with scipy.sparse)
        # print(self.lnlike0)
        # print(0.5 * np.dot(self.FDr, np.matmul(sigma_inv, self.FDr)))
        # print(- 0.5 * (logdet_phi + logdet_sigma + self.logdet_D))
        lnlike = self.lnlike0 + 0.5 * np.matmul(np.matmul(sigma_inv, self.FDr), self.FDr) - 0.5 * (logdet_chi + logdet_sigma + self.logdet_D)

        return lnlike