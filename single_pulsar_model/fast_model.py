import numpy as np
from model import fourier_matrix, normed_timing_matrix, phi_matrix


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


def make_new_sigma(phi, FDF):
    phi_inv = np.diag(phi**(-1))
    return phi_inv + FDF


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


    def __call__(self, x):
        # phi
        gamma_rn, log10_A_rn, log10_A_gw  = x

        # 20 microseconds: --
        phi = phi_matrix(self.freqs, log10_A_gw, log10_A_rn, gamma_rn)
        # --

        # 21 microseconds: --
        sigma = make_new_sigma(phi, self.FDF)
        # --

        # inverse (the long part! ... can be made shorter with scipy.sparse cholesky)
        # 700 microseconds: --
        cf = np.linalg.cholesky(sigma)
        c = np.linalg.solve(cf, self.b)
        sigma_inv = np.dot(c.T, c)
        # --

        logdet_sigma = np.sum(2 * np.log(np.diag(cf)))
        logdet_phi = np.sum(np.log(phi))
        # slow mat mul (speed up with scipy.sparse)
        # print(self.lnlike0)
        # print(0.5 * np.dot(self.FDr, np.matmul(sigma_inv, self.FDr)))
        # print(- 0.5 * (logdet_phi + logdet_sigma + self.logdet_D))
        lnlike = self.lnlike0 + 0.5 * np.dot(self.FDr, np.matmul(sigma_inv, self.FDr)) - 0.5 * (logdet_phi + logdet_sigma + self.logdet_D)

        return lnlike