import numpy as np
from enterprise.signals import utils
import scipy.linalg as sl
import scipy.sparse as sps
from sksparse.cholmod import cholesky


class FastLogLikelihood(object):
    """
    FOR USE WITH RED NOISE ONLY (common signals are required)
    Don't include the timing model. Seriously. Don't.
    """
    def __init__(self, pta, psrs):
        self.pta = pta
        for key in pta.signals:
            if 'timing_model' in key:
                raise Exception("Timing model detected in PTA object. Remove timing model to continue.")
        # get arrays of N and M, residuals
        N = pta.get_ndiag()
        N_inv = [np.diag(N**(-1)) for N in N]
        Mmat = [utils.normed_tm_basis(psrs[ii].Mmat)[0] for ii in range(len(psrs))]
        self.r = pta.get_residuals()

        # T = F when timing model isn't included (so don't include it!)
        self.F = pta.get_basis(pta.params)

        # compute D_inv and add to list
        self.D_inv = []
        self.FDFs = []
        self.FDrs = []

        logdet_Ds = []
        rDrs = []

        for ii in range(len(psrs)):
            left_mult = np.matmul(N_inv[ii], Mmat[ii])
            # S is the matrix to be inverted and its determinant found (if we want to use this)
            S = np.matmul(Mmat[ii].T, np.matmul(N_inv[ii], Mmat[ii]))
            cf = np.linalg.cholesky(S)
            c = np.linalg.solve(cf, np.identity(S.shape[0]))
            S_inv = np.dot(c.T, c)
            logdet_S = np.sum(2 * np.log(np.diag(cf)))
            Di = N_inv[ii] - np.matmul(left_mult, np.matmul(S_inv, left_mult.T))
            self.D_inv.append(Di)

            self.FDFs.append(np.matmul(self.F[ii].T, np.matmul(Di, self.F[ii])))
            self.FDrs.append(np.matmul(self.F[ii].T, np.matmul(Di, self.r[ii])))

            rDrs.append(np.matmul(self.r[ii].T, np.matmul(self.D_inv[ii], self.r[ii])))
            logdet_E = Mmat[ii].shape[1] * np.log(1e40)
            logdet_Ds.append(logdet_S + np.sum(np.log(N[ii])) + logdet_E)

        self.lnlikelihood0 = 0
        for ii in range(len(rDrs)):
            self.lnlikelihood0 += -0.5 * rDrs[ii] - 0.5 * logdet_Ds[ii]

    def _make_sigma(FDFs, phiinv):
        return sps.block_diag(FDFs, "csc") + sps.csc_matrix(phiinv)

    def __call__(self, xs, phiinv_method='cliques'):
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        loglike = self.lnlikelihood0

        phiinvs = self.pta.get_phiinv(params, logdet=True, method=phiinv_method)

        if self.pta._commonsignals:
            print('common signals triggered')
            phiinv, logdet_phi = phiinvs
            Sigma = self._make_sigma(self.FDFs, phiinv)
            FDr = np.concatenate(self.FDrs)
            try:
                cf = cholesky(Sigma)
                expval = cf(FDr)
            except:
                return -np.inf

            logdet_sigma = cf.logdet()
            loglike += 0.5 * (np.dot(FDr, expval) - logdet_sigma - logdet_phi)

        else:
            for FDr, FDF, pl in zip(self.FDrs, self.FDFs, phiinvs):
                if FDr is None:
                    continue

                phiinv, logdet_phi = pl
                Sigma = FDF + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, FDr)
                except:
                    return -np.inf

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
                loglike += 0.5 * (np.dot(FDr, expval) - logdet_sigma - logdet_phi)

        return loglike