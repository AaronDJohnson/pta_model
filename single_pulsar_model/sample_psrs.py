import numpy as np
import glob, pickle

from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals

from enterprise_extensions import sampler, hypermodel


def load_psrs(amp, trial, ephem='DE436'):
    """
    Load pulsar data from par and tim files with TEMPO2.

    Inputs:
        datadir (string): file path containing par and tim folders
        ephem (string): ephemeris name

    Returns:
        psrs (list): list of Pulsar objects
    """
    datadir = '../fake_psrs/fakes_gwb_amp_{0}_trial_{1}'.format(amp, trial)
    try:
        with open(datadir + '/psrs.pkl'.format(amp, trial), 'rb') as f:
            psrs_cut = pickle.load(f)

    except:
        parfiles = sorted(glob.glob(datadir + '/par/' + '*.par'))
        timfiles = sorted(glob.glob(datadir + '/tim/' + '*.tim'))

        psrs = []
        psrs_cut = []
        for p, t in zip(parfiles, timfiles):
            psr = Pulsar(p, t, ephem=ephem)
            psrs.append(psr)
        for p in psrs:
            time_tot = (max(p.toas) - min(p.toas)) / 86400 / 365.25
            if time_tot > 6:
                psrs_cut.append(p)
        
        # with open(datadir + '/psrs.pkl'.format(amp, trial), 'wb') as f:
        #     pickle.dump(psrs, f)
    return psrs_cut


def fake_model_1(psrs, rn='uniform'):
    # Find maximum time span to set GW freq sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters:
    efac = parameter.Constant(1)

    # red noise parameters
    if rn == 'linearexp':
        log10_A = parameter.LinearExp(-20, -11)
    elif rn == 'uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma = parameter.Uniform(0, 7)

    # GW red noise parameters
    # if crn == 'linearexp':
    # 	log10_A_gw = parameter.LinearExp(-20, -12)('log10_A_gw')
    # elif crn == 'uniform':
    # 	log10_A_gw = parameter.Uniform(-20, -12)('log10_A_gw')
    # else:
    # 	print('uniform and linearexp are the only options for red noise right now.')
    # gamma_gw = parameter.Constant(4.33)('gamma_gw')

    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

    # red noise (pl with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # common red noise (pl with 30 frequencies)
    # cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    # gw = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)
    
    # total model
    s = ef + rn + tm

    # initialize PTA
    models = []
    
    for p in psrs:
        models.append(s(p))

    pta = signal_base.PTA(models)
    return pta


def fake_model_2a(psrs, rn='uniform', crn='uniform'):
    # Find maximum time span to set GW freq sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters:
    efac = parameter.Constant(1)

    # red noise parameters
    if rn == 'linearexp':
        log10_A = parameter.LinearExp(-20, -11)
    elif rn == 'uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma = parameter.Uniform(0, 7)

    # GW red noise parameters
    if crn == 'linearexp':
        log10_A_gw = parameter.LinearExp(-20, -12)('log10_A_gw')
    elif crn == 'uniform':
        log10_A_gw = parameter.Uniform(-20, -12)('log10_A_gw')
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')

    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

    # red noise (pl with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # common red noise (pl with 30 frequencies)
    cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)

    
    # total model
    s = ef + rn + gw + tm

    # initialize PTA
    models = []
    
    for p in psrs:
        models.append(s(p))

    pta = signal_base.PTA(models)
    return pta


def fake_model_2a_no_tm(psrs, rn='uniform', crn='uniform'):
    # Find maximum time span to set GW freq sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters:
    efac = parameter.Constant(1)

    # red noise parameters
    if rn == 'linearexp':
        log10_A = parameter.LinearExp(-20, -11)
    elif rn == 'uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma = parameter.Uniform(0, 7)

    # GW red noise parameters
    if crn == 'linearexp':
        log10_A_gw = parameter.LinearExp(-20, -12)('log10_A_gw')
    elif crn == 'uniform':
        log10_A_gw = parameter.Uniform(-20, -12)('log10_A_gw')
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')

    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

    # red noise (pl with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # common red noise (pl with 30 frequencies)
    cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)

    
    # total model
    s = ef + rn + gw # + tm

    # initialize PTA
    models = []
    
    for p in psrs:
        models.append(s(p))

    pta = signal_base.PTA(models)
    return pta


def sample_odds_single(psrs, psr_num, amp, trial, num_points=2e5, neff=10000):
    print('Working on single pulsar number {0}.'.format(psr_num))
    outdir = './odds_sngl/fakes_amp_{0}_trial_{1}/{2}'.format(amp, trial, psrs[psr_num].name)

    nmodels = 2
    mod_index = np.arange(nmodels)

    # Make dictionary of PTAs.
    pta = dict.fromkeys(mod_index)

    pta[0] = fake_model_1(psrs)
    pta[1] = fake_model_2a(psrs)

    super_model = hypermodel.HyperModel(pta)
    sampler = super_model.setup_sampler(outdir=outdir,
                                        sample_nmodel=True)

    x0 = super_model.initial_sample()

    # sampler for N steps
    N = int(num_points)
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, neff=int(neff))


def sample_psrs(psrs, outdir, num_points=1e5, rn='uniform', crn='uniform', neff=2000):
    """
    Samples the pulsars based on a model with no correlations between pulsars.

    Inputs:
        psrs (list): list of pulsars to sample
        outdir (string): where to save the pulsars 
        num_points (int) [1e5]: number of points to sample
        rn (str) ['linearexp']: prior to place on the intrinsic red noise (linearexp or uniform)
        crn (str) ['uniform']: prior to place on the common red noise (linearexp or uniform)
    """
    # Find maximum time span to set GW freq sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters:
    efac = parameter.Constant(1)

    # red noise parameters
    if rn == 'linearexp':
        log10_A = parameter.LinearExp(-20, -11)
    elif rn == 'uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma = parameter.Uniform(0, 7)

    # GW red noise parameters
    if crn == 'linearexp':
        log10_A_gw = parameter.LinearExp(-20, -12)('log10_A_gw')
    elif crn == 'uniform':
        log10_A_gw = parameter.Uniform(-20, -12)('log10_A_gw')
    else:
        print('uniform and linearexp are the only options for red noise right now.')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')

    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

    # red noise (pl with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # common red noise (pl with 30 frequencies)
    cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)
    
    # total model
    s = ef + rn + gw + tm

    # initialize PTA
    models = []
    
    for p in psrs:
        models.append(s(p))

    pta = signal_base.PTA(models)
    sample = sampler.setup_sampler(pta, resume=True, outdir=outdir)
    x0 = np.hstack([p.sample() for p in pta.params])

    # sampler for N steps
    N = int(num_points)
    sample.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, neff=int(neff))


def sample_multiple(psrs, num_psrs, amp, trial, num_points=1e5, rn='uniform', crn='uniform', neff=10000):
    print('Working on {0} pulsars.'.format(len(psrs[:num_psrs + 1])))
    outdir = './chains_mult/fakes_amp_{0}_trial_{1}/{2}psrs'.format(amp, trial, len(psrs[:num_psrs + 1]))
    psrs_cut = psrs[:num_psrs + 1]
    sample_psrs(psrs_cut, outdir, num_points=num_points, rn=rn, crn=crn, neff=neff)


def sample_single(psrs, amp, trial, num_points=1e5, rn='uniform', crn='uniform', neff=10000):
    for psr_num in range(len(psrs)):
        print('Working on single pulsar number {0}.'.format(psr_num))
        outdir = './chains_sngl_{0}/fakes_amp_{1}_trial_{2}/{3}'.format(len(psrs), amp, trial, psrs[psr_num].name)
        sample_psrs([psrs[psr_num]], outdir, num_points=num_points, rn=rn, crn=crn, neff=neff)


