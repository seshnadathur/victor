import numpy as np
import scipy.interpolate as si
from scipy.special import legendre

class InputError(Exception):

    """Error raised when something is wrong with the input data"""

def multipoles_from_fn(frmu, r, ell=[0, 2, 4], even=True, npts=200):
    """
    Calculate the Legendre multipoles of a function evaulated at given vector r

    Parameters
    ----------
    frmu : function handle
        A function f(r, mu), where r is radial distance and mu is cosine of the angle theta

    r :  array
        A 1D array of length N, the radial distances at which to evaluate the multipoles

    ell : list or array of ints, default=[0, 2, 4]
        Legendre multipoles to evaluate

    even : bool, default=True
        Whether f(r, mu) is even or odd in mu – for even functions integrals are performed over [0, 1] and doubled.
        (Protects against out-of-bounds errors on interpolating functions built on mu within [0, 1] only)

    npts : int, default=200
        Number of values of mu to use when evaluating integrals

    Returns
    -------
    multipoles : dict
        Dictionary of multipoles as function of `r`; keys are multipole orders and each contains an array of length N
    """

    try: # failsafe in case just a single int is passed
        x = len(ell)
    except TypeError:
        ell = np.array([ell])
    multipoles = {}
    for l in ell:
        multipoles[f'{l}'] = np.zeros(len(r))

    if even:
        mu = np.linspace(0.0, 1.0, npts)
        factors = [2*l+1 for l in ell]
    else:
        mu = np.linspace(-1, 1, npts)
        factors = [(2*l+1) / 2 for l in ell]

    for i, l in enumerate(ell):
        lmu = legendre(l)(mu)
        for j in range(len(r)):
            y = frmu(r[j], mu).T[0]
            multipoles[f'{l}'][j] = factors[i] * np.trapz(y * lmu, mu)

    return multipoles

def fn_from_multipoles(r, poles, multipoles, npts=200):
    """
    Use multipole information to construct the function f(r, mu)

    Parameters
    ----------
    r : array
        1D array of length N with radial distances at which multipoles were evaluated

    poles : list or array of ints
        Order of the Legendre multipoles provided

    multipoles : 2D array
        Array of shape (M, N) where M is the length of `poles`. Each row should contain the values
        of the corresponding multipole evaluated at the radial distances contained in `r`

    npts : int, default=200
        Number of values of mu (in the range [-1, 1]) to use when building interpolator

    Returns
    -------
    func : instance of :class:`scipy.interpolate.interp2d`
        An interpolating function for f(r, mu)
    """

    poles = [poles] if isinstance(poles, int) else poles # failsafe in case just a single int is passed
    # check input
    if not multipoles.shape == (len(poles), len(r)):
        raise ValueError(f'Wrong shape of multipoles: expected ({len(poles)}, {len(r)}), but received {multipoles.shape}')
    mu = np.linspace(-1, 1, npts)
    func_grid = np.zeros((len(mu), len(r)))
    for i, ell in enumerate(poles):
        lmu = legendre(ell)(mu).reshape((len(mu), 1))
        func_grid += lmu * multipoles[i,:]
    func = si.interp2d(r, mu, func_grid)
    return func

def convert_old_model_files_to_hdf5(realspace_ccf_file, output_model_file, matter_ccf_file=None, velocity_file=None, beta_file=None):
    """
    Convert the set of model input files used for Victor from the old format to new
    """

    import h5py

    with h5py.File(output_model_file, 'w') as f:
        real_ccf = np.load(realspace_ccf_file, allow_pickle=True).item()
        r = real_ccf['rvals']
        f.create_dataset('r', data=r)
        real_multipoles = real_ccf['multipoles']
        if beta_file is not None:
            beta = np.load(beta_file, allow_pickle=True)
            f.create_dataset('beta', data=beta)
            f.create_dataset('monopole', data=real_multipoles[:, :int(real_multipoles.shape[1]/2)])
            f.create_dataset('quadrupole', data=real_multipoles[:, int(real_multipoles.shape[1]/2):])
        else:
            f.create_dataset('monopole', data=real_multipoles[:int(real_multipoles.shape[1]/2)])
            f.create_dataset('quadrupole', data=real_multipoles[int(real_multipoles.shape[1]/2):])

        if matter_ccf_file is not None:
            matter_ccf = np.load(matter_ccf_file, allow_pickle=True).item()
            r_for_delta = matter_ccf['rvals']
            delta = matter_ccf['delta']
            f.create_dataset('rdelta', data=r_for_delta)
            f.create_dataset('delta', data=delta)

        if velocity_file is not None:
            velocity = np.load(velocity_file, allow_pickle=True).item()
            r_for_sv = velocity['rvals']
            sigmav = velocity['sigma_v_los']
            f.create_dataset('rsv', data=r_for_sv)
            f.create_dataset('sigmav', data=sigmav)

def convert_old_data_files_to_hdf5(redshift_ccf_file, output_data_file, beta_file=None, covmat_file=None, output_covmat_file=None, beta_cov_file=None):
    """
    Convert the set of data files used for Victor from the old format to new
    """

    import h5py

    with h5py.File(output_data_file, 'w') as f:
        redshift_ccf = np.load(redshift_ccf_file, allow_pickle=True).item()
        s = redshift_ccf['rvals']
        f.create_dataset('s', data=s)
        redshift_multipoles = redshift_ccf['multipoles']
        if beta_file is not None:
            beta = np.load(beta_file, allow_pickle=True)
            f.create_dataset('beta', data=beta)
            f.create_dataset('monopole', data=redshift_multipoles[:, :int(redshift_multipoles.shape[1]/2)])
            f.create_dataset('quadrupole', data=redshift_multipoles[:, int(redshift_multipoles.shape[1]/2):])
        else:
            f.create_dataset('monopole', data=redshift_multipoles[:int(redshift_multipoles.shape[1]/2)])
            f.create_dataset('quadrupole', data=redshift_multipoles[int(redshift_multipoles.shape[1]/2):])

    if covmat_file is not None:
        covmat = np.load(covmat_file, allow_pickle=True)
        with h5py.File(output_covmat_file, 'w') as f:
            if beta_cov_file is not None:
                beta = np.load(beta_cov_file, allow_pickle=True)
                f.create_dataset('beta', data=beta)
            f.create_dataset('covmat', data=covmat)

def convert_hans_quijote_to_hdf5(input_fn, output_fn, reconvoids=True):

    import h5py
    import json

    with open(input_fn, 'rb') as json_file:
        data = json.load(json_file)

    txt = 'RECON' if reconvoids else 'REAL'
    r = np.array(data[0][f'CCF_multipole_Halo_{txt}_Void_{txt}_radius'])
    s = np.array(data[0][f'CCF_multipole_Halo_RSD_Void_{txt}_radius'])
    rdelta = np.array(data[0][f'profile_DM_REAL_Void_{txt}_radius'])
    rh = np.array(data[0][f'profile_Halo_REAL_Void_{txt}_radius'])

    tmp = np.array(data[0][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi0'])
    xir0 = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi2'])
    xir2 = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi4'])
    xir4 = np.empty((len(data), len(tmp)))

    tmp = np.array(data[0][f'CCF_multipole_Halo_RSD_Void_{txt}_xi0'])
    xis0 = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'CCF_multipole_Halo_RSD_Void_{txt}_xi2'])
    xis2 = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'CCF_multipole_Halo_RSD_Void_{txt}_xi4'])
    xis4 = np.empty((len(data), len(tmp)))

    tmp = np.array(data[0][f'profile_DM_REAL_Void_{txt}_delta'])
    delta = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'profile_DM_REAL_Void_{txt}_Delta'])
    int_delta = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'profile_Halo_REAL_Void_{txt}_v'])
    v = np.empty((len(data), len(tmp)))
    tmp = np.array(data[0][f'profile_Halo_REAL_Void_{txt}_sigma'])
    sigmav = np.empty((len(data), len(tmp)))

    with h5py.File(output_fn, 'w') as f:
        f.create_dataset('r', data=r)
        f.create_dataset('s', data=s)
        f.create_dataset('rdelta', data=rdelta)
        f.create_dataset('rv', data=rh)
        f.create_dataset('rsv', data=rh)
        for i in range(len(data)):
            xir0[i] = np.array(data[i][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi0'])
            xir2[i] = np.array(data[i][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi2'])
            xir4[i] = np.array(data[i][f'CCF_multipole_Halo_{txt}_Void_{txt}_xi4'])
            xis0[i] = np.array(data[i][f'CCF_multipole_Halo_RSD_Void_{txt}_xi0'])
            xis2[i] = np.array(data[i][f'CCF_multipole_Halo_RSD_Void_{txt}_xi2'])
            xis4[i] = np.array(data[i][f'CCF_multipole_Halo_RSD_Void_{txt}_xi4'])
            delta[i] = np.array(data[i][f'profile_DM_REAL_Void_{txt}_delta'])
            int_delta[i] = np.array(data[i][f'profile_DM_REAL_Void_{txt}_Delta'])
            v[i] = np.array(data[i][f'profile_Halo_REAL_Void_{txt}_v'])
            sigmav[i] = np.array(data[i][f'profile_Halo_REAL_Void_{txt}_sigma'])
        f.create_dataset('xi0_r', data=xir0)
        f.create_dataset('xi2_r', data=xir2)
        f.create_dataset('xi4_r', data=xir4)
        f.create_dataset('xi0_s', data=xis0)
        f.create_dataset('xi2_s', data=xis2)
        f.create_dataset('xi4_s', data=xis4)
        f.create_dataset('delta', data=delta)
        f.create_dataset('Delta', data=int_delta)
        f.create_dataset('vr', data=v)
        f.create_dataset('sigmav', data=sigmav)
        f.create_dataset('average_xi0_r', data=np.mean(xir0, axis=0))
        f.create_dataset('average_xi2_r', data=np.mean(xir2, axis=0))
        f.create_dataset('average_xi4_r', data=np.mean(xir4, axis=0))
        f.create_dataset('average_xi0_s', data=np.mean(xis0, axis=0))
        f.create_dataset('average_xi2_s', data=np.mean(xis2, axis=0))
        f.create_dataset('average_xi4_s', data=np.mean(xis4, axis=0))
        f.create_dataset('average_delta', data=np.mean(delta, axis=0))
        f.create_dataset('average_Delta', data=np.mean(int_delta, axis=0))
        f.create_dataset('average_vr', data=np.mean(v, axis=0))
        f.create_dataset('average_sigmav', data=np.mean(sigmav, axis=0))

        data_vec = np.hstack([xis0, xis2, xis4])
        covmat = np.cov(data_vec, rowvar=False)
        f.create_dataset('D_ell024_covmat', data=covmat)

        data_vec = np.hstack([xis0, xis2])
        covmat = np.cov(data_vec, rowvar=False)
        f.create_dataset('D_ell02_covmat', data=covmat)
