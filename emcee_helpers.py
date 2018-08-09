from __future__ import print_function, unicode_literals
import numpy as np
import tables
import pickle


def emcee_best(sampler):
    """ return the 'best-fit' from emcee sampler

    Parameters
    ----------
    sampler: emcee.EnsembleSampler
        sampler from emcee

    Returns
    -------
    best: ndarray[DTYPE, ndim=1]
        best fit parameter vector
    """
    return np.array([ np.median(sampler.flatchain[:, k]) for k in range(sampler.flatchain.shape[1]) ])


def emcee_plots(lnp_fn, sampler, theta0=None, figout=True, plt=None,
                marginalized=True, triangle=True, chains=True):
    """ Plots diagnostic figures """
    if plt is None:
        import pylab as plt

    from figrc import plotMAP, plotCorr, plotDensity
    import figrc

    d = {}

    ndim = lnp_fn.ndim
    theta_names = [k.replace('_', ' ') for k in lnp_fn.theta_names]

    # making MAPs
    if marginalized is True:
        print('Creating figure: MAPs')
        nlines = 4
        plt.figure(figsize=((ndim // nlines + ndim % nlines) * 5,  5. * nlines ))

        for k in range(ndim):
            ax = plt.subplot(nlines, ndim // nlines + ndim % nlines, k + 1)
            try:
                plotMAP(sampler.flatchain[:, k], ax=ax, usehpd=False)
            except Exception as e:
                print('Issue with MAP of {0:s}: {1}'.format(theta_names[k], e))
            figrc.hide_axis(('right', 'top', 'left'), ax=ax)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel('')
            figrc.setNmajors(3, 4)

            if theta0 is not None:
                lim = ax.get_ylim()
                ax.vlines([theta0[k]], lim[0], lim[1], color='r')
                ax.vlines([np.median(sampler.flatchain[:, k])], lim[0], lim[1], color='b')
                ax.set_ylim(lim)
            ax.set_xlabel(theta_names[k])
            d[theta_names[k]] = sampler.flatchain[:, k]
        if figout is not None:
            plt.savefig('marginalized_' + figout, bbox_inches='tight')

    if triangle is True:
        print('Creating figure: Correlation plot (aka triangle plot)')
        plt.figure(figsize=(3 * len(theta_names), 3 * len(theta_names)))
        limits = {}
        plotCorr(d, theta_names, plotfunc=plotDensity, devectorize=True, limits=limits)
        if figout is not None:
            plt.savefig('triangle_' + figout, bbox_inches='tight')

    if chains is True:
        print('Creating figure: Chain diagnostics')
        plt.figure(figsize=(10, 7))
        chains = range(min([40, len(sampler.chain)]))
        for ak in range(ndim):
            ax = plt.subplot(2, ndim // 2 + ndim % 2, ak + 1)
            for i in chains:
                ax.plot(sampler.chain[i, :, ak])
            ax.set_xlabel('steps')
            ax.set_ylabel(theta_names[ak])
        if figout is not None:
            plt.savefig('chains_' + figout, bbox_inches='tight')


class SamplerData(object):
    """
    Dummy Emcee sampler that keeps the data in a similar manner
    but can be saved to and restored from disk
    It also keeps the initialization values

    Attributes
    ----------

    Initialization with a Sampler

    sampler: sampler instance
        emcee sampler

    theta0: ndarray
        initial guess

    Initialization from a file

    hdfname: str
        hdf file storing the sampler
    """
    def __init__(self, *args, **kwargs):
        if type(args[0]) == str:
            self.init_from_file(*args)
        else:
            assert(len(args) == 2), 'wrong input from sampler'
            self.init_from_sampler(*args)

    def init_from_sampler(self, sampler, theta0):
        self.theta0 = theta0
        self.chain = sampler.chain
        self.lnprobability = sampler.lnprobability
        self.naccepted = sampler.naccepted
        self.iterations = sampler.iterations
        self.attrs = {}

    def init_from_file(self, hdfname):
        with tables.open_file(hdfname, 'r') as hd5:
            self.theta0 = hd5.root.theta0[:]
            self.chain = hd5.root.sampler.chain[:]
            self.lnprobability = hd5.root.sampler.lnprobability[:]
            self.naccepted = hd5.root.sampler.naccepted
            self.iterations = hd5.root.sampler.iterations[:]
            try:
                self.attrs = pickle.loads(hd5.root.attrs[:][0])
            except UnicodeDecodeError:
                self.attrs = pickle.loads(hd5.root.attrs[:][0], encoding='latin-1' )

    @property
    def dim(self):
        return len(self.theta0)

    @property
    def acceptance_fraction(self):
        """
        The fraction of proposed steps that were accepted.
        """
        return self.naccepted / self.iterations

    @property
    def flatlnprobability(self):
        """
        A shortcut to return the equivalent of ``lnprobability`` but aligned
        to ``flatchain`` rather than ``chain``.
        """
        return self.lnprobability.flatten()

    @property
    def flatchain(self):
        """
        A shortcut for accessing chain flattened along the zeroth (walker)
        axis.
        """
        s = self.chain.shape
        return self.chain.reshape(s[0] * s[1], s[2])

    @property
    def acor(self):
        """
        The autocorrelation time of each parameter in the chain (length:
        ``dim``) as estimated by the ``acor`` module.

        """
        try:
            import acor
        except ImportError:
            raise ImportError("acor")
        s = self.dim
        t = np.zeros(s)
        for i in range(s):
            t[i] = acor.acor(self.chain[:, :, i])[0]
        return t

    def writeto(self, outname):
        """ save the sampler information for later use """
        with tables.open_file(outname, 'w') as hd5:
            grp = hd5.create_group('/', 'sampler', title="Sampler")
            hd5.create_array(grp, 'chain', self.chain)
            hd5.create_array(grp, 'lnprobability', self.lnprobability)
            hd5.create_array(grp, 'naccepted', self.naccepted)
            hd5.create_array(grp, 'iterations', np.atleast_1d(np.array(self.iterations)))
            hd5.create_array('/', 'theta0', self.theta0)
            hd5.create_array('/', 'attrs', np.atleast_1d(np.array(pickle.dumps(self.attrs)).astype('|S')))
            hd5.flush()


def sample_ball_init(theta0, nwalkers, lnp_fn, w_pos_noise=1e-3, var=10, maxiter=5, args=(), kwargs={}):
    """
    Produce a ball of walkers around an initial parameter value with a limited
    posterior value range to assure good initialization.

    Basically, it makes sure that the initial variance of the walkers' lnp
    values is is not more than var magnitudes (default 10)

    Hogg would say "Brutal!"

    Parameters
    ----------
    theta0: ndarray
        initial parameter vector

    nwalkers: int
        number of walkers to initialize

    lnp_fn: callable
        posterior or likelihood function

    w_pos_noise: float
        initial ball size that will be refined if necessary

    var: float
        variance criteria (default 10 magnitudes)

    Returns
    -------
    p0: sequence
        walker initialization postions
    """
    # make sure that the initial variance of the walkers' lnp values is is not
    # more than 10 magnitudes
    ndim = len(theta0)
    pvar = 2. * var

    niter = 0
    print('Emcee walkers initialization')
    while ((pvar - var > 0) & (niter < maxiter)):
        niter += 1
        print('\t iteration {1:d}, dispersion={0:e}, pvar={2:f} > {3:0.1f}'.format(w_pos_noise, niter, pvar, var))
        p0 = np.array([theta0 * np.random.normal(1, w_pos_noise, ndim) for i in range(nwalkers)])
        lnps = np.asarray([float(lnp_fn(p, *args, **kwargs)) for p in p0])
        test_inf = np.isfinite(lnps)
        while False in test_inf:
            print('\t\t reaffecting -inf for {0:d} starting points'.format(sum(~test_inf)))
            newp = np.array([theta0 * np.random.normal(1., w_pos_noise, ndim) for i in range(np.sum(~test_inf))])
            newlnp = np.asarray([float(lnp_fn(p, *args, **kwargs)) for p in newp])
            p0[~test_inf] = newp[:]
            lnps[~test_inf] = newlnp
            test_inf = np.isfinite(lnps)
        pvar = np.var(lnps)
        w_pos_noise /= 10.

    print('\t final dispersion={0:e}, pvar={1:e}'.format(w_pos_noise, pvar))
    print('Emcee walkers initialization ready')
    return p0


def trim_chains(pos, prob, ndim):
    """ Trim chains by proposing new positions to walkers running away

    Parameters
    ----------
    pos: array
        positions of the chains

    prob: array
        lnp associated to each chain position

    ndim: int
        number of dimensions (e.g, sampler.dim)

    Returns
    -------
    pos: ndarray
        new positions of all the chains with constrained run-aways
    """
    trim_threshold = 3 * np.std(prob)
    reset_ind = np.abs(prob - prob.max()) > trim_threshold
    reset_N = sum(reset_ind)
    best_ind = (prob == prob.max())

    print('Trimming chains: reaffecting {0:d} chains'.format(reset_N))

    best_N = sum(best_ind)

    # randomly affect chains to best values
    newpos = pos[best_ind][np.random.randint(0, best_N, reset_N)]
    # Add noise
    newpos += newpos * np.random.normal(1, 1e-4, (reset_N, ndim))
    pos[reset_ind] = newpos

    return pos


try:
    from pbar import Pbar

    def run_mcmc_with_pbar(sampler, pos0, N, rstate0=None, lnprob0=None, desc=None,
                           **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result while also
        showing a progress bar

        Parameters
        ----------
        pos0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.

        N:
            The number of steps to run.

        lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        desc: str (optional)
            title of the progress bar

        kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        Returns
        -------
        t: tuple
            This returns the results of the final sample in whatever form
            :func:`sample` yields.  Usually, that's: ``pos``, ``lnprob``,
            ``rstate``, ``blobs`` (blobs optional)
        """
        if pos0 is None:
            if sampler._last_run_mcmc_result is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            pos0 = sampler._last_run_mcmc_result[0]
            if lnprob0 is None:
                rstate0 = sampler._last_run_mcmc_result[1]
            if rstate0 is None:
                rstate0 = sampler._last_run_mcmc_result[2]

        with Pbar(maxval=N, desc=desc) as pb:
            k = 0
            for results in sampler.sample(pos0, lnprob0, rstate0, iterations=N,
                                          **kwargs):
                k += 1
                pb.update(k)

        # store so that the ``pos0=None`` case will work.  We throw out the blob
        # if it's there because we don't need it
        sampler._last_run_mcmc_result = results[:3]

        return results
except ImportError:
    pass

try:
    import figrc
    import pylab as plt

    def triangle_figure(sdata, lbls=None, add_lnp=False, ticksrotation=45,
                        gaussian_ellipse=True, figout=True, lnp_cut=None, **kwargs):
        print('Creating figure: Correlation plot (aka triangle plot)')
        if lbls is None:
            lbls = 'all'

        d = {}
        theta_names = sdata.attrs.get('theta_names', None)
        if theta_names is None:
            theta_names = ['p{0:d}'.format(k) for k in range(sdata.dim)]
        if lnp_cut is None:
            for e, k in enumerate(theta_names):
                d[k] = sdata.flatchain[:, e]
        else:
            ind = sdata.flatlnprobability > lnp_cut
            for e, k in enumerate(theta_names):
                d[k] = sdata.flatchain[ind, e]

        if lbls == 'all':
            lbls = theta_names

        if add_lnp:
            lbls += ['lnp']
            if lnp_cut:
                d['lnp'] = sdata.flatlnprobability[ind]
            else:
                d['lnp'] = sdata.flatlnprobability

        plt.figure(figsize=(3 * len(lbls), 3 * len(lbls)))

        labels = kwargs.pop('labels', figrc.raw_string(lbls))

        figrc.triangle_plot(d, lbls, labels=labels,
                            ticksrotation=ticksrotation,
                            gaussian_ellipse=gaussian_ellipse, **kwargs)

        if figout not in (None, 'none', 'None'):
            plt.savefig('triangle_' + figout, bbox_inches='tight')

    def plot_1d_marginal_PDFs(sdata, lbls=None, add_lnp=False, lnp_cut=None,
                              **kwargs):

        print('Creating figure: 1D PDFs')

        if lbls is None:
            lbls = 'all'

        d = {}
        theta_names = sdata.attrs.get('theta_names', None)
        if theta_names is None:
            theta_names = ['p{0:d}'.format(k) for k in range(sdata.dim)]
        if lnp_cut is None:
            for e, k in enumerate(theta_names):
                d[k] = sdata.flatchain[:, e]
        else:
            ind = sdata.flatlnprobability > lnp_cut
            for e, k in enumerate(theta_names):
                d[k] = sdata.flatchain[ind, e]

        if lbls == 'all':
            lbls = theta_names

        if add_lnp:
            lbls += ['lnp']
            if lnp_cut:
                d['lnp'] = sdata.flatlnprobability[ind]
            else:
                d['lnp'] = sdata.flatlnprobability

        xlabels = kwargs.pop('labels', figrc.raw_string(lbls))

        figrc.plot_1d_PDFs(d, lbls, labels=xlabels, **kwargs)

    SamplerData.plot_triangle = triangle_figure
    SamplerData.plot_1d_pdfs = plot_1d_marginal_PDFs
except:
    pass
