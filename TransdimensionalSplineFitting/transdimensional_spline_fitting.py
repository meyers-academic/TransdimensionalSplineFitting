import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d
from scipy.stats import norm
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm

class BaseSplineModel(object):
    def __init__(self, data, N_possible_knots, xrange, height_prior_range, interp_type='linear'):
        """
        Params:
        ------
        data : `object`
            Abstract data object that gets passed to the likelihood
            and used in calculating the likelihood
        N_possible_knots : `int`
            number of possible knots
        xrange : `tuple`
            low and high values on the x-axis  (low, high)
            between which we place the N knots.
        height_prior_range : `tuple`
            low and high values for the uniform prior on the heights
            of the knots
        interp_type : `str`
            interpolation type. "linear," "cubic" and "akima" are
            the valid options.
        """
        self.data = data
        self.N_possible_knots = N_possible_knots
        self.xlow = xrange[0]
        self.xhigh = xrange[1]

        self.yhigh = height_prior_range[0]
        self.ylow = height_prior_range[1]

        self.interp_type = interp_type

        self.available_knots = np.linspace(self.xlow, self.xhigh, num=self.N_possible_knots)

        # keeps track of configuration, i.e. what points are turned on
        # or turned off.
        self.configuration = np.ones(self.N_possible_knots, dtype=bool)
        self.current_heights = np.ones(self.N_possible_knots)


    def evaluate_interp_model(self, xvals_to_evaluate, heights, config):
        """
        based on the supplied configuration and heights of the knots
        evaluate the model at `xvals_to_evaluate`.
        """
        if np.sum(config) == 0:
            return np.zeros(xvals_to_evaluate.size)
        elif np.sum(config) == 1:
            return heights[config].squeeze() * np.ones(xvals_to_evaluate.size)
        elif self.interp_type == 'linear':
            myfunc = interp1d(self.available_knots[config], heights[config],
                              fill_value='extrapolate')
        elif self.interp_type == 'cubic':
            myfunc = CubicSpline(self.available_knots[config], heights[config],
                                 extrapolate=True)
        elif self.interp_type == 'akima':
            myfunc = Akima1DInterpolator(self.available_knots[config], heights[config])
            return myfunc(xvals_to_evaluate, extrapolate=True)
        else:
            raise ValueError('available spline types are "linear," "cubic" and "akima"')
        return myfunc(xvals_to_evaluate)

    @abstractmethod
    def ln_likelihood(self, config, heights):
        """
        You will need to implement this yourself. It will take the model, and put it into whatever space
        it needs to be in to calculate your likelihood, and then calculate the likelihood.
        """
        pass


    def propose_birth_move(self):
        """
        propose to add "turn on" one of the available knots
        from the list of knots that are currently turned off.
        """
        if np.sum(self.configuration) == self.N_possible_knots:
            return (-np.inf, -np.inf, self.configuration, self.current_heights)
        else:
            idx_to_add = np.random.choice(np.where(~self.configuration)[0])
            new_heights = deepcopy(self.current_heights)
            new_config = deepcopy(self.configuration)
            new_config[idx_to_add] = True
            # draw from uniform prior
            new_heights[idx_to_add] = np.random.rand() * (self.yhigh - self.ylow) + self.ylow
            new_ll = self.ln_likelihood(new_config, new_heights)
            return new_ll, 0, new_config, new_heights

    def propose_death_move(self):
        """
        propose to "turn off" one of the current knots that are turned on.
        """
        if np.sum(self.configuration) == 0:
            return (-np.inf, -np.inf, self.configuration, self.current_heights)
        else:
            # pick one to turn off
            idx_to_add = np.random.choice(np.where(self.configuration)[0])
            new_heights = deepcopy(self.current_heights)
            new_config = deepcopy(self.configuration)

            # turn it off
            new_config[idx_to_add] = False
            # draw from uniform prior

            new_ll = self.ln_likelihood(new_config, self.current_heights)
            return new_ll, 0, new_config, new_heights

    def propose_change_amplitude_gaussian(self):
        """
        Pick one of the knots that are turned
        on and propose to change
        its height by some small amount.
        """
        # random point to turn on
        if np.sum(self.configuration) == 0:
            return -np.inf, -np.inf, self.configuration, self.current_heights
        idx_to_change = np.random.choice(np.where(self.configuration)[0])

        # draw a "scale" factor between 1/10 and 1/3 of prior range
        scalefac = (self.yhigh - self.ylow) * np.random.rand() * (1/3 - 1/10) + 1/10

        # propose to jump an amount given by zero-mean Gaussian with standard
        # deviation given by scalefac above
        new_heights = deepcopy(self.current_heights)
        new_heights[idx_to_change] = self.current_heights[idx_to_change] + np.random.randn() * scalefac

        new_ll = self.ln_likelihood(self.configuration, new_heights)
        return new_ll, 0, self.configuration, new_heights

    def propose_change_amplitude_prior_draw(self):
        """
        choose one of the knots that are turned on and propose
        a new height that is drawn from the prior.
        """
        if np.sum(self.configuration) == 0:
            return -np.inf, -np.inf, self.configuration, self.current_heights

        # random point to turn on
        idx_to_change = np.random.choice(np.where(self.configuration)[0])

        # propose draw from prior
        new_heights = deepcopy(self.current_heights)
        new_heights[idx_to_change] = (self.yhigh - self.ylow) * np.random.rand() + self.ylow

        new_ll = self.ln_likelihood(self.configuration, new_heights)
        return new_ll, 0, self.configuration, new_heights

    def sample(self, Niterations, proposal_weights=(10, 10, 1, 1), prior_test=False,
               start_config=None, start_heights=None):
        """
        Run RJMCMC sampler

        Parameters:
        -----------
        Niterations : `int`
            Number of MCMC samples
        proposal_weights : `list` or `np.ndarray`, optional
            list of weights for proposals. In order they are currently:
                [birth, death, prior draw, gaussian]
        prior_test : `bool`, optional, default=False
            If True, it sets likelihood to 0 always and
            samples only from the prior. Vital to check that
            when adding new proposals you still get back the prior.
        start_config : `np.ndarray`, optional, default=None
            Array of boolean values that turn on or off certain knots.
        start_heights: `np.ndarray`, optional, default=None
            Array of starting heights for the knots.

        Returns:
        --------
        results : `SamplerResults`
            sampler results object that contains configurations,
            heights, acceptances, likelihoods, and proposal types
            for each MCMC step.

            configurations = (Nsamples x Nspline points), for example.
        """

        if start_config is not None:
            if np.size(start_config) == self.N_possible_knots:
                self.configuration = start_config
            else:
                print('Start config you entered is not compatible...starting with all knots on')
        if start_heights is not None:
            if np.size(start_heights) == self.N_possible_knots:
                self.current_heights = start_heights
            else:
                print('Start heights you entered is not compatible...starting with heights at zero')
        configurations = np.zeros((Niterations, self.N_possible_knots))
        heights = np.zeros((Niterations, self.N_possible_knots))

        acceptances = np.zeros(Niterations, dtype=bool)
        move_types = np.zeros(Niterations, dtype=int)

        lls = np.zeros(Niterations, dtype=float)

        current_ll = -np.inf

        # list of functions for proposals
        proposals = [self.propose_birth_move, self.propose_death_move,
                     self.propose_change_amplitude_prior_draw, self.propose_change_amplitude_gaussian]

        for ii in tqdm(range(Niterations)):
            # choose proposal
            myval = np.random.rand()
            # choose proposal function with weights that were specified
            proposal_idx = np.random.choice(np.arange(len(proposals)), p=np.array(proposal_weights) / np.sum(proposal_weights))
            # while np.sum(self.configuration) == 0 and proposal_idx == 1:
            #     proposal_idx = np.random.choice(np.arange(len(proposals)), p=np.array(proposal_weights) / np.sum(proposal_weights))
            # while np.sum(self.configuration) == np.size(self.configuration) and proposal_idx == 0:
            #     proposal_idx = np.random.choice(np.arange(len(proposals)), p=np.array(proposal_weights) / np.sum(proposal_weights))

            # get proposed points
            proposed_ll, proposed_logR, proposed_config, proposed_heights = proposals[proposal_idx]()
            if prior_test:
                proposed_ll = 0
                
            # need to handle ratio of probabilities of birth vs. death proposals
            # because prob(n -> n+1) would be different from prob(n+1 -> n)
            # if ratio with which we make proposals is different.
            # this is subtle...usually the ratio of *which* proposals you choose
            # doesn't matter
            if proposal_idx == 0:
                q = np.log(proposal_weights[1] / proposal_weights[0])
            elif proposal_idx ==1:
                q = np.log(proposal_weights[0] / proposal_weights[1])
            else:
                q = 0

            hastings_ratio = min(np.log(1), proposed_ll - current_ll + proposed_logR + q)
            compare_val = np.log(np.random.rand())

            if compare_val < hastings_ratio:
                self.configuration = proposed_config
                self.current_heights = proposed_heights
                current_ll = proposed_ll
                acc = True
            else:
                acc = False

            configurations[ii] = self.configuration
            heights[ii] = self.current_heights
            acceptances[ii] = acc
            move_types[ii] = proposal_idx
            lls[ii] = current_ll

        return SamplerResults(acceptances, configurations, heights, lls, move_types)


class SamplerResults(object):
    def __init__(self, acceptances, configurations, heights, lls, move_types):
        self.acceptances = acceptances
        self.configurations = configurations
        self.heights = heights
        self.lls = lls
        self.move_types = move_types


class SmoothCurveDataObj(object):
    """
    A data class that can be used with our spline model
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitSmoothCurveModel(BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    def ln_likelihood(self, config, heights):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        model = self.evaluate_interp_model(self.data.data_xvals, heights, config)
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))
