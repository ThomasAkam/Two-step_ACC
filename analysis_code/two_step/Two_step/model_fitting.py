import math
import time

import numpy as np
import scipy.optimize as op

from scipy.stats import multivariate_normal
from sklearn.utils import resample
from functools import partial

from . import parallel_processing as pp

#------------------------------------------------------------------------------------
# Fitting functions.
#------------------------------------------------------------------------------------

def repeated_fit_population(sessions, agent, n_draws=1000, n_repeats=10, tol=0.0001, max_iter=200,
                            verbose=False):
    '''Run fit population repeatedly with randomised intial population level
    parameters and return fit with best integrated likelihood.'''
    fit_func = partial(fit_population, sessions, init='rand', eval_BIC={'n_draws':n_draws},
                       tol=tol, max_iter=max_iter, verbose=verbose)
    repeated_fits = pp.map(fit_func, [agent]*n_repeats)
    best_fit = repeated_fits[np.argmax([fit['iBIC']['int_lik'] for fit in repeated_fits])]
    best_fit['repeated_fits'] = repeated_fits # Store all fits on best fit.
    return best_fit


def fit_population(sessions, agent, init=None, eval_BIC=True, tol=0.0001,
                   max_iter=200, verbose=True):
    ''' Fits population level parameters using the Expectation Maximisation method from Huys et al.'''   

    def M_step(session_fits): 
        # Adjust population mean and variance to maximise the expectation of the log likelihood.
        ses_params_U  = np.array([fit['params_U']  for fit in session_fits])  
        ses_diag_hess = np.array([fit['diag_hess'] for fit in session_fits])
        pop_means = np.mean(ses_params_U, 0)
        pop_vars  = np.mean(ses_params_U**2.-1./ses_diag_hess,0)-pop_means**2.
        return {'means': pop_means,'vars' : pop_vars, 'SDs'  : np.sqrt(pop_vars)}

    start_time = time.time()

    _clear_fitting_variables(sessions) # Clear any fitting variables left on sessions from prior aborted fit.

    if agent.type == 'log_reg': # Precalculate session predictors for logistic regression agents.
        for session in sessions: 
            session.predictors = agent._get_session_predictors(session)

    fit_evo = {'dists':[], 'prob':[], 'iLik':[], 'lik_rel_cng':[-1]} # Stores evolution of fit over EM iterations.

    # Initialise population level distributions for first round of MAP.

    if init is None:
        pop_dists = {'means': np.zeros(agent.n_params),
                     'vars' : np.ones(agent.n_params)*6.25}
    elif init == 'rand':
        pop_dists = {'means': np.random.randn(agent.n_params),
                     'vars' : np.ones(agent.n_params)*6.25}
    elif 'session_fits' in init.keys(): # Precalculated session fits passed in (used to speed up permutation tests).
        pop_dists = M_step(init['session_fits'])
        for session, fit in zip(sessions, init['session_fits']):
            session.init_params_U = fit['params_U'] 
    elif 'pop_dists' in init.keys():
        pop_dists = init['pop_dists']

    # Create standard normal distribution samples for integrated likelihood evaluation.
    n_sns = [25,100] # Number of samples per session to estimate integrated likelihood in [early,late] fitting.
    sns = np.random.normal(size=[len(sessions), n_sns[0], agent.n_params])

    # EM algorithm iterations.

    for k in range(max_iter): # EM  algorithm iterations.

        if verbose and k > 0: print('EM round: {} '.format(k), end = '')

        # E - Step: Evaluate the new posterior distribution for each sessions parameters.

        session_fits = pp.map(partial(fit_session, agent=agent, pop_dists=pop_dists), sessions)
            
        # M - step: Adjust population distribution mean and variance to maximise the expectation of the log likelihood.

        pop_dists = M_step(session_fits) 

        # Store population parameter evolution.

        fit_evo['dists'].append(pop_dists)
        fit_evo['prob'].append(sum([fit['prob'] for fit in session_fits]))
        fit_evo['iLik'].append(evaluate_iBIC(sessions, agent, pop_dists, sns=sns))

        if k > 0: # Test for convergence
            fit_evo['lik_rel_cng'].append((fit_evo['iLik'][-2]-fit_evo['iLik'][-1])/fit_evo['iLik'][-1])
            if verbose: print('Int. lik. relative change: {:.4}, using {} samples per session.'
                              .format(fit_evo['lik_rel_cng'][-1], sns.shape[1]))
            if k > 1:
                pred_next_rel_cng = 2*fit_evo['lik_rel_cng'][-1] - fit_evo['lik_rel_cng'][-2]
                if pred_next_rel_cng < 1.2*tol and fit_evo['lik_rel_cng'][-1] < 4*tol and sns.shape[1] == n_sns[0]:
                    # Increace number of samples used for likelihood estimation.
                    sns = np.random.normal(size=[len(sessions), n_sns[1], agent.n_params])
                    fit_evo['iLik'][-1] = evaluate_iBIC(sessions, agent, pop_dists, sns=sns)
                if fit_evo['lik_rel_cng'][-1] < tol: 
                    if verbose: print('EM fitting Converged.')
                    break

        for session, fit in zip(sessions, session_fits):
            session.init_params_U = fit['params_U'] # Start fit_session gradient decent from previous rounds fit.

    if verbose: print(('Elapsed time: ' + str(time.time() - start_time)))
 
    _clear_fitting_variables(sessions) # Remove variables added to sessions during fit.

    population_fit = {'session_fits': session_fits,
                       'pop_dists'   : pop_dists,
                       'fit_evo'     : fit_evo,
                       'agent_name'  : agent.name,
                       'param_names' : agent.param_names,
                       'param_ranges': agent.param_ranges}
                       
    if eval_BIC:  # eval_BIC can be boolean or argument dict for evaluate_iBIC.
        if type(eval_BIC) is bool: eval_BIC = {} 
        population_fit['iBIC'] = evaluate_iBIC(sessions, agent, population_fit, **eval_BIC)
     
    return population_fit
    

def fit_session(session, agent, pop_dists=None, repeats=5, max_attempts=10, verbose=False):
    '''Find maximum a posteriori agent parameter values for given session and means and 
    variances of population level prior distributions.'''

    if pop_dists is None: # No prior provided, use (almost completely) uninformative prior.
         pop_dists = {'means': np.zeros(agent.n_params), 
                      'vars' :np.ones(agent.n_params)*100.}

    if hasattr(session, 'init_params_U'): repeats = 1 # Use provided initial value for gradient descent.

    fit_func = lambda params_U: session_log_posterior(params_U, session, agent,
                                                      pop_dists, sign = - 1.)

    hess_func = lambda params_U: session_log_posterior(params_U, session, agent,
                                        pop_dists, sign = 1., eval_grad = False)

    fits = []
    for r in range(repeats): # Number of fits to perform with different starting conditions.
        valid_fit_found = False
        for a in range(max_attempts): # Number of attempts to find fit with valid Hessian.
            if hasattr(session, 'init_params_U') and a == 0:
                init_params_U = session.init_params_U # Use provided initial values.
            else:
                init_params_U = np.random.randn(agent.n_params)*3.
            fit = op.minimize(fit_func, init_params_U, jac=agent.calculates_gradient,
                              options={'disp': verbose, 'gtol': 1e-7})
            fit['hessdiag'] = _Hess_diag(hess_func, fit['x'])
            if max(fit['hessdiag']) < 0: 
                valid_fit_found = True
                break
        if not valid_fit_found:
            if verbose: print('Warning: valid fit not found in fit_session')
            fit['hessdiag'][fit['hessdiag']>=0] = -1e-6
        fits.append(fit)

    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.

    return {'params_U'  : fit['x'],
            'params_T'  : _trans_UT(fit['x'], agent.param_ranges),
            'prob'      : - fit['fun'], 
            'diag_hess' : fit['hessdiag'],
            'n_trials'  : session.n_trials,
            'sID'       : session.subject_ID} 

#------------------------------------------------------------------------------------
# Helper functions.
#------------------------------------------------------------------------------------

def session_log_posterior(params_U, session, agent, pop_dists, eval_grad=True, sign=1.):
    '''Evaluate the log posterior probability of behaviour in a single session 
    for a given set of parameter values and population level mean and variances.
    '''
    pop_means, pop_vars = (pop_dists['means'], pop_dists['vars']) 

    log_prior_prob = - (len(params_U) / 2.) * np.log(2 * np.pi) - np.sum(np.log(pop_vars)) \
                      / 2. - sum((params_U - pop_means) ** 2. / (2 * pop_vars))
    
    params_T = _trans_UT(params_U, agent.param_ranges)
    if agent.calculates_gradient and eval_grad:
        log_lik, log_lik_grad_T = agent.session_likelihood(session, params_T, eval_grad = True)
        log_lik_grad_U = _trans_grad_TU(params_T, log_lik_grad_T, agent.param_ranges)
        log_prior_prob_grad = ((pop_means - params_U) / pop_vars)
        log_posterior_prob = log_lik + log_prior_prob
        log_posterior_grad = log_lik_grad_U + log_prior_prob_grad
        return (sign * log_posterior_prob, sign * log_posterior_grad)
    else:
        log_lik = agent.session_likelihood(session, params_T)
        log_posterior_prob = log_lik + log_prior_prob
        return (sign * log_posterior_prob)


def grad_check(session, agent, params_T=None):
    '''Check analytical likelihood gradient returned by agent.'''
    if params_T is None:
        params_U = np.random.normal(np.zeros(agent.n_params),np.ones(agent.n_params)*2)
        params_T = _trans_UT(params_U, agent.param_ranges)
    fit_func  = lambda params_T: agent.session_likelihood(session, params_T, eval_grad = True)
    lik_func  = lambda params_T: fit_func(params_T)[0]
    grad_func = lambda params_T: fit_func(params_T)[1]
    l2error = op.check_grad(lik_func, grad_func, params_T)
    print(('Error between finite difference and analytic derivatives = ' + str(l2error)))


def _Hess_diag(fun, x, dx=1e-4):
    '''Evaluate the diagonal elements of the hessian matrix using the 3 point
    central difference formula with spacing of dx between points.'''
    n = len(x)
    hessdiag = np.zeros(n)
    for i in range(n):
        dx_i    = np.zeros(n)
        dx_i[i] = dx
        hessdiag[i] = (fun(x + dx_i) + fun(x - dx_i) - 2. * fun(x)) / (dx ** 2.)
    return hessdiag


def _sample_params_T(population_fit):
    '''Draw a sample of paramter values in transformed space from population 
    distribution.'''
    params_U = np.random.normal(population_fit['pop_dists']['means'], 
                                population_fit['pop_dists']['SDs'])
    params_T = _trans_UT(params_U, population_fit['param_ranges'])
    return params_T


def _precalculate_fits(sessions, agent):
    ''' Fit agent to each session and store fits on sessions.  Used in permutation
    testing, the precalculated fits are used for first round of EM when fitting
    populations.  This save recalculating these fits for each permuted population.'''
    pop_dists = {'means': np.zeros(agent.n_params), 
                 'vars' :np.ones(agent.n_params)*6.25}
    fit_func = partial(fit_session, agent=agent, pop_dists=pop_dists, repeats=10)
    print('Precalculating session fits.', end = '')
    for fit, session in zip(pp.imap(fit_func, sessions), sessions):
        session.fit = fit
        print('.', end = '')
    print('\n')


def _clear_fitting_variables(sessions):
    '''Delete temporary variables added to sessions during population fit.'''
    for session in sessions:
        if hasattr(session, 'init_params_U'):
            del session.init_params_U
        if hasattr(session, 'predictors'):
            del(session.predictors)

#------------------------------------------------------------------------------------
# iBIC evaluation
#------------------------------------------------------------------------------------

def evaluate_iBIC(sessions, agent, pop_dists, n_draws=1000, n_boot=5000, 
                  sns=None, return_likelihood=False):
    '''Return the integrated BIC score for given agent, sessions & population fit.
    iBIC is aproximated by sampling from the population distribution.  95%  confidence
    interval on the integrated likelihood are evaluated by bias corrected bootstrap 
    resampling.  Argument sns allows a set of standard normal samples (u = 0. sd = 1) 
    to be passed in and used to generate the samples from the population distribution.'''

    if not 'means' in pop_dists.keys(): # population_fit rather than pop_dists passed in.
        pop_dists = pop_dists['pop_dists']

    return_likelihood = return_likelihood or (sns is not None) # Return likelihood rather than full iBIC data.

    if sns is None: sns = np.random.normal(size=[len(sessions), n_draws, agent.n_params])

    param_samples = sns*pop_dists['SDs']+pop_dists['means']

    _sample_ses_log_liks_ = partial(_sample_ses_log_liks, agent=agent)

    ses_log_lik_samples = np.array(pp.starmap(_sample_ses_log_liks_, 
                                              zip(sessions, param_samples))).T

    m = np.mean(ses_log_lik_samples)
    integrated_likelihood = sum(np.log(np.mean(np.exp(ses_log_lik_samples-m),0))+m)

    if return_likelihood: return integrated_likelihood

    means_hessian = _pop_means_hessian(pop_dists, param_samples, ses_log_lik_samples, integrated_likelihood)

    if n_boot: # Estimate 95% confidence interval on likelihood by bootstrap resampling.

        bs_int_liks = pp.map(_resampled_int_lik, [ses_log_lik_samples]*n_boot)
        bs_int_liks = np.sort(bs_int_liks) 
        bs_bias = integrated_likelihood-np.median(bs_int_liks)
        lik_95_conf = np.array((bs_int_liks[round(n_boot*0.025)]+bs_bias,
                                bs_int_liks[round(n_boot*0.975)]+bs_bias))
    else:
        lik_95_conf=None

    # Convert integrated likelihood to BIC score.

    if agent.type == 'RL_stim': # Agent has seperate parameters for stim and non-stim trials
        n_trials  = sum([s.n_trials for s in sessions])             # Total number of trials.
        n_stim_trials = sum([sum(s.stim_trials) for s in sessions]) # Number of stim trials
        n_nons_trials = n_trials - n_stim_trials                    # Number of non-stim trials.
        n_params =  len(agent.param_names)
        n_stim_params = n_nons_params = sum([pn[-2:] == '_s' for pn in agent.param_names])
        n_both_params = n_params - n_stim_params - n_nons_params # parameters which apply on all trials.
        iBIC = (-2*integrated_likelihood + 2*(n_both_params*np.log(n_trials) +
                n_stim_params*np.log(n_stim_trials) + n_nons_params*np.log(n_nons_trials)))
    else:
        if hasattr(agent,'_select_trials') and bool(agent.trial_select):
            n_trials = sum([sum(agent._select_trials(s)) for s in sessions])
        else:
            n_trials = sum([s.n_trials for s in sessions])
        iBIC = -2*integrated_likelihood + 2*agent.n_params*np.log(n_trials)

    return {'score'        : iBIC,
            'int_lik'      : integrated_likelihood,
            'means_hessian': means_hessian,
            'choice_prob'  : np.exp(integrated_likelihood/n_trials),
            'lik_95_conf'  : lik_95_conf}

def _sample_ses_log_liks(session, sample_params, agent):
    return np.array([agent.session_likelihood(session, 
                     _trans_UT(sp, agent.param_ranges)) for sp in sample_params])

def _resampled_int_lik(ses_log_lik_samples):
    m = np.mean(ses_log_lik_samples)
    return sum(np.log(np.mean(np.exp(resample(ses_log_lik_samples)-m),0))+m)

def _pop_means_hessian(pop_dists, param_samples, ses_log_lik_samples, integrated_likelihood, rel_dx=1e-3):
    '''Evaluate the hessian of the data log likelihood with respect to the means of the 
    population level distributions using importance sampling.'''
    n_params = len(pop_dists['means'])
    means_hessian = np.zeros(n_params)
    for i in range(n_params):
        delta_u = np.zeros(n_params)
        dx = rel_dx*pop_dists['SDs'][i]
        delta_u[i] = dx
        shifted_int_lik_1 = _shift_dist_likelihood(delta_u, pop_dists, param_samples, ses_log_lik_samples)
        delta_u[i] = -dx
        shifted_int_lik_2 = _shift_dist_likelihood(delta_u, pop_dists, param_samples, ses_log_lik_samples)
        means_hessian[i] = (1./dx**2)*(shifted_int_lik_1-2*integrated_likelihood+shifted_int_lik_2)
    return means_hessian

def _shift_dist_likelihood(delta_u, pop_dists, param_samples, ses_log_lik_samples):
    '''Evaluate data likelihood for population level distribution with means shifted by delta_u
    using importance sampling - i.e. by reweighting the set of likelihood samples from the
    non-shifted distribution.'''
    w = (multivariate_normal.pdf(param_samples,mean=pop_dists['means']+delta_u,cov=pop_dists['vars']) /
         multivariate_normal.pdf(param_samples,mean=pop_dists['means']        ,cov=pop_dists['vars']))
    w = w / w.sum(1, keepdims=True)
    m = np.mean(ses_log_lik_samples)
    return sum(np.log(np.sum(np.exp(ses_log_lik_samples-m)*w.T,0))+m)

# -------------------------------------------------------------------------------------
# Parameter transformation between unconstrained and transformed space.
# -------------------------------------------------------------------------------------

def _trans_UT(values_U, param_ranges):
    '''Transform parameters from unconstrained to transformed space.'''
    if param_ranges[0] == 'all_unc': return values_U
    values_T = []
    for u, rng in zip(values_U, param_ranges):
        if rng   == 'unit':
            if u < -100.: u = -100. # Protect against overflow in exponential.
            values_T.append(1./(1.+math.exp(-u)))
        elif rng == 'pos':
            values_T.append(u + 3. if u > -2. else math.exp(u+2.))
        elif rng == 'unc':
            values_T.append(u)
    return np.array(values_T)


def _trans_TU(values_T, param_ranges):
    '''Transform parameters from transformed to unconstrained space.'''
    if param_ranges[0] == 'all_unc':
        return values_T
    values_U = []
    for t, rng in zip(values_T, param_ranges):
        if rng   == 'unit':
            if t < 1e-100:
                t = 1e-100
            elif t > 1-1e-15:
                t = 1-1e-15
            values_U.append(-math.log((1./t)-1.))
        elif rng == 'pos':
            values_U.append(t - 3. if t > 1. else math.log(t)-2.)
        elif rng == 'unc':
            values_U.append(t)
    return np.array(values_U)


def _trans_grad_TU(values_T, gradients_T, param_ranges):
    '''Transform gradient wrt paramters from transformed to unconstrained space.'''
    if param_ranges[0] == 'all_unc': return gradients_T
    gradients_U = []
    for t, dLdt, rng in zip(values_T, gradients_T, param_ranges):
        if rng   == 'unit':
            gradients_U.append(t * (1.-t) * dLdt)
        elif rng == 'pos':
            gradients_U.append(dLdt if t > 1. else t * dLdt)
        elif rng == 'unc':
            gradients_U.append(dLdt)
    return np.array(gradients_U)