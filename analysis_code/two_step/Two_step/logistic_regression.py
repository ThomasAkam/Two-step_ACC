import numpy as np

from sklearn.utils import resample
from functools import partial

from . import utility as ut
from . import model_fitting as mf 
from . import model_plotting as mp
from . import parallel_processing as pp
from .stim_analysis import _print_P_values

# -------------------------------------------------------------------------------------
# logistic_regression_model
# -------------------------------------------------------------------------------------

class _logistic_regression_model():
    '''
    Superclass for logistic regression models which provides generic likelihood and 
    likelihood gradient evaluation.  To implement a specific logistic regression model 
    this class is subclassed with a _get_session_predictors function which takes a
    session as its argument and returns the array of predictors.  

    The trial_select variable can be used to specify rules for including only a subset 
    of trials in the analysis.  Set this variable to False to use all trials. include
    '''

    def __init__(self):

        self.n_params = 1 + len(self.predictors)

        self.param_ranges = ('all_unc', self.n_params)
        self.param_names  = ['bias'] + self.predictors

        if not hasattr(self, 'trial_select'): # Selection 
            self.trial_select = False

        self.calculates_gradient = True
        self.type = 'log_reg'

    def _select_trials(self, session):
        if 'selection_type' in self.trial_select.keys():
            selected_trials = session.select_trials(self.trial_select['selection_type'],
                                                    self.trial_select['select_n'],
                                                    block_type=self.trial_select['block_type'])
        else:
            selected_trials=np.ones(session.n_trials,bool)
        if 'trial_mask' in self.trial_select.keys():
            trial_mask = getattr(session, self.trial_select['trial_mask'])
            if self.trial_select['invert_mask']:
                trial_mask = ~trial_mask
            selected_trials = selected_trials & trial_mask
        return selected_trials
        
    def session_likelihood(self, session, params_T, eval_grad = False):

        bias = params_T[0]
        weights = params_T[1:]

        choices = session.trial_data['choices']

        if not hasattr(session,'predictors'):
            predictors = self._get_session_predictors(session) # Get array of predictors
        else:
            predictors = session.predictors

        assert predictors.shape[0] == session.n_trials, 'predictor array does not match number of trials.'
        assert predictors.shape[1] == len(weights), 'predictor array does not match number of weights.'

        if self.trial_select: # Only use subset of trials.
            selected_trials = self._select_trials(session)
            choices = choices[selected_trials]
            predictors = predictors[selected_trials,:]

        # Evaluate session log likelihood.

        Q = np.dot(predictors,weights) + bias
        Q[Q < -ut.log_max_float] = -ut.log_max_float # Protect aganst overflow in exp.
        P = 1./(1. + np.exp(-Q))  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  # Probability of chosen action.

        session_log_likelihood = sum(ut.log_safe(Pc)) 

        # Evaluate session log likelihood gradient.

        if eval_grad:
            dLdQ  = - 1 + 2 * choices + Pc - 2 * choices * Pc
            dLdB = sum(dLdQ) # Likelihood gradient w.r.t. bias paramter.
            dLdW = sum(np.tile(dLdQ,(len(weights),1)).T * predictors, 0) # Likelihood gradient w.r.t weights.
            session_log_likelihood_gradient = np.append(dLdB,dLdW)
            return (session_log_likelihood, session_log_likelihood_gradient)
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Configurable logistic regression model.
# -------------------------------------------------------------------------------------

class config_log_reg(_logistic_regression_model):

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags        - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used.  
                Lags can also be specified which combine multiple trials in a single predictor, for example
                {'outcome': ['1','2-3','4-6']} If an interger or list is provided as the lags argument
                rather than a dict, all predictors are given this set of lags.

    trial_mask  - Subselect trials based on session attribute with specified name.  E.g. if 
                trial_mask is set to 'stim_choices', the variable session.stim_choices 
                (which must be a boolean array of length n_trials) will be used to select
                trials for each session fit. The additional invert_mask variable can be 
                used to invert the mask.  Used for subselecting trials with e.g. optogenetic
                stimulation.
    '''


    def __init__(self, predictors='standard', lags={},  trial_mask=None, invert_mask=False,
                 selection_type='xtr', select_n=20, block_type='all'):

        self.name = 'config_lr'

        if type(predictors) == list:
            self.base_predictors = predictors # predictor names ignoring lags.
        elif predictors == 'standard':
            self.base_predictors = ['side', 'correct','choice','outcome', 'trans_CR', 'trCR_x_out']
        elif predictors == 'lagged':
            self.base_predictors = ['side', 'choice','outcome', 'trans_CR', 'trCR_x_out']
            lags={p: ['1','2','3_4','5_8','8_12'] for p in 
                  ['choice', 'outcome', 'trans_CR', 'trCR_x_out']}
        elif predictors == '+same_mo':
            self.base_predictors = ['side', 'correct','choice','outcome', 'trans_CR', 'trCR_x_out', 'same_mo']

        if not type(lags) == dict: # Use same lags for all predictors.
            lags = {p:lags for p in predictors}

        self.predictors = []

        for predictor in self.base_predictors:
            if predictor in list(lags.keys()):
                if type(lags[predictor]) == list: # Use a specified set of lags.
                    for l in lags[predictor]:
                        self.predictors.append(predictor + '_lag_' + l)
                else:
                    for i in range(lags[predictor]): # Use lags from 1 to 
                        self.predictors.append(predictor + '_lag_' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        self.trial_select = {'selection_type': selection_type,
                             'select_n'      : select_n,
                             'block_type'    : block_type}        

        if trial_mask:
            self.trial_select['trial_mask']  = trial_mask
            self.trial_select['invert_mask'] = invert_mask

        _logistic_regression_model.__init__(self)

    def _get_session_predictors(self, session):
        'Calculate and return values of predictor variables for all trials in session.'

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype = bool)
        transitions_CR = transitions_AB == session.blocks['trial_trans_state']  
        transition_CR_x_outcome = transitions_CR == outcomes 
        correct = -0.5*(session.blocks['trial_rew_state']-1)* \
                       (2*session.blocks['trial_trans_state']-1) 

        bp_values = {} 

        for p in self.base_predictors:

            if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
                bp_values[p] =  correct
      
            elif p == 'side': # 0.5, -0.5 for left, right side reached at second step. 
                bp_values[p] = second_steps - 0.5

            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'trans_CR': # 0.5, -0.5 for common, rare transitions.  
                bp_values[p] = ((transitions_CR) == choices)  - 0.5               

            elif p == 'trCR_x_out': # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
                bp_values[p] = (transition_CR_x_outcome  == choices) - 0.5   

            elif p == 'same_mo': # Same motor action to repeat choice.
                second_steps_l1 = _lag(second_steps,1)
                bp_values[p] = ((second_steps == second_steps_l1) == choices) - 0.5
                bp_values[p][0] = 0

        # Generate lagged predictors from base predictors.

        predictors = np.zeros([session.n_trials, self.n_predictors])

        for i,p in enumerate(self.predictors):  
            if '_lag_' in p: # Get lag from predictor name.
                lag = p.split('_lag_')[1].split('_')
                bp_name = p.split('_lag_')[0]
            else:        # Use default lag.
                lag = '1'
                bp_name = p
            if len(lag) == 1: # Lag is a specified number of trials.
                l = int(lag[0])
                predictors[l:, i] = bp_values[bp_name][:-l]
            else: # Lag is a specified range of trials.
                for l in range(int(lag[0]), int(lag[1])+1):
                    predictors[l:, i] += bp_values[bp_name][:-l]

        return predictors

def _lag(x, i): # Apply lag of i trials to array x.
    x_lag = np.zeros(x.shape, x.dtype)
    if i > 0:
        x_lag[i:] = x[:-i]
    elif i < 0:
        x_lag[:i] = x[-i:]
    return x_lag

# -------------------------------------------------------------------------------------
# Logistic regression analyses.
# -------------------------------------------------------------------------------------

def logistic_regression(sessions, predictors='standard', fig_no=1, title=None):
    ''' Run and plot logistic regression analysis on specified sessions using
    logistic regression model with specified predictors. 
    '''
    model = config_log_reg(predictors)
    population_fit = mf.fit_population(sessions, model)
    if predictors == 'lagged':
        mp.lagged_fit_plot(population_fit, fig_no=fig_no, title=title)
    else:
        mp.model_fit_plot(population_fit, fig_no, title=title)

# -------------------------------------------------------------------------------------
# Bootstrap significance testing.
# -------------------------------------------------------------------------------------

def predictor_significance_test(sessions, agent, n_perms=1000, file_name=None):
    '''Test whether logistic regression predictor loadings are significantly
    different from zero by bootstrap resampling subjects.'''

    mf._precalculate_fits(sessions, agent) # Store first round fits on sessions.
    
    permute_and_fit = partial(_permute_and_fit, agent=agent)

    bootstrap_fits = []
    for i, bs_fit in enumerate(pp.imap(permute_and_fit, [sessions]*n_perms, ordered=False)):
        bootstrap_fits.append(bs_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9:
            p_value_dict = _eval_P_values(bootstrap_fits, agent)
            _print_P_values(p_value_dict, n_perms=i+1, file_name=file_name)

    for session in sessions: del(session.fit) # Clear precalcuated fits.

def _eval_P_values(bootstrap_fits, agent):
    pop_means = np.array([bf['pop_dists']['means'] for bf in bootstrap_fits])
    P_values = np.min((np.mean(pop_means > 0, 0), np.mean(pop_means < 0, 0)),0)*2.
    return dict(zip(agent.param_names, P_values))

def _permute_and_fit(sessions, agent):
    subjects = set([s.subject_ID for s in sessions])
    resampled_sessions = []
    for subject in resample(list(subjects)):
        resampled_sessions += [s for s in sessions if s.subject_ID == subject]
    rs_fits = [session.fit for session in resampled_sessions]   
    return mf.fit_population(resampled_sessions, agent, verbose=False, init={'session_fits':rs_fits})