import numpy as np
import Two_step.utility as ut
import Two_step.model_fitting as mf
import Two_step.model_plotting as mp
from Two_step.logistic_regression import _logistic_regression_model

# -------------------------------------------------------------------------------------
# Configurable logistic regression Model.
# -------------------------------------------------------------------------------------

class config_log_reg(_logistic_regression_model):

    def __init__(self, predictors = ['choice','outcome'], lags = 3, norm = False,
                 orth = False, trial_mask = None, invert_mask = False):

        self.name = 'config_lr'
        self.base_predictors = predictors # predictor names ignoring lags.
        self.orth = orth 
        self.norm = norm

        if type(lags) == int:
            lags = {p:lags for p in predictors}

        self.predictors = [] # predictor names including lags.
        for predictor in self.base_predictors:
            if predictor in list(lags.keys()):
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '_lag_' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        if trial_mask:
            self.trial_select = {'trial_mask' : trial_mask,
                                 'invert_mask': invert_mask}
        else:
            self.trial_select = None

        _logistic_regression_model.__init__(self)

    def _get_session_predictors(self, session):
        '''Calculate and return values of predictor variables for all trials in session.
        '''
        # Evaluate base (non-lagged) predictors from session events.

        choices, outcomes = session.unpack_CO(dtype = bool)

        bp_values = {} 

        for p in self.base_predictors:

            if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
                bp_values[p] = -0.5 * (session.blocks['trial_rew_state'] - 1)
      
            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5
                
        # predictor orthogonalization.

        if self.orth: 
            for A, B in self.orth: # Remove component of predictor A that is parrallel to predictor B. 
                bp_values[A] = bp_values[A] - ut.projection(bp_values[B], bp_values[A])

        # predictor normalization.
        if self.norm:
            for p in self.base_predictors:
                bp_values[p] = bp_values[p] * 0.5 / np.mean(np.abs(bp_values[p]))

        # Generate lagged predictors from base predictors.

        predictors = np.zeros([session.n_trials, self.n_predictors])

        for i,p in enumerate(self.predictors):  
            if '_lag_' in p: # Get lag from predictor name.
                lag = int(p.split('_lag_')[1]) 
                bp_name = p.split('_lag_')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            predictors[lag:, i] = bp_values[bp_name][:-lag]

        return predictors


# -------------------------------------------------------------------------------------
# Logistic regression analysis.
# -------------------------------------------------------------------------------------

def logistic_regression(sessions, predictors = 'standard', lags = 3, fig_no = 1):
    ''' Run and plot logistic regression analysis on specified sessions using
    logistic regression model with specified predictors. 
    '''
    if type(predictors) is list:
        model = config_log_reg(predictors, lags)
    else:
        assert predictors in ['standard'], 'Invalid predictors argument.'
        model = config_log_reg()
    population_fit = mf.fit_population(sessions, model)
    mp.model_fit_plot(population_fit, fig_no)