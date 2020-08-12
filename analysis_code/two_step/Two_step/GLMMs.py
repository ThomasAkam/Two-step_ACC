from pymer4.models import Lmer
import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns

#---------------------------------------------------------------------------------------------------------
# Two_step_LR_model
#---------------------------------------------------------------------------------------------------------

class Two_step_LR_model():
    '''
    Class for generating trial by trial predictors for logistic regression analysis of 
    two-step data.

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags       - By default each predictor is only used at a lag of -1 (i.e. one trial 
                 predicting the next). The lags argument is used to specify the use of 
                 additional lags for specific predictors: 
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on
                 the previous 3 trials should be used as predictors, while the choices 
                 on the previous 2 trials should be used. Lags can also be specified 
                 which combine multiple trials in a single predictor, for example
                 {'outcome': ['1','2_3','4_6']}. If an interger or list is provided
                 as the lags argument rather than a dict, all predictors are given
                 this set of lags.
    '''

    def __init__(self, predictors = ['side', 'correct', 'choice', 'outcome', 'trans_CR', 'trCR_x_out'],
                 lags={}, selection_type='xtr', select_n=20, block_type='all'):

        self.name = 'config_lr'
        self.base_predictors = predictors # predictor names ignoring lags.
        self.selection_type = selection_type # Controls which trials are included in analysis.
        self.select_n = select_n             # Controls which trials are included in analysis.
        self.block_type = block_type

        if not type(lags) == dict: # Use same lags for all predictors.
            lags = {p:lags for p in predictors}

        self.predictors = [] # predictor names including lags.

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

    def get_session_dataframe(self, session):
        '''Make a Pandas dataframe containing the predictors, dependent variable (choices), and 
        grouping variables (subject and session).'''

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype = bool)
        transitions_CR = transitions_AB == session.blocks['trial_trans_state']  
        transition_CR_x_outcome = transitions_CR == outcomes 
        correct = -0.5*(session.blocks['trial_rew_state']-1)*(2*session.blocks['trial_trans_state']-1) 

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

        session_df = pd.DataFrame(data=predictors, columns=self.predictors)
        # Add dependent variable and grouping variables.
        session_df['Y'] = session.trial_data['choices'] # Dependent variables.
        session_df['subject'] = session.subject_ID      
        session_df['session'] = session.file_name

        # Select trials to include in analysis.
        trials_to_use = session.select_trials(self.selection_type, self.select_n, block_type=self.block_type)

        return session_df.loc[trials_to_use]

# ---------------------------------------------------------------------------------------------

def _lag(x, i): # Apply lag of i trials to array x.
    x_lag = np.zeros(x.shape, x.dtype)
    if i > 0:
        x_lag[i:] = x[:-i]
    elif i < 0:
        x_lag[:i] = x[-i:]
    return x_lag

#---------------------------------------------------------------------------------------------------------
# GLMM fitting.
#---------------------------------------------------------------------------------------------------------

def GLMM_fit(sessions, grouping='subject', lr_model=Two_step_LR_model(), 
             remove_rand_effects=None, plot=True):
    '''Fit a mixed effect logistic regression model to predict choices in sessions
    using the predictors specified by Two_step_LR_model and the specified grouping
    for random effects. The model is fit using the R package lme4 called via the 
    pymer4 Python module.'''
    assert grouping in ['session', 'subject', 'subject/session'], \
        "grouping must be 'session', 'subject' or 'subject/session'"
    # Setup dataframe
    sessions_df = make_sessions_dataframe(sessions, lr_model)
    # Setup GLMM
    predictors_string = '(' + ' + '.join(lr_model.predictors) + ')'
    if remove_rand_effects:
        formula = 'Y ~ {X} + ({X} - ({R}) || {G})'.format(
            X=predictors_string, R=' + '.join(remove_rand_effects), G=grouping)
    else:
        formula = 'Y ~ {X} + ({X} || {G})'.format(X=predictors_string, G=grouping)
    print(formula)
    model = Lmer(formula, family='binomial', data=sessions_df)
    model.fit(summarize=False)
    print(model.summary())
    if plot:
        plot_summary(model)
    return model

def stim_analysis_GLMM(sessions, grouping='subject', lr_model=Two_step_LR_model(),
                       remove_rand_effects=None, plot=True):
    '''Test whether opto stimulation has an effect on logistic regression model 
    parameter loadings.  The logistic regression models predictors are all 
    interacted with the stim/no-stim condition variable with random effects on 
    all predictors except any specified by the remove_rand_effects variable.'''
    assert grouping in ['session', 'subject', 'subject/session'], \
        "grouping must be 'session', 'subject' or 'subject/session'"
    # Setup dataframe
    sessions_df = make_sessions_dataframe(sessions, lr_model)
    # Setup GLMM
    predictors_string = '(' + ' + '.join(lr_model.predictors) + ')'
    if remove_rand_effects:
        formula = 'Y ~ {X} * stim_trials + ({X} * stim_trials - ({R}) || {G})'.format(
            X=predictors_string, R=' + '.join(remove_rand_effects), G=grouping)
    else:
        formula = 'Y ~ {X} * stim_trials + ({X} * stim_trials || {G})'.format(X=predictors_string, G=grouping)
    print(formula)
    model = Lmer(formula, family='binomial', data=sessions_df)
    model.fit(summarize=False)
    print(model.summary())
    if plot:
        plot_summary(model)
    return model

def stim_by_group_analysis_GLMM(sessions_opto, sessions_ctrl, grouping='subject', 
        lr_model=Two_step_LR_model(), remove_rand_effects=None, plot=True):
    '''Test whether opto stimulation has differential effects in an 'opto' group 
    expressing the construct and a 'ctrl' group not expressing the construct but
    recieving light stimulation. The logistic regression models predictors are all 
    interacted with the stim/no-stim condition and group variables with random effects  
    on all predictors except any specified by the remove_rand_effects variable.'''
    assert grouping in ['session', 'subject', 'subject/session'], \
        "grouping must be 'session', 'subject' or 'subject/session'"
    # Setup dataframe
    sessions_opto_df = make_sessions_dataframe(sessions_opto, lr_model)
    sessions_opto_df['group'] =  1
    sessions_ctrl_df = make_sessions_dataframe(sessions_ctrl, lr_model)
    sessions_ctrl_df['group'] = -1
    sessions_df = pd.concat([sessions_opto_df, sessions_ctrl_df])
    # Setup GLMM
    predictors_string = '(' + ' + '.join(lr_model.predictors) + ')'
    if remove_rand_effects:
        formula = 'Y ~ {X} * stim_trials * group + ({X} * stim_trials * group - ({R}) || {G})'.format(
            X=predictors_string, R=' + '.join(remove_rand_effects), G=grouping)
    else:
        formula = 'Y ~ {X} * stim_trials * group + ({X} * stim_trials * group || {G})'.format(X=predictors_string, G=grouping)
    print(formula)
    model = Lmer(formula, family='binomial', data=sessions_df)
    model.fit(summarize=False)
    print(model.summary())
    if plot:
        plot_summary(model)
    return model

def make_sessions_dataframe(sessions, lr_model):
    '''Make a pandas dataframe comprising the lr_model predictors, the dependent variable
    (choices), and the subject and session IDs to use for grouping random effects.'''
    session_data_frames = []
    for i, session in enumerate(sessions):
        session_df = lr_model.get_session_dataframe(session)
        if hasattr(session, 'stim_trials'):
            trials_to_use = session.select_trials(lr_model.selection_type, 
                lr_model.select_n, block_type=lr_model.block_type)
            session_df['stim_trials'] = session.stim_trials[trials_to_use].astype(float)
        session_data_frames.append(session_df)
    return pd.concat(session_data_frames)

#---------------------------------------------------------------------------------------------------------
# GLMM plotting
#---------------------------------------------------------------------------------------------------------

def plot_summary(model, figsize=(12, 6), error_bars='ci', ranef=True, 
                 intercept=True, ranef_alpha=.5, coef_fmt='o', line=False):

        m_ranef = model.fixef
        m_fixef = model.coefs

        if not intercept:
            m_ranef = m_ranef.drop('(Intercept)', axis=1)
            m_fixef = m_fixef.drop('(Intercept)', axis=0)

        if error_bars == 'ci':
            col_lb = m_fixef['Estimate'] - m_fixef['2.5_ci']
            col_ub = m_fixef['97.5_ci'] - m_fixef['Estimate']
        elif error_bars == 'se':
            col_lb, col_ub = m_fixef['SE'], m_fixef['SE']

        # For seaborn
        m = pd.melt(m_ranef)

        f, ax = plt.subplots(1, 1, figsize=figsize)

        if ranef:
            alpha_plot = ranef_alpha
        else:
            alpha_plot = 0

        sns.stripplot(x='variable', y='value', data=m, ax=ax,
                      size=6, alpha=alpha_plot, color='grey')

        ax.errorbar(x=range(m_fixef.shape[0]), y=m_fixef['Estimate'], yerr=[
                     col_lb, col_ub], fmt=coef_fmt, capsize=0, elinewidth=3, color='black', ms=6, zorder=9999999999)

        if line:
            ax.plot(range(m_fixef.shape[0]), m_fixef['Estimate'], color='k', linewidth=2)

        ax.hlines(y=0, xmin=-1,
                  xmax=model.coefs.shape[0], linestyles='--', color='grey')

        plt.xlim(-1,xmax=model.coefs.shape[0])
        plt.xticks(rotation=-45, ha='left')

        plt.ylabel('Log odds')

        return ax