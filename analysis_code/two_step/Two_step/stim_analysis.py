import pylab as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, linregress
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from random import shuffle
from seaborn import regplot
from statsmodels.api import OLS    

from . import plotting as pl
from . import model_plotting as mp
from . import model_fitting as mf
from . import logistic_regression as lr 
from . import RL_agents as rl
from . import utility as ut
from . import parallel_processing as pp

def group_info(sessions):
    subject_IDs = sorted(list(set([s.subject_ID for s in sessions])))
    print('n subjects: {}'.format(len(subject_IDs)))

    def print_data(sessions, name):
        n_sessions = len(sessions)
        n_stim_trials = np.sum([np.sum( s.stim_trials) for s in sessions])
        n_nons_trials = np.sum([np.sum(~s.stim_trials) for s in sessions])
        print('{}: {} sessions, {} stim trials, {} non-stim trials.'
              .format(name, n_sessions, n_stim_trials, n_nons_trials))

    for sID in subject_IDs:
        print_data([s for s in sessions if s.subject_ID == sID], sID)
    print_data(sessions, 'Total')
        
def stay_prob_analysis(sessions, trial_select='xtr', fig_no=1):
    '''Stay probability analysis analysing seperately trials where stimulation was
    and was not delivered between first and second choice (i.e. to affect stay).'''

    trial_mask_stim = [np.concatenate([ s.stim_trials[1:],[False]]) for s in sessions]
    trial_mask_nons = [np.concatenate([~s.stim_trials[1:],[True ]]) for s in sessions]

    per_sub_stay_probs_stim = pl.stay_probability_analysis(sessions, ebars = 'SEM',
                     selection = trial_select, fig_no = 0, trial_mask = trial_mask_stim)
    

    per_sub_stay_probs_nons = pl.stay_probability_analysis(sessions, ebars = 'SEM',
                     selection = trial_select, fig_no = 0, trial_mask = trial_mask_nons)
    
    stay_probs_stim = np.nanmean(per_sub_stay_probs_stim, 0)
    stay_probs_nons = np.nanmean(per_sub_stay_probs_nons, 0)
    SEM_stim = ut.nansem(per_sub_stay_probs_stim, 0)
    SEM_nons = ut.nansem(per_sub_stay_probs_nons, 0)

    plt.figure(fig_no, figsize=[2.5,2.3]).clf()
    plt.bar(np.arange(4), stay_probs_nons, 0.35, color = 'b',
            yerr = SEM_nons, error_kw = {'ecolor': 'y', 'capsize': 3, 'elinewidth': 3})
    plt.bar(np.arange(4) + 0.35, stay_probs_stim, 0.35, color = 'r',
            yerr = SEM_stim, error_kw = {'ecolor': 'y', 'capsize': 3, 'elinewidth': 3})
    plt.xlim(-0.35,4)
    plt.ylim(0.5,plt.ylim()[1])
    plt.xticks(np.arange(4) + 0.35,['Rew com.', 'Rew rare', 'Non com.', 'Non rare'])
    plt.ylabel('Stay probability')
    p_values = [ttest_rel(per_sub_stay_probs_stim[:,i], per_sub_stay_probs_nons[:,i]).pvalue
                for i in range(4)]
    print('Paired t-test P values:')
    for i, t in enumerate(['Rew com.:', 'Rew rare:', 'Non com.:', 'Non rare:']):
        print(t + ' {:.3f}'.format(p_values[i]))

#------------------------------------------------------------------------------------
# Logistic regression analyses.
#------------------------------------------------------------------------------------

def logistic_regression_comparison(sessions, predictors='standard', fig_no=1,
                                   title=None, agent=None, lagged_plot=False):
    '''Plot regression model fits to stim and non-stim trials.'''
    if not agent: agent = lr.config_log_reg(predictors)
    agent.trial_select['trial_mask'] = 'stim_trials'
    agent.trial_select['invert_mask'] = False
    fit_stim = mf.fit_population(sessions, agent, eval_BIC=True)
    agent.trial_select['invert_mask'] = True
    fit_nons = mf.fit_population(sessions, agent, eval_BIC=True)
    if lagged_plot:  # Used plot for lagged regression model.
        lagged_fit_comp_plot(fit_nons, fit_stim, fig_no, ebars='pm95')
    else:
        model_fit_comp_plot(fit_nons, fit_stim, fig_no, ebars='pm95')
    plt.ylim(-1,1.5)
    if title: plt.suptitle(title)

def logistic_regression_test(sessions, predictors='standard', n_perms=5000, n_true_fit=11,
                             post_stim=False, file_name=None, agent=None):
    '''Perform permutation testing to evaluate whether difference in logistic regression fits
    between stim and non stim trial data is statistically significant.'''

    if not agent: agent = lr.config_log_reg(predictors, trial_mask='stim_trials')

    print('Fitting original dataset.')
    
    fit_func = partial(_stim_nons_diff_LR, agent=agent, permute=False)

    fit_test_data = {'test_var_names':agent.param_names,
                     'true_fits': pp.map(fit_func, [sessions]*n_true_fit)}
    
    for session in sessions:
        session.true_stim_trials = deepcopy(session.stim_trials)
    
    print('Fitting permuted datasets.')

    _stim_nons_diff_LR_ = partial(_stim_nons_diff_LR, agent=agent, permute=True)
    fit_test_data['perm_fits'] = []
    for i, perm_fit in enumerate(pp.imap(_stim_nons_diff_LR_, [sessions]*n_perms, ordered=False)):
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        fit_test_data['perm_fits'].append(perm_fit)
        if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)
    
    for session in sessions: 
        session.stim_trials = session.true_stim_trials
        del(session.true_stim_trials)

    return fit_test_data

def _stim_nons_diff_LR(sessions, agent, permute):
    '''Evaluate difference between regression weights for stim and non-stim trials
    , if permute=True the stim/non-stim trial labels are randomly permuted before fitting.'''
    if permute:
        for session in sessions:
            np.random.shuffle(session.stim_trials)
    agent.trial_select['invert_mask'] = True
    fit_A = mf.fit_population(sessions, agent) # Non-stim trials fit.
    agent.trial_select['invert_mask'] = False
    fit_B = mf.fit_population(sessions, agent) # Stim trials fit.
    differences = fit_A['pop_dists']['means']-fit_B['pop_dists']['means']
    return {'fit_A': fit_A,
            'fit_B': fit_B,
            'differences': differences}

def log_reg_stim_x_group_interaction(sessions_A, sessions_B, predictors='standard', n_perms=5000,
                                     n_true_fit=11, file_name=None, agent=None):
    '''Permutation test to evaluate whether the effect of stimulation in group A is 
    different from the effect of stimulation in group B.  The stim effect (difference 
    between stim and non-stim trials) is evaluated for groups A and B and a difference
    between the effects is calculated. An ensemble of permuted datasets is then created
    by permuting subjects between groups and used to calculate the null distribution 
    of the effect difference.'''

    if not agent: agent = lr.config_log_reg(predictors, trial_mask='stim_trials')

    print('Fitting original dataset.')
    fit_test_data = {'test_var_names':agent.param_names[:],
                     'true_fits' : pp.map(_LR_interaction_fit,
                                      [(sessions_A, sessions_B, agent)]*n_true_fit)}

    perm_datasets = [_permuted_dataset(sessions_A, sessions_B, 'cross_subject') + [agent] 
                     for _ in range(n_perms)]

    fit_test_data['perm_fits'] = []

    for i, perm_fit in enumerate(pp.imap(_LR_interaction_fit, perm_datasets, ordered=False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)

    return fit_test_data

def _LR_interaction_fit(fit_data):
    # Evaluate group-by-stim interaction difference.
    sessions_A, sessions_B, agent = fit_data

    # Evaluate difference between stim and non-stim trials for each group.
    diff_A = _stim_nons_diff_LR(sessions_A, agent, permute=False)['differences']
    diff_B = _stim_nons_diff_LR(sessions_B, agent, permute=False)['differences']

    return {'differences': diff_A - diff_B}

# Group comparison utility functions.

def model_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='SD'):
    '''Compare two different model fits.'''
    mp.model_fit_plot(fit_1, fig_no, col='b', clf=clf , x_offset=-0.11, sub_medians=sub_medians, ebars=ebars)
    mp.model_fit_plot(fit_2, fig_no, col='r', clf=False, x_offset= 0.11, title=title, 
                      sub_medians=sub_medians, ebars=ebars)

def lagged_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='pm95'):
    mp.lagged_fit_plot(fit_1, fig_no, ebars, linestyle='-', cm=plt.cm.jet, xo=-0.00, sub_MAP=sub_medians)
    mp.lagged_fit_plot(fit_2, fig_no, ebars, linestyle=':', cm=plt.cm.hsv, xo= 0.00, sub_MAP=sub_medians, clf=False)

def _model_fit_P_values(fit_test_data, file_name=None):
    '''Evaluate P values from differences between true and permuted datasets'''
    true_differences = np.median([f['differences'] for f in fit_test_data['true_fits']], axis=0)
    perm_differences = np.array([f['differences'] for f in fit_test_data['perm_fits']])
    diff_ranks = np.mean(true_differences > perm_differences, 0)
    P_values = 2*np.minimum(diff_ranks,1-diff_ranks)
    n_perms = len(fit_test_data['perm_fits'])
    P_value_dict = OrderedDict([(pn,pv) for pn, pv in
                                zip(fit_test_data['test_var_names'], P_values)])
    diff_rank_dict = OrderedDict([(pn,dr) for pn, dr in
                                zip(fit_test_data['test_var_names'], diff_ranks)])
    fit_test_data.update({'true_differences': true_differences,
                          'perm_differences': perm_differences,
                          'P_values'        : P_value_dict,
                          'diff_ranks'      : diff_rank_dict,
                          'n_perms'         : n_perms})
    _print_P_values(fit_test_data['P_values'], n_perms, file_name)

def _print_P_values(P_value_dict, n_perms=None, file_name=None):
    if file_name: 
        _print_P_values(P_value_dict, n_perms) # Print to standard out then print to file.
    file = open(file_name + '.txt', 'w') if file_name else None
    print('P values' + (' ({} permutations):'.format(n_perms) if n_perms else ':'), file=file)
    name_len = max([len(name) for name in P_value_dict.keys()])
    for pn, pv in P_value_dict.items():
        stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        print('   ' + pn.ljust(name_len) + ': {:.4f}'.format(pv) + '  ' + stars, file=file)
    if file_name: file.close() 

#------------------------------------------------------------------------------------
# RL - LR correlation.
#------------------------------------------------------------------------------------

def RL_LR_correlation(sessions, fig_no=1):
    '''Correlate the effect of stimulation on the transition predictor
    with RL model paramters across subjects'''
    # Fit RL model to all trials.
    RL_agent = rl.MFmoMF_MB_dec(['bs','rb','ec','mc'])
    RL_fit = mf.fit_population(sessions, RL_agent)
    # Fit regression model seperately to stim and non-stim trial.
    LR_model = lr.config_log_reg()
    LR_model.trial_select['trial_mask'] = 'stim_trials'
    LR_model.trial_select['invert_mask'] = False
    LR_fit_stim = mf.fit_population(sessions, LR_model)
    LR_model.trial_select['invert_mask'] = True
    LR_fit_nons = mf.fit_population(sessions, LR_model)
    # Make data frame with parameter fits for each subject.
    ses_LR_params_stim =  np.vstack([sf['params_T'] for sf in LR_fit_stim['session_fits']])
    ses_LR_params_nons =  np.vstack([sf['params_T'] for sf in LR_fit_nons['session_fits']])
    ses_RL_params = np.vstack([sf['params_T'] for sf in RL_fit['session_fits']])
    ses_df = pd.DataFrame({pn: ses_RL_params[:,i] for i,pn in enumerate(RL_agent.param_names)})
    ses_df['d_trans'] = (ses_LR_params_stim[:,LR_model.param_names.index('trans_CR')] -
                         ses_LR_params_nons[:,LR_model.param_names.index('trans_CR')])
    ses_df['subject'] = np.array([s.subject_ID for s in sessions])
    sub_df = ses_df.groupby('subject').mean()
    # Plot correlation of G_mb with stim effect on transition predictor.
    plt.figure(fig_no, clear=True, figsize=[3.3,3])
    regplot('G_mb', 'd_trans', sub_df)
    plt.xlabel('Model-based weight')
    plt.ylabel('Stim change in\ntransition predictor')
    plt.tight_layout()
    res = linregress(sub_df['G_mb'], sub_df['d_trans'])
    print('Slope: {:.3f} r: {:.3f} P value: {:.4f}'.format(
        res.slope, res.rvalue, res.pvalue))
    # Regress stim effect with multiple RL model parameters.
    X = sub_df[['G_mb','G_td','G_tdm','mc']]
    X.insert(0,'const',1)
    print(OLS(sub_df['d_trans'], X).fit().summary())

#--------------------------------------------------------------------------------------------------
#  Permuted dataset generation.
#---------------------------------------------------------------------------------------------------

def _permuted_dataset(sessions_A, sessions_B, perm_type='ignore_subject'):
    ''' Generate permuted datasets by randomising assignment of sessions between groups A and B.
    perm_type argument controls how permutations are implemented:
    'within_subject' - Permute sessions within subject such that each permuted group has the same
                     number of session from each subject as the true datasets.
    'cross_subject' - All sessions from a given subject are assigned to one or other of the permuted datasets.
    'ignore_subject' - The identity of the subject who generated each session is ignored in the permutation.
    'within_group' - Permute subjects within groups that are subsets of all subjects.  
                     Animal assignment to groups is specified by groups argument which should be 
                     a list of lists of animals in each grouplt.
    '''
    assert perm_type in ('within_subject', 'cross_subject', 'ignore_subject',
                         'within_sub_&_cyc'), 'Invalid permutation type.'
    all_sessions = sessions_A + sessions_B
    all_subjects = list(set([s.subject_ID for s in all_sessions]))

    if perm_type == 'ignore_subject':  # Shuffle sessions ignoring which subject each session is from.        
        shuffle(all_sessions)
        perm_ses_A = all_sessions[:len(sessions_A)]
        perm_ses_B = all_sessions[len(sessions_A):]

    elif perm_type == 'cross_subject':  # Permute subjects across groups (used for cross subject tests.)
        n_subj_A     = len(set([s.subject_ID for s in sessions_A]))        
        shuffle(all_subjects)   
        perm_ses_A = [s for s in all_sessions if s.subject_ID in all_subjects[:n_subj_A]]
        perm_ses_B = [s for s in all_sessions if s.subject_ID in all_subjects[n_subj_A:]]
    
    elif perm_type == 'within_subject': # Permute sessions keeping number from each subject in each group constant.
        perm_ses_A = []
        perm_ses_B = []
        for subject in all_subjects:
            subject_sessions_A = [s for s in sessions_A if s.subject_ID == subject]
            subject_sessions_B = [s for s in sessions_B if s.subject_ID == subject]
            all_subject_sessions = subject_sessions_A + subject_sessions_B
            shuffle(all_subject_sessions)
            perm_ses_A += all_subject_sessions[:len(subject_sessions_A)]
            perm_ses_B += all_subject_sessions[len(subject_sessions_A):]
    
    elif perm_type == 'within_sub_&_cyc': # Permute sessions keeping number from each subject and cycle in each group constant.
        perm_ses_A = []
        perm_ses_B = []
        all_cycles = list(set([s.cycle for s in all_sessions]))
        for subject in all_subjects:
            for cycle in all_cycles:
                sub_cyc_sessions_A = [s for s in sessions_A if 
                    s.subject_ID == subject and s.cycle == cycle]
                sub_cyc_sessions_B = [s for s in sessions_B if 
                    s.subject_ID == subject and s.cycle == cycle]
                all_sub_cyc_sessions = sub_cyc_sessions_A + sub_cyc_sessions_B
                shuffle(all_sub_cyc_sessions)
                perm_ses_A += all_sub_cyc_sessions[:len(sub_cyc_sessions_A)]
                perm_ses_B += all_sub_cyc_sessions[len(sub_cyc_sessions_A):]

    return [perm_ses_A, perm_ses_B]