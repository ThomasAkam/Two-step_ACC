import pylab as plt
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact, ttest_rel, linregress
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
from . import simulation as sm
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
        

def RT_analysis(sessions, fig_no=1, title=None):
    '''Plot first step reaction times for stimulated vs non-stimulated trials
    on a per subject basis.'''
    sIDs = sorted(set([s.subject_ID for s in sessions]))
    plt.figure(fig_no, figsize = [6, 2.8]).clf()
    plt.subplot(1,2,1).locator_params(nbins = 3)
    all_non_stim_RTs, all_stim_RTs = ([], [])
    def _stim_RTs(sessions):
        stim_RTs = []
        non_stim_RTs = []
        for session in sessions:
            ITI_start_times = session.times['ITI_start']
            center_poke_times = session.ordered_times(['high_poke', 'low_poke'])
            reaction_times = 1000 * pl._latencies(ITI_start_times,  center_poke_times)
            reaction_times = reaction_times[:session.n_trials-1]
            stim_RTs.append(    reaction_times[ session.stim_trials[1:]])
            non_stim_RTs.append(reaction_times[~session.stim_trials[1:]])
        stim_RTs = np.concatenate(stim_RTs)
        non_stim_RTs = np.concatenate(non_stim_RTs)
        return(non_stim_RTs, stim_RTs)  
    median_RTs = []
    for sID in sIDs:
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        non_stim_RTs, stim_RTs = _stim_RTs(subject_sessions)
        median_RTs.append([np.median(non_stim_RTs), np.median(stim_RTs)])
        plt.plot([1,2], median_RTs[-1],'o-')
        all_non_stim_RTs.append(non_stim_RTs)
        all_stim_RTs.append(stim_RTs)
        print('Subject {} n stim trials : {}'.format(sID, len(stim_RTs)))
    median_RTs = np.array(median_RTs)
    print('Paired T-test p value: {}'.format(ttest_rel(median_RTs[:,0], median_RTs[:,1])[1]))
    stim_cum_hist, bin_edges     = pl._cumulative_histogram(np.concatenate(all_stim_RTs))
    non_stim_cum_hist, bin_edges = pl._cumulative_histogram(np.concatenate(all_non_stim_RTs))
    plt.xticks([1,2], ['Non-stim', 'Stim'])
    plt.ylabel('Median reaction time (ms)')
    plt.xlim(0.5,2.5)
    plt.ylim(500,2000)
    plt.subplot(1,2,2).locator_params(nbins = 3)
    plt.plot(bin_edges[:-1], non_stim_cum_hist)
    plt.plot(bin_edges[:-1], stim_cum_hist, 'r')
    plt.xlabel('Reaction time (ms)')
    plt.ylabel('Cum. fraction of trials')
    plt.tight_layout()
    if title:plt.suptitle(title)

def stim_timing_plot(sessions, fig_no=1):

    def _get_stim_timing(session):
        n_stim = len(session.times['stim_on']) - 1
        stim_timing = np.zeros([n_stim,5])
        for i in range(n_stim):
            stim_timing[i,0] = session.times['stim_on'][i]     # Stim on  time
            stim_timing[i,1] = session.times['stim_off'][i]    # Stim off time
            outcome_time = session.times['outcome'][session.times['outcome'].searchsorted(stim_timing[i,1])-1]
            post_choice_ind = session.times['choice'].searchsorted(outcome_time)
            stim_timing[i,2] = session.times['choice'][post_choice_ind-1] # Previous choice time
            stim_timing[i,3] = outcome_time                    # Outcome time
            stim_timing[i,4] = session.times['choice'][post_choice_ind]   # Subsequent choice time
        stim_timing = stim_timing - np.tile(stim_timing[:,0],[5,1]).T # Make times relative to stim on.
        return stim_timing

    if type(sessions) == list: # List of sessions passed in.
        stim_timings = np.vstack([_get_stim_timing(s) for s in sessions])
    else: # Single session past in.
        stim_timings = _get_stim_timing(sessions)

    n_stim = stim_timings.shape[0]
    stim_timings = stim_timings[stim_timings[:,1].argsort()] # Sort by stim duration.
    plt.figure(fig_no).clf()
    plt.plot(stim_timings[:,:2].T,np.tile(np.arange(n_stim),(2,1)),'r')
    plt.plot(stim_timings[:,2],np.arange(n_stim),'b.')
    plt.plot(stim_timings[:,4],np.arange(n_stim),'b.')
    plt.plot(stim_timings[:,3],np.arange(n_stim),'k.')
    plt.xlim(-5,8)
    plt.ylim(-1,n_stim + 1)


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



def bias_analysis(sessions, fig_no=1, title=None):
    'Plot fraction of high choices for stim and non-stim trials.'
    sIDs = sorted(set([s.subject_ID for s in sessions]))
    plt.figure(fig_no, figsize = [2.5,2.3]).clf()
    stim_p_high, nons_p_high = ([],[])
    for sID in sIDs:
        sub_sessions = [s for s in sessions if s.subject_ID == sID]
        stim_n_trial, stim_n_high, nons_n_trial, nons_n_high = (0,0,0,0)
        for session in sub_sessions:
            choices = session.trial_data['choices'].astype(bool)
            stim_n_trial += sum(session.stim_trials)
            stim_n_high  += sum(session.stim_trials & choices)
            nons_n_trial += sum(~session.stim_trials)
            nons_n_high  += sum(~session.stim_trials & choices)
        stim_p_high.append(stim_n_high/stim_n_trial)
        nons_p_high.append(nons_n_high/nons_n_trial)
        plt.plot([1,2], [nons_p_high[-1], stim_p_high[-1]],'o-')
    plt.xticks([1,2], ['Non-stim', 'Stim'])
    plt.ylabel('Fraction high choice')
    plt.xlim(0.5,2.5)
    if title:plt.suptitle(title)
    print('Paired T-test p value: {}'.format(ttest_rel(stim_p_high, nons_p_high)[1]))

#------------------------------------------------------------------------------------
# Correct probability analysis.
#------------------------------------------------------------------------------------

def p_correct_analysis(sessions, fig_no=1, title=None, last_n=20):
    '''Plot fraction correct in end trials of non-neutral blocks for stimulated vs
    non-stimulated trials on a per subject basis.'''
    sIDs = sorted(set([s.subject_ID for s in sessions]))
    plt.figure(fig_no, figsize = [2.5,2.3]).clf()
    stim_p_corrects, nons_p_corrects = ([],[])
    for sID in sIDs:
        sub_sessions = [s for s in sessions if s.subject_ID == sID]
        stim_p_correct, n_stim_trials = _eval_p_correct(sub_sessions, True , last_n)
        nons_p_correct, n_nons_trials = _eval_p_correct(sub_sessions, False, last_n)
        plt.plot([1,2], [nons_p_correct, stim_p_correct],'o-')
        stim_p_corrects.append(stim_p_correct)
        nons_p_corrects.append(nons_p_correct)
        n_correct_stim = int(stim_p_correct * n_stim_trials)
        n_incorrect_stim = n_stim_trials - n_correct_stim
        n_correct_nons = int(nons_p_correct * n_nons_trials)
        n_incorrect_nons = n_nons_trials - n_correct_nons
        subject_P_value = fisher_exact([[n_correct_stim  , n_correct_nons  ],
                                        [n_incorrect_stim, n_incorrect_nons]])[1]
        print('Subject {} n stim trials : {}, Fisher exact P value: {}'.format(sID, n_stim_trials, round(subject_P_value,3)))
    print('Paired T-test p value: {}'.format(ttest_rel(stim_p_corrects, nons_p_corrects)[1]))
    plt.xticks([1,2], ['Non-stim', 'Stim'])
    plt.ylabel('Fraction correct')
    plt.xlim(0.5,2.5)
    if title:plt.suptitle(title)

def _eval_p_correct(sessions, stim, last_n=15):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks.'
    n_correct, n_trials = (0, 0)
    for session in sessions:
        if last_n == 'all':  # Use all trials in block non neutral blocks.
            selected_trials = session.select_trials('all', block_type='non_neutral')
        else:  # Use only last_n  trials of non neutral blocks. 
            selected_trials = session.select_trials('end', last_n, block_type='non_neutral')
        if stim:
            selected_trials = selected_trials &  session.stim_trials
        else:
            selected_trials = selected_trials & ~session.stim_trials
        n_trials += sum(selected_trials)
        correct_choices = session.trial_data['choices'] == \
                          np.array(session.blocks['trial_rew_state'],   bool) ^ \
                          np.array(session.blocks['trial_trans_state'], bool)
        n_correct += sum(correct_choices[selected_trials])
    p_correct = n_correct / n_trials
    return p_correct, n_trials

#------------------------------------------------------------------------------------
# Logistic regression analyses.
#------------------------------------------------------------------------------------

def logistic_regression_comparison(sessions, predictors='standard', fig_no=1,
                                   title=None, agent=None, lagged_plot=False):
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

    stim_nons_difference = _stim_nons_diff_ps_LR if post_stim else _stim_nons_diff_LR

    print('Fitting original dataset.')
    
    fit_func = partial(stim_nons_difference, agent=agent, permute=False)

    fit_test_data = {'test_var_names':agent.param_names,
                     'true_fits': pp.map(fit_func, [sessions]*n_true_fit)}
    
    for session in sessions:
        session.true_stim_trials = deepcopy(session.stim_trials)
    
    print('Fitting permuted datasets.')

    _stim_nons_diff_LR_ = partial(stim_nons_difference, agent=agent, permute=True)
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


def _stim_nons_diff_ps_LR(sessions, agent, permute):
    '''Version of _stim_nons_diff_LR which evaluates parameter difference between
    non-stim and post-stim trials.'''
    if permute:
        for session in sessions:
            np.random.shuffle(session.stim_trials)
    for s in sessions:
        s.post_stim_trials = np.hstack([False, s.stim_trials[:-1]])
        s.nons_trials = ~ (s.stim_trials | s.post_stim_trials)
    agent.trial_select['invert_mask'] = False
    agent.trial_select['trial_mask'] = 'nons_trials'
    fit_A = mf.fit_population(sessions, agent)
    agent.trial_select['trial_mask'] = 'post_stim_trials'
    fit_B = mf.fit_population(sessions, agent)
    for s in sessions:
        del s.post_stim_trials
        del s.nons_trials
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


def post_stim_logistic_regression(sessions, predictors='standard', fig_no=1,
                                   title=None, agent=None, lagged_plot=False):
    '''Logistic regression analysis of behaviour on trial after stim trials.'''
    if not agent: agent = lr.config_log_reg(predictors)
    for s in sessions:
        s.post_stim_trials = np.hstack([False, s.stim_trials[:-1]])
        s.nons_trials = ~ (s.stim_trials | s.post_stim_trials)
    agent.trial_select['invert_mask'] = False
    agent.trial_select['trial_mask'] = 'post_stim_trials'
    fit_post_stim = mf.fit_population(sessions, agent, eval_BIC=True)
    agent.trial_select['trial_mask'] = 'nons_trials'
    fit_nons = mf.fit_population(sessions, agent, eval_BIC=True)
    if lagged_plot:  # Used plot for lagged regression model.
        lagged_fit_comp_plot(fit_nons, fit_post_stim, fig_no, ebars='pm95')
    else:
        model_fit_comp_plot(fit_nons, fit_post_stim, fig_no, ebars='pm95')
    plt.ylim(-1,1.5)
    if title: plt.suptitle(title)
    for s in sessions:
        del s.post_stim_trials
        del s.nons_trials

# Group comparison utility functions.

def model_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='SD'):
    'Compare two different model fits.'
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

#------------------------------------------------------------------------------------
# RL analyses.
#------------------------------------------------------------------------------------

def model_fit_plot(population_fit, fig_no=1, title=None, sub_medians=True, ebars='pm95'):
    '''Plot model fit where some parameters take seperate values on stim and non_stim
    trials.  Function creates two seperate pop_fits one of which has non-stim values
    and the other of which has stim-values with NaNs for parameters that are not split.
    The two seperate fits are then plotted using the standard model_fit_plot()'''

    param_names = population_fit['param_names']
    bp_names = [pn for pn in param_names if pn[-2:] != '_s'] 
    n_ind = [param_names.index(n) for n in bp_names] # Index of non-stim parameters.
    s_ind = [param_names.index(n+'_s') if n+'_s' in param_names else None
             for n in bp_names] #Index of stim parameter value

    def get_values(values):
        values_n = values[n_ind]
        values_s = np.array([values[i] if i is not None else np.NaN
                             for i in s_ind])
        return values_n, values_s

    pop_fit_n = deepcopy(population_fit)
    pop_fit_n['param_names']  = bp_names
    pop_fit_n['param_ranges'] = [population_fit['param_ranges'][i] for i in n_ind]
    pop_fit_s = deepcopy(pop_fit_n)

    pop_means_n, pop_means_s = get_values(population_fit['pop_dists']['means'])
    pop_SDs_n, pop_SDs_s = get_values(population_fit['pop_dists']['SDs'])
    pop_fit_n['pop_dists'] = {'means': pop_means_n, 'SDs': pop_SDs_n}
    pop_fit_s['pop_dists'] = {'means': pop_means_s, 'SDs': pop_SDs_s}

    if ebars=='pm95':
        means_hessian_n, means_hessian_s = get_values(population_fit['iBIC']['means_hessian'])
        pop_fit_n['iBIC'] = {'means_hessian': means_hessian_n}
        pop_fit_s['iBIC'] = {'means_hessian': means_hessian_s}

    for i, ses_fit in enumerate(population_fit['session_fits']):
        params_T_n, params_T_s = get_values(ses_fit['params_T'])
        params_U_n, params_U_s = get_values(ses_fit['params_U'])

        pop_fit_n['session_fits'][i]['params_T'] = params_T_n
        pop_fit_s['session_fits'][i]['params_T'] = params_T_s
        pop_fit_n['session_fits'][i]['params_U'] = params_U_n
        pop_fit_s['session_fits'][i]['params_U'] = params_U_s

    mp.model_fit_plot(pop_fit_n, fig_no, clf = True,  col = 'b', x_offset = -0.1,
                      sub_medians=sub_medians, ebars=ebars)
    mp.model_fit_plot(pop_fit_s, fig_no, clf = False, col = 'r', x_offset =  0.1, title=title,
                      sub_medians=sub_medians, ebars=ebars)

def RL_fit_test(sessions, agent_class, stim_params='all', n_perms=1000, n_true_fit=5, method='ind',
                n_fit_locked=5, file_name=None):
    '''Perform permutation testing to evaluate whether difference in logistic regression fits
    between stim and non stim trial data is statistically significant - use a fit of the model
    in which parameters are constrained to be the same on stim and non-stim trials as starting
    condition for fit in which parameters are allowed to be different.'''

    assert method in ['ind', 'full','part','init'], 'Invalid method'

    if method == 'ind':    # Fit stim and non-stim parameters completely independently.
        n_fit_locked=0
    elif method == 'full': # Fit to convergence with stim and non-stim parameters locked then allow to diverge.
        fit_lck_max_iter, fit_lck_tol = (200, 0.0001)
    elif method == 'part': # Fit part way to convergence with stim and non-stim parameters locked then allow to diverge.
        fit_lck_max_iter, fit_lck_tol = (200, 0.001)
    elif method == 'init': # Fit single EM round with stim and non-stim parameters locked then allow to diverge.
        fit_lck_max_iter, fit_lck_tol = (1  , 0.001)

    agent = agent_class(stim_params)

    if n_fit_locked:

        print('Initial fit with with stim and non-stim parameters locked.')

        agent_locked = agent_class([]) # Agent with stim and non-stim parameters locked to same value.

        if n_fit_locked > 1: 
            fit_locked = mf.repeated_fit_population(sessions, agent_locked, n_draws=5000, n_repeats=n_fit_locked,
                                                    tol=fit_lck_tol, max_iter=fit_lck_max_iter, verbose=True)
        else:
            fit_locked = mf.fit_population(sessions, agent_locked, tol=fit_lck_tol, max_iter=fit_lck_max_iter)

        session_fits = _locked_to_split(agent_locked, agent, fit_locked)
    else:
        session_fits, fit_locked = (None, None)

    fit_test_data = {'test_var_names': [pn for i, pn in enumerate(agent.bp_names)
                                        if not agent.np_ind[i] == agent.sp_ind[i]],
                     'fit_locked'      : fit_locked}

    print('Fitting original dataset.')

    true_fit_func = partial(_stim_nons_diff_RL, agent=agent, session_fits=session_fits, permute=False)

    fit_test_data['true_fits'] = pp.map(true_fit_func, [sessions]*n_true_fit)

    print('Fitting permuted datasets.')
    
    for session in sessions:
        session.true_stim_trials = deepcopy(session.stim_trials)
    
    perm_fit_func = partial(_stim_nons_diff_RL, agent=agent, session_fits=session_fits, permute=True)
    fit_test_data['perm_fits'] = []
    for i, perm_fit in enumerate(pp.imap(perm_fit_func, [sessions]*n_perms, ordered=False)):
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        fit_test_data['perm_fits'].append(perm_fit)
        if i > 0 and i%10 == 9:
            if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)
    
    for session in sessions: 
        session.stim_trials = session.true_stim_trials
        del(session.true_stim_trials)

    return fit_test_data

def _stim_nons_diff_RL(sessions, agent, session_fits, permute):
    '''Evaluate difference between stim and non-stim trial parameters, if permute=True the stim/non-stim
    trial labels are randomly permuted before fitting.'''
    init = {'session_fits':session_fits} if session_fits else None
    if permute:
        for session in sessions:
            np.random.shuffle(session.stim_trials)
    fit = mf.fit_population(sessions, agent, init=init)

    differences = []
    for i in range(len(agent.bp_names)):
        if not agent.np_ind[i] == agent.sp_ind[i]: # Param takes seperate values on stim trials.
            differences.append(fit['pop_dists']['means'][agent.np_ind[i]] - 
                               fit['pop_dists']['means'][agent.sp_ind[i]])
    differences = np.array(differences)
    return {'fit': fit, 'differences': differences}

def _locked_to_split(agent_locked, agent_split, fit_locked):
    '''Create session_fits to initialise EM for agent_split whose paramters take different values
    on stim and non-stim trials given fit_locked which is fit for equivilent agent_locked
    whose parameters are constrained to be the same on stim and non stim trials.'''
    param_map = np.zeros(agent_split.n_params, int)
    
    for i,pn in enumerate(agent_split.param_names):
        base_pn = pn[:-2] if pn[-2:] == '_s' else pn
        param_map[i] = agent_locked.param_names.index(base_pn)
    
    session_fits = [{'params_U' : session_fit['params_U'][param_map],
                     'diag_hess': session_fit['diag_hess'][param_map]}
                     for session_fit in fit_locked['session_fits']]
    return session_fits

def RL_stim_x_group_interaction(sessions_A, sessions_B, agent_class, stim_params='all', n_perms=5000,
                                n_true_fit=5, file_name=None):
    '''Permutation test for RL agents to evaluate whether the effect of stimulation in group A
     is different from the effect of stimulation in group B.'''

    agent = agent_class(stim_params)
    agent_locked = agent_class([])

    print('Fitting original dataset.')

    fit_test_data = {'test_var_names':[pn for i, pn in enumerate(agent.bp_names)
                                        if not agent.np_ind[i] == agent.sp_ind[i]],
                     'true_fits' : pp.map(_RL_interaction_fit,
                                      [(sessions_A, sessions_B, agent, agent_locked)]*n_true_fit)}

    perm_datasets = [_permuted_dataset(sessions_A, sessions_B, 'cross_subject') + [agent, agent_locked] 
                     for _ in range(n_perms)]

    print('Fitting permuted datasets.')

    fit_test_data['perm_fits'] = []
    for i, perm_fit in enumerate(pp.imap(_RL_interaction_fit, perm_datasets, ordered = False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)

    return fit_test_data

def _RL_interaction_fit(fit_data):
    # Evaluate group-by-stim interaction difference.
    sessions_A, sessions_B, agent, agent_locked = fit_data   

    # Initial fitting with stim and non-stim parameters locked.
    session_fits_A = _locked_to_split(agent_locked, agent, mf.fit_population(sessions_A, agent_locked))
    session_fits_B = _locked_to_split(agent_locked, agent, mf.fit_population(sessions_B, agent_locked))
    
    # Fit and evaluate difference between stim and non stim paramters staring fit from locked fits.
    diffs_A =  _stim_nons_diff_RL(sessions_A, agent, session_fits_A, permute=False)['differences']
    diffs_B =  _stim_nons_diff_RL(sessions_B, agent, session_fits_B, permute=False)['differences']

    return {'differences': diffs_A-diffs_B}


def two_stage_fit(sessions, agent_class, eval_BIC=False, rand_init=False):
    ''' Fit agent to population by first fitting to convergence assuming no 
    difference between stimulated and non-stimulated trials, then fit full
    model using this fit to initialise the fitting.'''
    agent_locked = agent_class([]) # Agent with stim and non-stim parameters locked to same value.
    fit_locked = mf.fit_population(sessions, agent_locked, init='rand' if rand_init else None)
    agent = agent_class('all') # Agent with stim and non-stim parameters free to differ.
    session_fits = _locked_to_split(agent_locked, agent, fit_locked)
    return mf.fit_population(sessions, agent, init={'session_fits':session_fits}, eval_BIC=eval_BIC)


def repeated_two_stage_fit(sessions, agent_class, n_draws=1000, n_repeats=10):
    '''Run two_stage_fit repeatedly with randomised intial population level
    parameters and return fit with best integrated likelihood.'''
    fit_func = partial(two_stage_fit, sessions, rand_init=True, eval_BIC={'n_draws':n_draws})
    repeated_fits = pp.map(fit_func, [agent_class]*n_repeats)
    best_fit = repeated_fits[np.argmax([fit['iBIC']['int_lik'] for fit in repeated_fits])]
    best_fit['repeated_fits'] = repeated_fits # Store all fits on best fit.
    return best_fit

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

#------------------------------------------------------------------------------------
# RL model lesioning.
#------------------------------------------------------------------------------------

def Lesion_RL_model(agent_class, RL_fit, stim_params, stim_values=None, predictors='standard',
                    fig_no=1, n_trials=500, n_ses=4000, LR_agent=None):
    '''Simulate the effect of opto stim on an RL agent by setting specified 
    parameters to zero on stim trials. A logistic regression analysis is performed to 
    evaluate the effect of the stimulation.  The optional stim values argument allows
    a dict to be pased in with the stim params names as keys which specifies the value
    the parameter takes on stim trials.'''

    RL_agent = agent_class(stim_params)

    assert set(RL_fit['param_names']) == set(RL_agent.bp_names), 'Fit does not match agent.'

    pop_dists_sim = {'means':np.zeros(RL_agent.n_params),
                     'SDs'  :np.zeros(RL_agent.n_params)}
    for i, param_name in enumerate(RL_agent.param_names):
        if param_name[-2:] == '_s': # Stim trial parameter. 
            if stim_values is None: 
                # Set paramameters to zero on stim trials.
                pop_dists_sim['means'][i] = mf._trans_TU([1e-8],
                                                [RL_agent.param_ranges[i]])
                pop_dists_sim['SDs'][i]   = 1e-8
            else: # Set parameters to specified value.
                j = RL_fit['param_names'].index(param_name[:-2])
                pop_dists_sim['means'][i] = mf._trans_TU([stim_values[param_name[:-2]]],
                                                         [RL_agent.param_ranges[i]])
                pop_dists_sim['SDs'][i]   = RL_fit['pop_dists']['SDs'][j]               
        else:
            j = RL_fit['param_names'].index(param_name)
            pop_dists_sim['means'][i] = RL_fit['pop_dists']['means'][j]
            pop_dists_sim['SDs'][i]   = RL_fit['pop_dists']['SDs'][j]
        
    pop_fit_sim = {'pop_dists'   : pop_dists_sim,
                   'param_names' : RL_agent.param_names,
                   'param_ranges': RL_agent.param_ranges}

    sim_sessions = sm.sim_sessions_from_pop_fit(RL_agent, pop_fit_sim,
                                                n_ses, n_trials)
    print('n sim trials: {}'.format(sum([s.n_trials for s in sim_sessions])))
    if not LR_agent:
        LR_agent = lr.config_log_reg(predictors)
    LR_agent.trial_select['trial_mask'] = 'stim_trials'
    LR_agent.trial_select['invert_mask'] = True
    nons_LR_fit = mf.fit_population(sim_sessions, LR_agent)
    LR_agent.trial_select['invert_mask'] = False
    stim_LR_fit = mf.fit_population(sim_sessions, LR_agent)

    plt.figure(fig_no, figsize = [3,2.3]).clf()
    plt.plot(np.arange(LR_agent.n_params)+0.5, nons_LR_fit['pop_dists']['means'],
             linestyle = '', marker = 'o', markeredgecolor = 'none',
             markersize = 7, color = 'b')
    plt.plot(np.arange(LR_agent.n_params)+0.5, stim_LR_fit['pop_dists']['means'],
             linestyle = '', marker = 'o', markeredgecolor = 'none',
             markersize = 7, color = 'r')    
    plt.xlim(0 ,LR_agent.n_params)
    plt.plot([0,LR_agent.n_params],[0,0],'k')
    plt.ylabel('Log odds')
    plt.xticks(np.arange(LR_agent.n_params)+0.5, LR_agent.param_names,
                rotation = -45, ha = 'left')
    plt.title(RL_agent.name + ': ' + str(stim_params))
    #plt.xlim(4,7)
    #plt.ylim(-0.1,0.4)


def set_of_lesions_analysis(agent_class, RL_fit, predictors='standard',
                            start_fig_no=1, n_trials=500, n_ses=4000, LR_agent=None):
    '''Run a set of model lesioning analyses to evaluate the effect of lesions
    of learning or performance in model-based and model-free system.'''
    def analysis_func(stim_params, fig_no):
        Lesion_RL_model(agent_class, RL_fit, stim_params, None, predictors,
                         fig_no, n_trials, n_ses, LR_agent)
    analysis_func(['G_td'], start_fig_no)
    analysis_func(['G_tdm'], start_fig_no+1)
    analysis_func(['G_mb'], start_fig_no+2)
    analysis_func(['alpQ'], start_fig_no+3)
    analysis_func(['alpT'], start_fig_no+5)
    analysis_func(['mc'], start_fig_no+7)