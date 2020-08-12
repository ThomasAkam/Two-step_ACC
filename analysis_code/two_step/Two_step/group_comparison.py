import numpy as np
from random import shuffle
import pylab as plt

from . import plotting as pl
from . import model_fitting as mf
from . import model_plotting as mp
from . import parallel_processing as pp
from .stim_analysis import _model_fit_P_values

def group_info(sessions):
    return {'n_subjects'  : len(set([s.subject_ID for s in sessions])),
            'n_sessions' : len(sessions),
            'n_blocks'   : sum([len(s.blocks['start_trials']) - 1 for s in sessions]),
            'n_trials'   : sum([s.n_trials for s in sessions])}

# -------------------------------------------------------------------------------------
# Model fit comparison
# -------------------------------------------------------------------------------------

def model_fit_comparison(sessions_A, sessions_B, agent, fig_no=1, title=None, ebars='pm95'):
    ''' Fit the two groups of sessions with the specified agent and plot the results on the same axis.'''
    eval_BIC = ebars == 'pm95'
    fit_A = mf.fit_population(sessions_A, agent, eval_BIC=eval_BIC)
    fit_B = mf.fit_population(sessions_B, agent, eval_BIC=eval_BIC)
    model_fit_comp_plot(fit_A, fit_B, fig_no=fig_no, sub_medians=True, ebars=ebars)
    if title:plt.suptitle(title)

def model_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='pm95'):
    'Compare two different model fits.'
    mp.model_fit_plot(fit_1, fig_no, col='b', clf=clf , x_offset=-0.11, sub_medians=sub_medians, ebars=ebars)
    mp.model_fit_plot(fit_2, fig_no, col='r', clf=False, x_offset= 0.11, title=title, 
                      sub_medians=sub_medians, ebars=ebars)

# -------------------------------------------------------------------------------------
# Permutation tests.
# -------------------------------------------------------------------------------------

def model_fit_test(sessions_A, sessions_B, agent,  perm_type, n_perms=5000,
                   n_true_fit=5, file_name=None):

    '''Permutation test for significant differences in model fits between two groups of 
    sessions.  Outline of procedure:
    1. Perform model fitting seperately on both groups of sessions.
    2. Evaluate distance metric (KL divergence or difference of means) between fits
    for each parameter.
    3. Generate ensemble of resampled datasets in which sessions are randomly allocated
    to A or B.
    4. Perform model fitting and evalute distance metric for each resampled dataset to
    get a distribution of the distance metric under the null hypothesis that there is
    no difference between groups.
    5. Compare the true distance metric for each parameter with the null distribution
    to get a P value.'''

    mf._precalculate_fits(sessions_A + sessions_B, agent) # Store first round fits on sessions.

    test_var_names = agent.param_names[:] # Names of variables being permutation tested.
    if agent.type == 'RL': test_var_names += ['Model-based influence','Model-free influence']

    print('Fitting original dataset.')
    fit_test_data = {'test_var_names':test_var_names,
                     'true_fits' : pp.map(_fit_dataset,
                                      [(sessions_A, sessions_B, agent)]*n_true_fit)}

    perm_datasets = [_permuted_dataset(sessions_A, sessions_B, perm_type) + [agent] 
                     for _ in range(n_perms)]

    fit_test_data['perm_fits'] = []

    for i, perm_fit in enumerate(pp.imap(_fit_dataset, perm_datasets, ordered=False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9: _model_fit_P_values(fit_test_data, file_name)

    for session in sessions_A + sessions_B: del(session.fit) # Clear precalcuated fits.


def _fit_dataset(fit_data):
    # Evaluate and store fits for one dataset consisting of two sets of sessions,
    # along with distances between each parameter value.
    sessions_A, sessions_B, agent = fit_data   
    session_fits_A = [session.fit for session in sessions_A]
    session_fits_B = [session.fit for session in sessions_B] 
    fit_A = mf.fit_population(sessions_A, agent, init={'session_fits':session_fits_A}, verbose=False)
    fit_B = mf.fit_population(sessions_B, agent, init={'session_fits':session_fits_B}, verbose=False)
    differences = fit_A['pop_dists']['means']-fit_B['pop_dists']['means']
    return {'fit_A': fit_A,
            'fit_B': fit_B,
            'differences': differences}

# Reversal test  ------------------------------------------------------------------------

def reversal_test(sessions_A, sessions_B, perm_type, n_perms=5000, by_type=False, double_exp=False):
    ''' Permutation test for differences in the fraction correct at end of blocks and
    the time constant of adaptation to block transitions.'''

    rev_types = ['both_rev', 'rew_rev', 'trans_rev'] if by_type else ['both_rev']
    params = ['tau_f','tau_s','fs_mix'] if double_exp else ['tau']

    def _rev_fit_diff(sessions_A, sessions_B):
        '''Evaluate difference in asymtotic choice probability and reversal 
        time constants for pair of fits to reversal choice trajectories.'''
        fit_A = pl.reversal_analysis(sessions_A, return_fits=True, by_type=by_type, double_exp=double_exp)
        fit_B = pl.reversal_analysis(sessions_B, return_fits=True, by_type=by_type, double_exp=double_exp)
        diffs = [fit_A['p_e'] - fit_B['p_e']]
        for rev_type in rev_types:
            for param in params:
                try:
                    diffs.append(fit_A[rev_type][param]  - fit_B[rev_type][param])
                except TypeError:
                    diffs.append(np.nan)
        return np.array(diffs)

    def _print_reversal_P_vals(true_diffs, perm_diffs):
        diff_ranks = np.mean(true_diffs>perm_diffs, axis=0) 
        p_values = 2*np.minimum(diff_ranks,1-diff_ranks)
        print('Block end choice probability P value: {:.4f}'.format(p_values[0]))
        i = 1
        for rev_type in rev_types:
            for param in params:
                print('{} {} P value'.format(rev_type, param).ljust(36) +': {:.4f}'.format(p_values[i]))
                i += 1

    true_rev_fit_diff     = _rev_fit_diff(sessions_A, sessions_B)

    permuted_rev_fit_diff = np.zeros([n_perms, true_rev_fit_diff.shape[0]])
    print('Reversal analysis permutation test:')
    for i in range(n_perms):
        if i > 0 and i%10 == 9:
            print('Fitting permuted sessions, round: {} of {}'.format(i+1, n_perms))
        perm_ses_A, perm_ses_B = _permuted_dataset(sessions_A, sessions_B, perm_type)
        permuted_rev_fit_diff[i,:] = _rev_fit_diff(perm_ses_A, perm_ses_B)

    _print_reversal_P_vals(true_rev_fit_diff, permuted_rev_fit_diff)

#---------------------------------------------------------------------------------------------------
#  Permuted dataset generation.
#---------------------------------------------------------------------------------------------------

def _permuted_dataset(sessions_A, sessions_B, perm_type='ignore_subject'):
    ''' Generate permuted datasets by randomising assignment of sessions between groups A and B.
    perm_type argument controls how permutations are implemented:
    'within_subject' - Permute sessions within subject such that each permuted group has the same
                     number of session from each subject as the true datasets.
    'cross_subject' - All sessions from a given subject are assigned to one or other of the permuted datasets.
    '''
    assert perm_type in ('within_subject', 'cross_subject'), 'Invalid permutation type.'
    all_sessions = sessions_A + sessions_B
    all_subjects = list(set([s.subject_ID for s in all_sessions]))

    if perm_type == 'cross_subject':  # Permute subjects across groups (used for cross subject tests.)
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

    return [perm_ses_A, perm_ses_B]
