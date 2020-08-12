# This script can be used to generate figures from the  two-step ACC inhibition 
# dataset.  To use it, import the script and then call the functions
# corresponding to individual figure panels.


from Two_step import di, sa, pp

#----------------------------------------------------------------------------
# Data import.
#----------------------------------------------------------------------------

# Import data from experiment 1.

exp_ACC_1 = di.Experiment('2015-07-09-JAWS')

ACC_jaws_sID_1 = [299, 300, 301, 302, 309, 310, 311, 312]
ACC_ctrl_sID_1 = [303, 304, 305, 306, 307, 308]

dates_stim_1 = ['2015-09-01','2015-09-02','2015-09-03','2015-09-04','2015-09-05',
                '2015-09-06','2015-09-08','2015-09-09','2015-09-10','2015-09-11',
                '2015-09-12','2015-09-14','2015-09-15', '2015-09-16','2015-09-17',
                '2015-09-18','2015-09-19', '2015-09-21', '2015-09-22']

sessions_jaws_1 = exp_ACC_1.get_sessions(ACC_jaws_sID_1, dates = dates_stim_1)
sessions_ctrl_1 = exp_ACC_1.get_sessions(ACC_ctrl_sID_1, dates = dates_stim_1)

# Import data from experiment 2.

exp_ACC_2 = di.Experiment('2016-02-07-JAWS')

ACC_jaws_sID_2 = [384,385,386,387,390,391,392,393]
ACC_ctrl_sID_2 = [380,381,382,383,388,389]

dates_stim_2 = ['2016-03-30','2016-03-31','2016-04-01','2016-04-03','2016-04-04',
                '2016-04-05','2016-04-06','2016-04-07','2016-04-08','2016-04-09',
                '2016-04-11','2016-04-12','2016-04-13','2016-04-14','2016-04-15',
                '2016-04-16','2016-04-18','2016-04-19','2016-04-20','2016-04-21']

sessions_jaws_2 = exp_ACC_2.get_sessions(ACC_jaws_sID_2, dates = dates_stim_2)
sessions_ctrl_2 = exp_ACC_2.get_sessions(ACC_ctrl_sID_2, dates = dates_stim_2)

# Combine data from both experiments.

sessions_jaws = sessions_jaws_1 + sessions_jaws_2
sessions_ctrl = sessions_ctrl_1 + sessions_ctrl_2

# Selection based on histology.

ACC_jaws_sID_good = [299, 300, 302, 310, 311, 312, 384, 385, 386, 390, 392]
sessions_jaws = [s for s in sessions_jaws if s.subject_ID in ACC_jaws_sID_good]

def save_experiments():
    '''Save the experiments data as .pkl files - greatly speeds up 
    subsequent loading of experiments.'''
    for experiment in [exp_ACC_1, exp_ACC_2]:
        experiment.save()

#----------------------------------------------------------------------------
# Figure
#----------------------------------------------------------------------------

def figure_6(multiprocessing=False):
    if multiprocessing:
        pp.enable_multiprocessing()
    sa.logistic_regression_comparison(sessions_jaws, title='JAWS', fig_no='6D1')
    sa.logistic_regression_comparison(sessions_ctrl, title='CTRL', fig_no='6D2')
    sa.RL_LR_correlation(sessions_jaws, fig_no='6E')

#----------------------------------------------------------------------------
# Permutation tests.
#----------------------------------------------------------------------------

def JAWS_LR_permutation_test(multiprocessing=False, n_perms=5000):
    '''n_core : number of cores to use for parallel processing
       n_perms: number of permutations to run.'''
    if multiprocessing:
        pp.enable_multiprocessing()
    sa.logistic_regression_test(sessions_jaws, fig_no=0, n_perms=n_perms,
                                file_name='JAWS_LR_full')
                                
def CTRL_LR_permutation_test(multiprocessing=False, n_perms=5000):
    if multiprocessing:
        pp.enable_multiprocessing()
    sa.logistic_regression_test(sessions_ctrl, fig_no=0, n_perms=n_perms,
                                file_name='CTRL_LR_full')

def LR_interaction_perm_test(multiprocessing=False, n_perms=5000):
    if multiprocessing:
        pp.enable_multiprocessing()
    sa.log_reg_stim_x_group_interaction(sessions_jaws, sessions_ctrl,
        n_perms=n_perms, file_name='LR_interaction_test')