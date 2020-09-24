# This script can be used to generate figures from the probabilistic 
# reversal learning task.  To use it, import the script and then call 
# the functions corresponding to individual figure panels.

from One_step import *

#----------------------------------------------------------------------------
# Data import.
#----------------------------------------------------------------------------

# Import data from first experiment.

exp_one = di.Experiment('2015-09-01-JAWS')

jaws_sID_1 = [355,356,357,362,363,364,365,366]
ctrl_sID_1 = [358,359,360,361,367,368]

dates_stim_1 =  ['2015-10-09','2015-10-10','2015-10-12','2015-10-13',
                 '2015-10-14','2015-10-15','2015-10-16','2015-10-17',
                 '2015-10-19','2015-11-26','2015-11-27','2015-11-28',
                 '2015-11-30','2015-12-01','2015-12-02','2015-12-04',
                 '2015-12-05','2015-12-07','2015-12-08']

sessions_jaws_1 = exp_one.get_sessions(jaws_sID_1, dates=dates_stim_1)
sessions_ctrl_1 = exp_one.get_sessions(ctrl_sID_1, dates=dates_stim_1)

# Import data from second experiment.

exp_two = di.Experiment('2016-02-07-JAWS')

jaws_sID_2 = [384,385,386,387,391,392,393]
ctrl_sID_2 = [380,381,382,383,388]

dates_stim_2 = ['2016-05-02','2016-05-03','2016-05-04','2016-05-05','2016-05-06',
              '2016-05-07','2016-05-08','2016-05-09','2016-05-10','2016-05-11',
              '2016-05-16','2016-05-17','2016-05-18','2016-05-19','2016-05-20',
              '2016-05-21','2016-05-23','2016-05-24','2016-05-25','2016-05-26',
              '2016-05-27','2016-05-28','2016-05-30','2016-05-31','2016-06-01',
              '2016-06-02']

sessions_jaws_2 = exp_two.get_sessions(jaws_sID_2, dates=dates_stim_2)
sessions_ctrl_2 = exp_two.get_sessions(ctrl_sID_2, dates=dates_stim_2)

# Combine data.

sessions_jaws = sessions_jaws_1 + sessions_jaws_2
sessions_ctrl = sessions_ctrl_1 + sessions_ctrl_2 

# Exclude based on histology.

ACC_jaws_sID_good = [355,356,357,362,364,365, 384, 385, 386, 392]
sessions_jaws = [s for s in sessions_jaws if s.subject_ID in ACC_jaws_sID_good]

def save_experiments():
    '''Save the experiments data as .pkl files - greatly speeds up subsequent loading
    of experiments as the individual session files do not have to be read.'''
    for experiment in [exp_one, exp_two]:
        experiment.save()

#----------------------------------------------------------------------------
# Figure panels.
#----------------------------------------------------------------------------

def figure_S8():
    pl.plot_session(sessions_ctrl[0], fig_no='S8B')
    pl.reversal_analysis(sessions_ctrl, fig_no='S8C')
    sa.logistic_regression_comparison(sessions_jaws, fig_no='S8D1', title='JAWS')
    sa.logistic_regression_comparison(sessions_ctrl, fig_no='S8D2', title='GFP')

#----------------------------------------------------------------------------
# Permutation tests.
#----------------------------------------------------------------------------

def JAWS_LR_permutation_test(n_core=1, n_perms=5000):
    '''n_core : number of cores to use for parallel processing
       n_perms: number of permutations to run.'''
    pp.enable_multiprocessing(n_core)
    sa.logistic_regression_test(sessions_jaws, n_perms=n_perms, file_name='JAWS_PRL_LR_test')
                                
def CTRL_LR_permutation_test(n_core=1, n_perms=5000):
    pp.enable_multiprocessing(n_core)
    sa.logistic_regression_test(sessions_ctrl, n_perms=n_perms, file_name='CTRL_PRL_LR_test')

def LR_interaction_perm_test(n_core=1, n_perms=5000):
    pp.enable_multiprocessing(n_core)
    sa.log_reg_stim_x_group_interaction(sessions_jaws, sessions_ctrl,
        n_perms=n_perms, file_name='PRL_LR_interaction_test')