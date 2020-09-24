# This script can be used to generate the panels in figure S1
# showing data from the task version without reversals in the
# Transition matrix. # To generate the panes import the
# script and call the function figure_S1().

from Two_step import di, lr, mf, pl, mp, gc, pp
from script_baseline import sessions

# Load data
exp_NTR = di.Experiment('2015-04-02-no_transition_reversals')

sessions_ntr = exp_NTR.get_sessions('all',[22,-1])

def figure_S1(multiprocessing=False):
    pl.reversal_analysis(sessions_ntr, fig_no='S1A')
    pl.stay_probability_analysis(sessions_ntr, fig_no='S1B')
    if multiprocessing: 
        pp.enable_multiprocessing()
    lr.logistic_regression(sessions, 'standard', fig_no='S1C')
    lr.logistic_regression(sessions, 'lagged'  , fig_no='S1D')

def permuation_tests(multiprocessing=False):
    gc.reversal_test(sessions, sessions_ntr, 'cross_subject')
    if multiprocessing: 
        pp.enable_multiprocessing()
    gc.model_fit_test(sessions, sessions_ntr, LR1_model   , 'cross_subject', file_name='LR1_ntr_test')
    gc.model_fit_test(sessions, sessions_ntr, LR_lag_model, 'cross_subject', file_name='LR_lag_ntr_test')