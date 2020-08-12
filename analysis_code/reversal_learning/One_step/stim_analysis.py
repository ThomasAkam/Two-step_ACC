from . import logistic_regression as lr 
import Two_step.stim_analysis as sa


def logistic_regression_comparison(sessions, lags=3, fig_no=1, title=None):
    agent = lr.config_log_reg(lags=lags, trial_mask='stim_trials')
    sa.logistic_regression_comparison(sessions, fig_no=fig_no, title=title, 
                                      agent=agent, lagged_plot=True)


def logistic_regression_test(sessions, lags=3, n_perms=5000, n_true_fit=5, file_name=None):
    agent = lr.config_log_reg(lags=lags, trial_mask='stim_trials')
    sa.logistic_regression_test(sessions, n_perms=n_perms, n_true_fit=n_true_fit,
                                 file_name=file_name, agent=agent)


def log_reg_stim_x_group_interaction(sessions_A, sessions_B, lags=3, n_perms=5000, n_true_fit=5,
                                     file_name=None):
    agent = lr.config_log_reg(lags=lags, trial_mask='stim_trials')
    sa.log_reg_stim_x_group_interaction(sessions_A, sessions_B, n_perms=n_perms,
                                         n_true_fit=n_true_fit, file_name=file_name, agent=agent)

