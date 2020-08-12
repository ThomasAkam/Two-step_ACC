# This script can be used to generate figures from the baseline two-step 
# task dataset.  To use it, import the script and then call the functions
# corresponding to individual figure panels.

from Two_step import di, pl, sm, lr, rl, gl

#----------------------------------------------------------------------------
# Data import.
#----------------------------------------------------------------------------

exp_base_0 = di.Experiment('2014-02-22-Baseline')
exp_base_1 = di.Experiment('2014-06-30-Baseline')
exp_base_2 = di.Experiment('2015-04-02-Baseline')

sessions = exp_base_0.get_sessions('all',[22,-1]) + \
           exp_base_1.get_sessions('all',[22,-1]) + \
           exp_base_2.get_sessions('all',[22,-1])

def save_experiments():
    '''Save the experiments data as .pkl files - greatly speeds up subsequent loading
    of experiments as the individual session files do not have to be read.'''
    for experiment in [exp_base_0, exp_base_1, exp_base_2]:
        experiment.save()

#----------------------------------------------------------------------------
# Figure panels.
#----------------------------------------------------------------------------

def figure_2D(): 
    pl.block_structure_plot(sessions[8], fig_no=1)  

def figure_2E():
    pl.reversal_analysis(sessions, by_type=True, fig_no=2)

def figure_2F():
    pl.reaction_times_second_step(sessions, fig_no=3)

def figure_3A():
    pl.stay_probability_analysis(sessions, fig_no=4, ylim=[0.5,0.75])

def figure_3B():
    sm.RL_agent_behaviour_comparison(sessions)

def figure_3S1():
    lr.logistic_regression(sessions, fig_no=6)
    
def Model_comparison(sessions):

    agents =   [rl.MF_MB(            ['bs','rb','ck']),
                rl.MF(               ['bs','rb','ck']),
                rl.MB(               ['bs','rb','ck']),
                rl.MF_MB_dec(        ['bs','rb','ck']), 
                rl.MF_dec(           ['bs','rb','ck']), 
                rl.MB_dec(           ['bs','rb','ck']), 
                rl.MF_MB_vdec(       ['bs','rb','ck']), 
                rl.MF_MB(            ['bs','rb',     'ec']),
                rl.MF_MB_vdec(       ['bs','rb',     'ec']), 
                rl.MF_MB_dec(        ['bs','rb',     'ec']), 
                rl.MF_MB_dec(        ['bs','rb',          'mc']), 
                rl.MF_MB_dec(        ['bs','rb',     'ec','mc']), 
                rl.MFmoMF_MB_dec(    ['bs','rb',     'ec','mc']),
                rl.MFmoMF_dec(       ['bs','rb',     'ec','mc']),
                rl.MB_dec(           ['bs','rb',     'ec','mc']),
                rl.MFmoMF_MB_dec_2lr(['bs','rb',     'ec','mc']),
                rl.MFmoMF_MB_dec_2sv(['bs','rb',     'ec','mc']), 
                rl.moMF_MB_dec(      ['bs','rb',     'ec','mc']),
                rl.MFmoMF_MB_vdec(   ['bs','rb',     'ec','mc']),
                rl.MF_moMF_MB_dec(   ['bs','rb',     'ec','mc']),
                rl.MFmoMF_MB_dec_net(['bs','rb',     'ec','mc']),
                rl.MFmoMF_MB_dec(    [     'rb',     'ec','mc']),
                rl.MFmoMF_MB_dec(    ['bs',          'ec','mc']),
                rl.MFmoMF_MB_dec(    ['bs','rb',          'mc']),
                rl.MFmoMF_MB_dec(    ['bs','rb',     'ec'     ]),
                rl.MFmoMF_MB_tdec(   ['bs','rb',     'ec','mc'])]
  
    ag_add_remove =    ['MFmoMF_MB_dec_bs_rb_ec_mc',
                        'MFmoMF_MB_dec_2lr_bs_rb_ec_mc',
                        'MFmoMF_MB_dec_2sv_bs_rb_ec_mc',
                        'moMF_MB_dec_bs_rb_ec_mc',
                        'MF_MB_dec_bs_rb_ec_mc',
                        'MFmoMF_dec_bs_rb_ec_mc',
                        'MFmoMF_MB_vdec_bs_rb_ec_mc',
                        'MF_moMF_MB_dec_bs_rb_ec_mc',
                        'MFmoMF_MB_dec_net_bs_rb_ec_mc',
                        'MFmoMF_MB_dec_rb_ec_mc',
                        'MFmoMF_MB_dec_bs_ec_mc',
                        'MFmoMF_MB_dec_bs_rb_mc',
                        'MFmoMF_MB_dec_bs_rb_ec',
                        'MFmoMF_MB_tdec_bs_rb_ec_mc']

    ag_construct = ['MF_MB_bs_rb_ck',
                    'MF_bs_rb_ck',
                    'MB_bs_rb_ck',
                    'MF_MB_dec_bs_rb_ck',
                    'MF_dec_bs_rb_ck',
                    'MB_dec_bs_rb_ck',
                    'MF_MB_vdec_bs_rb_ck',
                    'MF_MB_bs_rb_ec',
                    'MF_MB_vdec_bs_rb_ec',
                    'MF_MB_dec_bs_rb_ec',
                    'MF_MB_dec_bs_rb_mc',
                    'MF_MB_dec_bs_rb_ec_mc',
                    'MFmoMF_MB_dec_bs_rb_ec_mc',
                    'MFmoMF_dec_bs_rb_ec_mc',
                    'MB_dec_bs_rb_ec_mc']

    fits = mc.BIC_model_comparison(sessions, ag_construct, n_draws=5000, n_repeats=10, fig_no=1,
                                   save_exp=exp_base_2, file_name='baseline_model_comp_fits')

def GLMMs():
    # Lagged regression.
    lags = {p: ['1','2','3_4','5_8','8_12'] 
            for p in ['choice', 'outcome', 'trans_CR', 'trCR_x_out']}
    LR_model = gl.Two_step_LR_model(['side', 'choice', 'outcome', 'trans_CR', 'trCR_x_out'], lags=lags)
    gl.GLMM_fit(sessions, lr_model=LR_model, remove_rand_effects=['outcome_lag_5_8', 'trans_CR_lag_8_12'])
    # One trial back regression - all trials.
    gl.GLMM_fit(sessions)
    # One trial back regression - neutral blocks.
    LR_model = gl.Two_step_LR_model(block_type='neutral')
    gl.GLMM_fit(sessions, lr_model=LR_model)
