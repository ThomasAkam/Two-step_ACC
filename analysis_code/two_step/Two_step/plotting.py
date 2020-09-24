''' Plotting and analysis functions.'''

import pylab as plt
import numpy as np

from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.stats import ttest_rel, sem
from sklearn.utils import resample

from . import utility as ut

#----------------------------------------------------------------------------------
# Session plot.
#----------------------------------------------------------------------------------

def session_plot(session, show_TO=False, ylabel=True):
    '''Plot of choice moving average and reward block structure for single session.'''
    choices, transitions, second_steps, outcomes = session.unpack_trial_data(dtype=bool)
    second_steps = second_steps * 1.1-0.05
    mov_ave = ut.exp_mov_ave(choices, initValue=0.5)

    if hasattr(session, 'blocks'):
        #transitions = transitions == session.blocks['trial_trans_state'] # Convert transitions AB to transtions CR.
        for i in range(len(session.blocks['start_trials'])):
            y = [0.9,0.5,0.1][session.blocks['reward_states'][i]]  # y position coresponding to reward state.
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            if session.blocks['transition_states'][i]:
                plt.plot(x, [y,y], 'orange', linewidth=2)
            else:
                y = 1 - y  # Invert y position if transition is inverted.
                plt.plot(x, [y,y], 'purple', linewidth=2)

    plt.plot(mov_ave,'k.-', markersize=3)    

    if show_TO:
        def symplot(y,guard,symbol):
            x_ind = np.where(guard)[0]
            plt.plot(x_ind,y[x_ind],symbol, markersize=5)
        symplot(second_steps,  transitions &  outcomes,'ob' )
        symplot(second_steps,  transitions & ~outcomes,'xb')
        symplot(second_steps, ~transitions &  outcomes,'og')
        symplot(second_steps, ~transitions & ~outcomes,'xg')  
    plt.plot([0,len(choices)],[0.75,0.75],'--k')
    plt.plot([0,len(choices)],[0.25,0.25],'--k')

    plt.xlabel('Trial Number')
    plt.yticks([0,0.5,1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(0,len(choices))
    if ylabel:plt.ylabel('Choice moving average')

def block_structure_plot(session, fig_no=1):
    '''plot the block structure for a single session.'''
    x = np.vstack([session.blocks['start_trials'], 
                   session.blocks['end_trials']]).T.reshape([1,-1])[0]
    rs = np.vstack([session.blocks['reward_states'], 
                   session.blocks['reward_states']]).T.reshape([1,-1])[0]
    rew_left = np.array([0.2,0.4,0.8])[rs]
    rew_rght = np.array([0.8,0.4,0.2])[rs]
    ts = np.vstack([session.blocks['transition_states'], 
                    session.blocks['transition_states']]).T.reshape([1,-1])[0]
    trans_prob = np.array([0.2,0.8])[ts]
    plt.figure(fig_no, figsize=[6.5, 4]).clf()
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1]) 
    plt.subplot(gs[0])
    session_plot(session)
    plt.xlabel('')
    plt.subplot(gs[1])
    plt.plot(x,rew_rght,'r', linewidth=1.5, alpha=0.8)
    plt.plot(x,rew_left,'b', linewidth=1.5, alpha=0.8)
    plt.ylim(0,1)
    plt.xlim(x[0],x[-1])
    plt.ylabel('Reward probs.')
    plt.subplot(gs[2])
    plt.plot(x,  trans_prob,'orange', linewidth=1.5, alpha=0.8)
    plt.plot(x,1-trans_prob,'purple', linewidth=1.5, alpha=0.8)
    plt.ylim(0,1)
    plt.xlim(x[0],x[-1])
    plt.ylabel('Transition probs.')
    plt.xlabel('Trial number')

#----------------------------------------------------------------------------------
# Reaction time analysis.
#----------------------------------------------------------------------------------

def reaction_times_second_step(sessions, fig_no=1, return_subs=False):
    '''Reaction times for second step pokes as function of common / rare transition.'''

    def get_SSRTs(sessions):
        common_RTs = []
        rare_RTs = []
        for session in sessions:
            reaction_times = _second_step_RTs(session)
            transitions = session.blocks['trial_trans_state'] == session.trial_data['transitions']  # common vs rare.                 
            min_len = min(len(reaction_times), len(transitions))
            reaction_times = reaction_times[:min_len]
            transitions = transitions[:min_len]
            common_RTs.append(reaction_times[ transitions])
            rare_RTs.append(  reaction_times[~transitions])
        RT_common = np.median(np.hstack(common_RTs))
        RT_rare   = np.median(np.hstack(rare_RTs))
        return RT_common, RT_rare
    
    subjects = sorted(list(set([s.subject_ID for s in sessions])))
    sub_RTs_common = np.zeros(len(subjects))
    sub_RTs_rare   = np.zeros(len(subjects))
    for i, sID in enumerate(subjects):
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        sub_RTs_common[i], sub_RTs_rare[i] = get_SSRTs(subject_sessions)

    if return_subs: # Return for individual subjects
        return sub_RTs_common, sub_RTs_rare

    mean_RT_common = np.mean(sub_RTs_common)
    mean_RT_rare   = np.mean(sub_RTs_rare)
    SEM_RT_common = sem(sub_RTs_common)
    SEM_RT_rare   = sem(sub_RTs_rare)

    if fig_no:
        print('Mean reaction times: Common: {:.1f}, Rare:{:.1f}'.format(mean_RT_common,mean_RT_rare))
        plt.figure(fig_no, figsize = [1.8,2.3]).clf()
        plt.bar([1,2],[mean_RT_common, mean_RT_rare], yerr = [SEM_RT_common,
                                                              SEM_RT_rare])
        plt.xlim(0.5,2.5)
        plt.ylim(mean_RT_common * 0.8, mean_RT_rare * 1.1)
        plt.xticks([1, 2], ['Common', 'Rare'])
        plt.title('Second step reaction times')
        plt.ylabel('Reaction time (ms)')
        print(('Paired t-test P value: {}'.format(ttest_rel(sub_RTs_common,
                                                            sub_RTs_rare)[1])))
    else:
        return mean_RT_common, mean_RT_rare, SEM_RT_common, SEM_RT_rare

def _second_step_RTs(session):
    '''Return the second step reaction times for a session in ms'''
    t = session.times
    left_reaction_times  = _latencies(t['left_active' ], t['left_poke' ])
    right_reaction_times = _latencies(t['right_active'], t['right_poke'])
    return 1000*np.hstack((left_reaction_times,right_reaction_times))[
        np.argsort(np.hstack((t['left_active'],t['right_active'])))]

def _latencies(event_times_A, event_times_B):
    '''Evaluate the latency between each event A and the first event B
     that occurs afterwards.'''
    latencies = np.outer(event_times_B, np.ones(len(event_times_A))) - \
                np.outer(np.ones(len(event_times_B)), event_times_A)
    latencies[latencies <= 0] = np.inf
    latencies = np.min(latencies,0)
    return latencies

#----------------------------------------------------------------------------------
# Trial timing analysis
#----------------------------------------------------------------------------------

def trial_timings_analysis(sessions, fig_no=1, clf=True, col='b', max_t=10000, log_x=False):
    '''Evaluate the distribution of latencies between different trial events and return 
    the median latencies as a dictionary.
    '''
    OC_lat_rew, OC_lat_non, CC_lat_rew, CC_lat_non, CO_lat = ([],[],[],[],[])
    for session in sessions:
        outcomes = session.trial_data['outcomes'].astype(bool)
        ses_OC_lat = _latencies(session.times['outcome'],session.times['choice'])
        OC_lat_rew.append(ses_OC_lat[np.where( outcomes[:len(ses_OC_lat)])])
        OC_lat_non.append(ses_OC_lat[np.where(~outcomes[:len(ses_OC_lat)])])
        ses_CC_lat = session.times['choice'][1:] - session.times['choice'][:-1]
        CC_lat_rew.append(ses_CC_lat[np.where( outcomes[:len(ses_CC_lat)])])
        CC_lat_non.append(ses_CC_lat[np.where(~outcomes[:len(ses_CC_lat)])])
        ses_CO_lat = _latencies(session.times['choice'],session.times['outcome'])
        CO_lat.append(ses_CO_lat)
    OC_lat_rew = 1000*np.hstack(OC_lat_rew)
    OC_lat_non = 1000*np.hstack(OC_lat_non)
    CC_lat_rew = 1000*np.hstack(CC_lat_rew)
    CC_lat_non = 1000*np.hstack(CC_lat_non)
    CO_lat = 1000*np.hstack(CO_lat)
    OC_lat_rew = OC_lat_rew[np.isfinite(OC_lat_rew)]
    OC_lat_non = OC_lat_non[np.isfinite(OC_lat_non)]
    CO_lat = CO_lat[np.isfinite(CO_lat)]
    bin_edges0 = (np.logspace(np.log10(1000) , np.log10(max_t) , num=1000) if log_x 
                 else np.linspace(0,max_t, 1000)) 
    OC_cum_hist_rew, b = _cumulative_histogram(OC_lat_rew, bin_edges0)
    OC_cum_hist_non, b = _cumulative_histogram(OC_lat_non, bin_edges0)
    CC_cum_hist_rew, b = _cumulative_histogram(CC_lat_rew, bin_edges0)
    CC_cum_hist_non, b = _cumulative_histogram(CC_lat_non, bin_edges0)
    bin_edges1 = (np.logspace(np.log10(100) , np.log10(max_t) , num=1000) if log_x 
                 else np.linspace(0,max_t, 1000)) 
    CO_cum_hist, b = _cumulative_histogram(CO_lat, bin_edges1)
    OC_median_rew = np.median(OC_lat_rew)
    OC_median_non = np.median(OC_lat_non)
    CC_median_rew = np.median(CC_lat_rew)
    CC_median_non = np.median(CC_lat_non)
    CO_median = np.median(CO_lat)
    if fig_no:
        plt.figure(fig_no,figsize=[8,3])
        if clf: plt.clf()
        plt.subplot(1,3,1)
        plt.title('Choice - outcome')
        plot_func = plt.semilogx if log_x else plt.plot
        plot_func(bin_edges1[:-1],CO_cum_hist, color=col)
        plt.xlim(0,3000)
        plt.ylim(0,1)
        plt.xlabel('Time post choice (ms)')
        plt.ylabel('Fraction of trials')
        plt.subplot(1,3,2)
        plt.title('Outcome - choice')
        plot_func(bin_edges0[:-1],OC_cum_hist_rew, color=col)
        plot_func(bin_edges0[:-1],OC_cum_hist_non, color=col, linestyle='--')
        plt.xlim(0,max_t)
        plt.ylim(0,1)
        plt.xlabel('Time post outcome (ms)')
        plt.subplot(1,3,3)
        plt.title('Choice - choice')
        plot_func(bin_edges0[:-1],CC_cum_hist_rew, color=col)
        plot_func(bin_edges0[:-1],CC_cum_hist_non, color=col, linestyle='--')
        plt.xlim(0,max_t)
        plt.ylim(0,1)
        plt.xlabel('Time post choice (ms)')
        plt.tight_layout()
        print('Median choice-outcome latency: {:.0f}'.format(CO_median))
        print('Median outcome-choice latency: rew: {:.0f}, non: {:.0f}'
              .format(OC_median_rew, OC_median_non))
        print('Median choice-choice latency : rew: {:.0f}, non: {:.0f}'
              .format(CC_median_rew, CC_median_non))
    else:
        return {'CO': CO_median,'OC_rew': OC_median_rew,'OC_non': OC_median_non,
                'CC_rew': CC_median_rew, 'CC_non': CC_median_non}

def _cumulative_histogram(data, bin_edges=np.arange(0,3001)):
    h = np.histogram(data, bin_edges)[0]
    cum_hist = np.cumsum(h) / len(data)
    return cum_hist, bin_edges

#----------------------------------------------------------------------------------
# Poke timeing analysis.
#----------------------------------------------------------------------------------

def poke_timeing_analysis(sessions, fig_no=1):
    '''Plot histograms showing the timeing of pokes of different types around trial
    start and following common and rare transitions.'''
    fs_cent_poke = [] # First step center poke histograms
    fs_side_poke = [] # First step side poke histograms
    sc_cent_poke = [] # Second step common trans center poke histograms
    sr_cent_poke = [] # Second step rare trans center poke histograms
    sc_cors_poke = [] # Second step common trans correct side poke histograms
    sc_incs_poke = [] # Second step common trans inccorrect side poke histograms
    sr_cors_poke = [] # Second step rare trans correct side poke histograms
    sr_incs_poke = [] # Second step rare trans inccorrect side poke histograms
    bins_fs = np.arange(-1000,260,10)
    bins_ss = np.arange(0,610,10)
    for session in sessions:
        t = session.times
        CR = session.blocks['trial_trans_state'] == session.trial_data['transitions']  # Common/rare transtitions
        CR_l = CR[ session.trial_data['second_steps'].astype(bool)] # Common/rare for left second step
        CR_r = CR[~session.trial_data['second_steps'].astype(bool)] # Common/rare for right second step
        center_pokes = session.ordered_times(['high_poke', 'low_poke'])
        side_pokes   = session.ordered_times(['left_poke', 'right_poke'])
        second_steps = session.ordered_times(['left_active', 'right_active'])
        fs_cent_poke.append(PETH(center_pokes, t['trial_start'], bins_fs))
        fs_side_poke.append(PETH(side_pokes  , t['trial_start'], bins_fs))
        sc_cent_poke.append(PETH(center_pokes, second_steps[ CR], bins_ss))
        sr_cent_poke.append(PETH(center_pokes, second_steps[~CR], bins_ss))
        sc_cors_poke.append(PETH(t['left_poke' ], t['left_active' ][ CR_l], bins_ss))
        sc_cors_poke.append(PETH(t['right_poke'], t['right_active'][ CR_r], bins_ss))
        sc_incs_poke.append(PETH(t['left_poke' ], t['right_active'][ CR_r], bins_ss))
        sc_incs_poke.append(PETH(t['right_poke'], t['left_active' ][ CR_l], bins_ss))
        sr_cors_poke.append(PETH(t['left_poke' ], t['left_active' ][~CR_l], bins_ss))
        sr_cors_poke.append(PETH(t['right_poke'], t['right_active'][~CR_r], bins_ss))
        sr_incs_poke.append(PETH(t['left_poke' ], t['right_active'][~CR_r], bins_ss))
        sr_incs_poke.append(PETH(t['right_poke'], t['left_active' ][~CR_l], bins_ss))
    plt.figure(fig_no, figsize=[12,3.5], clear=True)
    plt.subplot(1,3,1)
    plt.fill_between(bins_fs[:-1], np.mean(fs_cent_poke,0), color='C0', alpha=0.5, label='center')
    plt.fill_between(bins_fs[:-1], np.mean(fs_side_poke,0), color='C1', alpha=0.5, label='side')
    plt.xlabel('Time relative to trial start (ms)')
    plt.ylabel('Poke count')
    plt.xlim(bins_fs[0], bins_fs[-1])
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.title('Pokes around trial start')
    plt.subplot(1,3,2)
    plt.fill_between(bins_ss[:-1], np.mean(sc_cent_poke,0), color='C0', alpha=0.5, label='center')
    plt.fill_between(bins_ss[:-1], np.mean(sc_cors_poke,0), color='C2', alpha=0.5, label='correct side')
    plt.fill_between(bins_ss[:-1], np.mean(sc_incs_poke,0), color='C3', alpha=0.5, label='incorrect side')
    plt.xlabel('Time relative to second-step state (ms)')
    plt.xlim(bins_ss[0], bins_ss[-1])
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.title('Pokes following common transition')
    plt.subplot(1,3,3)
    plt.fill_between(bins_ss[:-1], np.mean(sr_cent_poke,0), color='C0', alpha=0.5, label='center poke')
    plt.fill_between(bins_ss[:-1], np.mean(sr_cors_poke,0), color='C2', alpha=0.5, label='correct side')
    plt.fill_between(bins_ss[:-1], np.mean(sr_incs_poke,0), color='C3', alpha=0.5, label='incorrect side')
    plt.xlabel('Time relative to second-step state (ms)')
    plt.xlim(bins_ss[0], bins_ss[-1])
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.title('Pokes following rare transition')
    plt.tight_layout()


def PETH(times_A, times_B, bins):
        '''Return a PETH for times_A relative to times_B'''
        latencies = 1000*(times_A - times_B[:,None]).flatten()
        latencies = latencies[(latencies>bins[0]) & (latencies<bins[-1])]
        return np.histogram(latencies, bins)[0]

#----------------------------------------------------------------------------------
# Reversal analysis
#----------------------------------------------------------------------------------

def reversal_analysis(sessions, pre_post_trials=[-15,40], fig_no=1, return_fits=False, clf=True,
                      cols=0, by_type=False, title=None, plot_tau=False, double_exp=False):

    '''Analysis of choice trajectories around reversals in reward probability and
    transition proability.  Fits exponential decay to choice trajectories following reversals.'''

    if len(set(np.hstack([s.blocks['transition_states'] for s in sessions]))) == 1:
        by_type = False # Can't evaluate by reversal type if data has no transition reversals.

    last_n = - pre_post_trials[0]
    p_e = _end_of_block_p_correct(sessions, last_n)
    if by_type: # Analyse reversals in reward and transition probabilities seperately.
        per_subject_rr = _per_subject_ave_choice_trajs(sessions, 'reward_reversal', pre_post_trials)
        fit_rr = _fit_exp_to_choice_traj(per_subject_rr, p_e, pre_post_trials, last_n, double_exp)
        per_subject_tr = _per_subject_ave_choice_trajs(sessions, 'transition_reversal', pre_post_trials)
        fit_tr = _fit_exp_to_choice_traj(per_subject_tr, p_e,pre_post_trials, last_n, double_exp)
    else:
        fit_rr, fit_tr = (None, None)

    per_subject_br = _per_subject_ave_choice_trajs(sessions, 'any_reversal', pre_post_trials)
    fit_br = _fit_exp_to_choice_traj(per_subject_br, p_e, pre_post_trials, last_n, double_exp)

    if return_fits:
        return {'p_e'      :p_e,
                'rew_rev'  :fit_rr,
                'trans_rev':fit_tr,
                'both_rev' :fit_br}
    else:
        colors = (('c','b'),('y','r'))[cols]
        figsize = [8.5, 2.3] if by_type else [2.5,2.3]
        plt.figure(fig_no, figsize = figsize)
        if clf:plt.clf()   
        if by_type:
            plt.subplot(1,3,1)
            plt.title('Reversal in reward probabilities', fontsize = 'small')
            _plot_exponential_fit(fit_rr, p_e, pre_post_trials, last_n, colors[1], plot_tau)
            _plot_mean_choice_trajectory(per_subject_rr, pre_post_trials, colors[0])
            plt.subplot(1,3,2)
            plt.title('Reversal in transition probabilities', fontsize = 'small')
            _plot_exponential_fit(fit_tr, p_e, pre_post_trials, last_n, colors[1], plot_tau)
            _plot_mean_choice_trajectory(per_subject_tr, pre_post_trials, colors[0])
            plt.subplot(1,3,3)
            plt.title('Both reversals combined', fontsize = 'small')
        _plot_exponential_fit(fit_br, p_e, pre_post_trials, last_n, colors[1], plot_tau)
        _plot_mean_choice_trajectory(per_subject_br, pre_post_trials, colors[0])
        ax = plt.figure(fig_no).get_axes()[0]
        ax.set_xlabel('Trials relative to block transition.')
        ax.set_ylabel('Fraction of choices to pre-reversal correct side.')
        if title: plt.suptitle(title)
        print(('Average block end choice probability: {}'.format(p_e)))
        if double_exp:
            if by_type:
                print(('Reward probability reversal, tau fast: {:.2f}, tau slow: {:.2f},  fast weight: {:.2f}, P_0: {:.2f}'
                       .format(fit_rr['tau_f'], fit_rr['tau_s'],fit_rr['fs_mix'], fit_rr['p_0'])))
                print(('Trans. probability reversal, tau fast: {:.2f}, tau slow: {:.2f},  fast weight: {:.2f}, P_0: {:.2f}'
                       .format(fit_tr['tau_f'], fit_tr['tau_s'],fit_tr['fs_mix'], fit_tr['p_0'])))
            print(('Combined reversals,          tau fast: {:.2f}, tau slow: {:.2f},  fast weight: {:.2f}, P_0: {:.2f}'
                   .format(fit_br['tau_f'], fit_br['tau_s'],fit_br['fs_mix'], fit_br['p_0'])))
        else:
            if by_type:
                print(('Reward probability reversal, tau: {:.2f}, P_0: {:.2f}'.format(fit_rr['tau'], fit_rr['p_0'])))
                print(('Trans. probability reversal, tau: {:.2f}, P_0: {:.2f}'.format(fit_tr['tau'], fit_tr['p_0'])))
            print(('Combined reversals,          tau: {:.2f}, P_0: {:.2f}'.format(fit_br['tau'], fit_br['p_0'])))

def _block_index(blocks):
    '''Create dict of boolean arrays used for indexing block transitions,
    Note first value of index corresponds to second block of session.'''
    return {
    'to_neutral'          : np.array(blocks['reward_states'][1:]) == 1,
    'from_neutral'        : np.array(blocks['reward_states'][:-1]) == 1,
    'reward_reversal'     : np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2,
    'transition_reversal' : np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]),
    'any_reversal'        : (np.abs(np.array(blocks['reward_states'][:-1]) - np.array(blocks['reward_states'][1:])) == 2) | \
                            (np.array(blocks['reward_states'][:-1]) == np.array(blocks['reward_states'][1:]))}

def _get_choice_trajectories(sessions, trans_type, pre_post_trials):
    '''Evaluates choice trajectories around transitions of specified type. Returns float array
     of choice trajectories of size (n_transitions, n_trials). Choices are coded such that a 
    choice towards the option which is correct before the transition is 1, the other choice is 0,
    if the choice trajectory extends past the ends of the blocks before and after the transition
    analysed, it is padded with nans.'''
    choice_trajectories = []
    n_trans_analysed = 0
    for session in sessions:
        blocks = session.blocks
        selected_transitions = _block_index(blocks)[trans_type]
        n_trans_analysed +=sum(selected_transitions)
        start_trials = np.array(blocks['start_trials'][1:])[selected_transitions] # Start trials of blocks following selected transitions.
        end_trials = np.array(blocks['end_trials'][1:])[selected_transitions]     # End trials of blocks following selected transitions.
        prev_start_trials = np.array(blocks['start_trials'][:-1])[selected_transitions] # Start trials of blocks preceding selected transitions.
        transition_states = np.array(blocks['transition_states'][:-1])[selected_transitions] # Transition state of blocks following selected transitions.
        reward_states = np.array(blocks['reward_states'][:-1])[selected_transitions] # Reward state of blocks following selected transitions.

        for start_trial,  end_trial,  prev_start_trial,  reward_state,  transition_state in \
            zip(start_trials, end_trials, prev_start_trials, reward_states, transition_states):

            trial_range = start_trial + np.array(pre_post_trials)
            if trial_range[0] < prev_start_trial:
                pad_start = prev_start_trial - trial_range[0] 
                trial_range[0] = prev_start_trial
            else:
                pad_start = 0
            if trial_range[1] > end_trial:
                pad_end = trial_range[1] - end_trial
                trial_range[1] = end_trial
            else:
                pad_end = 0
            choice_trajectory = session.trial_data['choices'][trial_range[0]:trial_range[1]].astype(bool)                        
            choice_trajectory = (choice_trajectory == bool(reward_state) ^ bool(transition_state)).astype(float)
            if pad_start:
                choice_trajectory = np.hstack((ut.nans(pad_start), choice_trajectory))
            if pad_end:
                choice_trajectory = np.hstack((choice_trajectory, ut.nans(pad_end)))
            choice_trajectories.append(choice_trajectory)
    if choice_trajectories:
        return np.vstack(choice_trajectories)
    else:
        return np.empty([0,np.diff(pre_post_trials)[0]])

def _per_subject_ave_choice_trajs(sessions, trans_type, pre_post_trials):
    return np.array([np.nanmean(_get_choice_trajectories([s for s in sessions if s.subject_ID == ID],
                    trans_type, pre_post_trials),0) for ID in set([s.subject_ID for s in sessions])])

def _plot_mean_choice_trajectory(per_subject_ave_traj, pre_post_trials, col='b'):
    x = np.arange(pre_post_trials[0], pre_post_trials[1])
    per_sub_mean = np.nanmean(per_subject_ave_traj,0)
    per_sub_SD   = np.nanstd(per_subject_ave_traj,0)
    plt.fill_between(x, per_sub_mean-per_sub_SD, per_sub_mean+per_sub_SD, alpha=0.2, color=col)
    plt.plot(x, per_sub_mean ,col, linewidth=1.5)
    plt.plot([0,0],[0,1],'k--')
    plt.plot([pre_post_trials[0], pre_post_trials[1]-1],[0.5,0.5],'k:')
    plt.ylim(0,1)
    plt.xlim(pre_post_trials[0],pre_post_trials[1])

def _plot_exponential_fit(fit, p_e, pre_post_trials, last_n, col='r', plot_tau=False):
    t = np.arange(0,pre_post_trials[1])
    double_exp = 'tau_f' in fit.keys()
    if double_exp:
        exp_traj = _double_exp_choice_traj([fit['tau_f'], fit['tau_ratio'], fit['fs_mix']], fit['p_0'], p_e, t)
    else:
        exp_traj = _exp_choice_traj(fit['tau'], fit['p_0'], p_e, t)

    full_traj = np.hstack([ut.nans(-pre_post_trials[0]-last_n), np.ones(last_n) * fit['p_0'], exp_traj])
    plt.plot(np.arange(pre_post_trials[0], pre_post_trials[1]),full_traj, col, linewidth=1.5)
    if plot_tau: plt.plot([fit['tau'],fit['tau']],[0,1],':'+ col)
    plt.locator_params(nbins=4)

def _end_of_block_p_correct(sessions, last_n=15):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks.'
    n_correct, n_trials = (0, 0)
    for session in sessions:
        if last_n == 'all':  # Use all trials in non neutral blocks.
            block_end_trials = session.select_trials('all', block_type='non_neutral')
        else:  # Use only last_n  trials of non neutral blocks. 
            block_end_trials = session.select_trials('end', last_n, block_type='non_neutral')
        n_trials += sum(block_end_trials)
        correct_choices = session.trial_data['choices'] == \
                          np.array(session.blocks['trial_rew_state'],   bool) ^ \
                          np.array(session.blocks['trial_trans_state'], bool)
        n_correct += sum(correct_choices[block_end_trials])
    p_correct = n_correct / n_trials
    return p_correct

def _fit_exp_to_choice_traj(subject_ave_trajs, p_e, pre_post_trials,  last_n, double_exp):
    '''Fit an exponential or double exponential curve to the cross-subject average choice 
    trajectory using squared error cost function.'''
    t = np.arange(0,pre_post_trials[1])
    p_0 = np.nanmean(subject_ave_trajs[:,:-pre_post_trials[0]]) # Choice probability at end of previous block.
    mean_traj = np.nanmean(subject_ave_trajs,0)[-pre_post_trials[1]:]
    if double_exp: # Double exponential fit.
        params = minimize(_double_exp_fit_error, np.array([3.,10.,0.5]), args=(mean_traj, p_0, p_e, t),
                          method='L-BFGS-B', bounds=[(1e-6,200), (1.,500),(0,1)])['x']
        tau_f, tau_ratio, fs_mix = params
        tau_s = tau_f*tau_ratio
        return {'p_0':p_0,'tau_f':tau_f, 'tau_s': tau_s, 'tau_ratio': tau_ratio, 'fs_mix': fs_mix}
    else: # Single exponential fit.
        tau = minimize(_exp_fit_error, np.array([15.]), args=(mean_traj, p_0, p_e, t),
                       method='L-BFGS-B', bounds=[(1e-6,200)])['x'][0]
        return {'p_0':p_0,'tau':tau}

def _exp_fit_error(tau, mean_traj, p_0, p_e, t):
    return np.sum((mean_traj-_exp_choice_traj(tau, p_0, p_e, t))**2)

def _exp_choice_traj(tau, p_0, p_e, t):
    return (1. - p_e) + (p_0 + p_e - 1.) * np.exp(-t/tau)

def _double_exp_fit_error(params, mean_traj, p_0, p_e, t):
    return np.sum((mean_traj-_double_exp_choice_traj(params, p_0, p_e, t))**2)

def _double_exp_choice_traj(params, p_0, p_e, t):
    tau_f, tau_ratio, fs_mix = params
    tau_s = tau_f*tau_ratio
    return (1. - p_e) + (p_0 + p_e - 1.) * (fs_mix*np.exp(-t/tau_f)+(1-fs_mix)*np.exp(-t/tau_s))

def per_animal_end_of_block_p_correct(sessions, last_n=15, fig_no=1, col='b', clf=True, verbose=False):
    '''Evaluate probabilty of correct choice in last n trials of non-neutral blocks on
    a per animals basis.'''
    p_corrects = []
    for sID in sorted(set([s.subject_ID for s in sessions])):
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        p_corrects.append(_end_of_block_p_correct(subject_sessions, last_n))
        if verbose: print('subject {}: {:.3g}'.format(sID, p_corrects[-1]))
    if verbose: print('Mean: {:.3g}, SD: {:.3g}'.format(np.mean(p_corrects), np.std(p_corrects)))
    plt.figure(fig_no)
    if clf: plt.clf()
    n_sub = len(p_corrects)
    plt.scatter(0.2*np.random.rand(n_sub)-0.3*int(clf),p_corrects, s=8,  facecolor=col, edgecolors='none', lw=0)
    plt.errorbar(0.1-0.3*int(clf), np.mean(p_corrects), np.std(p_corrects),linestyle='', marker='', linewidth=2, color=col)
    plt.xlim(-1,1)
    plt.xticks([])
    plt.ylabel('Prob. correct choice')
    print(f'End of block probabiliy correct: {np.mean(p_corrects):.3f} + {np.std(p_corrects):3f}')
    return p_corrects

def reversal_tau_confidence_intervals(sessions, n_resample=1000, cross_subject=True):
    '''Evaluate confidence intervals for the reversal time constants by
    bootstrap resampling from population of sessions. If cross_subject is
    True, resampling is done at the level of subject, to give cross subject
    confidence intervals, if False resampling is done at the level of sessions'''
    def get_rev_data(sessions):
        r = reversal_analysis(sessions, by_type=True, return_fits= True)
        return np.array([r['p_e'], r['rew_rev']['tau'], r['trans_rev']['tau']])
    true_rev_data = get_rev_data(sessions)
    perm_rev_data = np.zeros([n_resample, 3])
    for i in range(n_resample):
        if i%10 == 0: print('Fitting resampled sessions {} of {}'
                            .format(i+1,n_resample))
        perm_rev_data[i,:] = get_rev_data(resample(sessions))
    SDs = np.std(perm_rev_data,0)
    tau_diff_true = true_rev_data[1]   - true_rev_data[2]
    tau_diff_perm = perm_rev_data[:,1] - perm_rev_data[:,2]
    td_P_value = 1.- np.mean(np.sign(tau_diff_perm) == np.sign(tau_diff_true))
    print('End of block P correct : {:.3f} ± {:.3f}'.format(true_rev_data[0],SDs[0]))
    print('Reward reversal tau    : {:.2f} ± {:.2f}'.format(true_rev_data[1],SDs[1]))
    print('Transition reversal tau: {:.2f} ± {:.2f}'.format(true_rev_data[2],SDs[2]))
    print('Tau difference P value : {:.3f}'.format(td_P_value))

#----------------------------------------------------------------------------------
# Stay probability Analysis
#----------------------------------------------------------------------------------

def stay_probability_analysis(sessions, ebars='SEM', selection='xtr', fig_no=1, by_trans=False, 
                              ylim=[0.,1], trial_mask=None, block_type='all', title=None):
    '''Stay probability analysis.'''
    assert ebars in [None, 'SEM', 'SD'], 'Invalid error bar specifier.'
    n_sessions = len(sessions)
    all_n_trials, all_n_stay = (np.zeros([n_sessions,12]), np.zeros([n_sessions,12]))
    for i, session in enumerate(sessions):
        trial_select = session.select_trials(selection, block_type=block_type)
        if trial_mask:
            trial_select = trial_select & trial_mask[i]
        trial_select_A = trial_select &  session.blocks['trial_trans_state']
        trial_select_B = trial_select & ~session.blocks['trial_trans_state']
        #Eval total trials and number of stay trial for A and B blocks.
        all_n_trials[i,:4] , all_n_stay[i,:4]  = _stay_prob_analysis(session, trial_select_A)
        all_n_trials[i,4:8], all_n_stay[i,4:8] = _stay_prob_analysis(session, trial_select_B)
        # Evaluate combined data.
        all_n_trials[i,8:] = all_n_trials[i,:4] + all_n_trials[i,[5,4,7,6]]
        all_n_stay[i,8:] = all_n_stay[i,:4] + all_n_stay[i,[5,4,7,6]]
    if not ebars: # Don't calculate cross-animal error bars.
        mean_stay_probs = np.nanmean(all_n_stay / all_n_trials, 0)
        y_err  = np.zeros(12)
    else:
        session_sIDs = np.array([s.subject_ID for s in sessions])
        unique_sIDs = list(set(session_sIDs))
        n_subjects = len(unique_sIDs)
        per_subject_stay_probs = np.zeros([n_subjects,12])
        for i, sID in enumerate(unique_sIDs):
            session_mask = session_sIDs == sID # True for sessions with correct animal ID.
            per_subject_stay_probs[i,:] = sum(all_n_stay[session_mask,:],0) / sum(all_n_trials[session_mask,:],0)
        mean_stay_probs = np.nanmean(per_subject_stay_probs, 0)
        if ebars == 'SEM':
            y_err = ut.nansem(per_subject_stay_probs, 0)
        else:
            y_err = np.nanstd(per_subject_stay_probs, 0)
    if fig_no:
        if by_trans: # Plot seperately by transition block type.
            plt.figure(fig_no).clf()
            plt.subplot(1,3,1)
            plt.bar(np.arange(1,5), mean_stay_probs[:4], yerr=y_err[:4])
            plt.ylim(ylim)
            plt.xlim(0.4,4.6)
            plt.title('A transitions normal.', fontsize='small')
            plt.xticks([1,2,3,4],['1/A', '1/B', '0/A', '0/B'])
            plt.ylabel('Stay Probability')
            plt.subplot(1,3,2)
            plt.bar(np.arange(1,5), mean_stay_probs[4:8], yerr=y_err[4:8])
            plt.ylim(ylim)
            plt.xlim(0.4,4.6)
            plt.title('B transitions normal.', fontsize='small')
            plt.xticks([1,2,3,4],['1/A', '1/B', '0/A', '0/B'])
            plt.subplot(1,3,3)
            plt.title('Combined.', fontsize='small')
        else:
            plt.figure(fig_no, figsize=[2.5,2.3]).clf()
        plt.bar(np.arange(1,5), mean_stay_probs[8:], yerr=y_err[8:])
        plt.ylim(ylim)
        plt.xlim(0.4,4.6)
        plt.xticks([1,2,3,4],['1/N', '1/R', '0/N', '0/R'])
        if title: plt.title(title)
    else:
        return per_subject_stay_probs[:,8:]

def _stay_prob_analysis(session, trial_select):
    '''Analysis for stay probability plots using binary mask to select trials.'''
    choices, transitions, outcomes = session.unpack_trial_data('CTO', bool)
    stay = choices[1:] == choices[:-1]
    transitions, outcomes, trial_select = (transitions[:-1], outcomes[:-1], trial_select[:-1])
    stay_go_by_type = [stay[( outcomes &  transitions) & trial_select],  # A transition, rewarded.
                       stay[( outcomes & ~transitions) & trial_select],  # B transition, rewarded.
                       stay[(~outcomes &  transitions) & trial_select],  # A transition, not rewarded.
                       stay[(~outcomes & ~transitions) & trial_select]]  # B transition, not rewarded.
    n_trials_by_type = [len(s) for s in stay_go_by_type]
    n_stay_by_type =   [sum(s) for s in stay_go_by_type]
    return n_trials_by_type, n_stay_by_type

#----------------------------------------------------------------------------------
# Functions called by session and experiment classes.
#----------------------------------------------------------------------------------

def plot_day(exp, day):
    if day < 0: day = exp.n_days + day + 1
    day_sessions = exp.get_sessions('all', days=day)
    plt.figure(day)
    for i, session in enumerate(day_sessions):
        plt.subplot(len(day_sessions),1,i+1)
        session_plot(session, ylabel=False)
        plt.ylabel(session.subject_ID)
    plt.suptitle('Day number: {}, Date: '.format(session.day) + session.date)
   
def plot_subject(exp, sID, day_range=[0, np.inf]):
    subject_sessions =  exp.get_sessions(sID, 'all')
    if hasattr(subject_sessions[0], 'day'):
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.day)
        sorted_sessions = [s for s in sorted_sessions if
                           s.day >= day_range[0] and s.day <= day_range[1]]
    else:
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.number)
    n_sessions = len(sorted_sessions)
    plt.figure(sID).clf()
    for i,session in enumerate(sorted_sessions):
        plt.subplot(n_sessions, 1, i+1)
        session_plot(session, ylabel=False)
        if hasattr(session, 'date'):
            plt.ylabel(session.date)
        else:
            plt.ylabel(session.number)
    plt.tight_layout()

def plot_session(session, fig_no=1):
    'Plot data from a single session.'
    plt.figure(fig_no, figsize=[7.5, 1.8]).clf()
    session_plot(session)