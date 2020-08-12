''' Plotting and analysis functions.'''

import pylab as plt
import numpy as np

import Two_step.utility as ut 
import Two_step.plotting as plo 

# Session plot ------------------------------------------------------------

def session_plot(session, ylabel = True):
    'Plot of choice moving average and reward block structure for single session.'
    choices, outcomes = session.unpack_CO(dtype = bool)
    mov_ave = ut.exp_mov_ave(choices)

    plt.plot(mov_ave,'k.-', markersize = 3)

    if hasattr(session, 'blocks'):
        #transitions = transitions == session.blocks['trial_trans_state'] # Convert transitions AB to transtions CR.
        for i in range(len(session.blocks['start_trials'])):
            y = [0.9,0.5,0.1][session.blocks['reward_states'][i]]  # y position coresponding to reward state.
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            plt.plot(x, [y,y], 'blue', linewidth = 2)  

    plt.plot([0,len(choices)],[0.75,0.75],'--k')
    plt.plot([0,len(choices)],[0.25,0.25],'--k')

    plt.xlabel('Trial Number')
    plt.yticks([0,0.5,1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(0,len(choices))
    if ylabel:plt.ylabel('Choice moving average')

def plot_session(session, fig_no = 1):
    'Plot data from a single session.'
    plt.figure(fig_no).clf()
    session_plot(session)

#----------------------------------------------------------------------------------
# Choice probability trajectory analyses.
#----------------------------------------------------------------------------------


def reversal_analysis(sessions, pre_post_trials = [-3,16], last_n = 3, fig_no = 3,
                      return_fits = False, clf = True, cols = 0):
    '''Analysis of choice trajectories around reversals in reward probability and
    transition proability.  Fits exponential decay to choice trajectories following reversals.'''

    p_1 = _end_of_block_p_correct(sessions, last_n)
    choice_trajectories = _get_choice_trajectories(sessions, pre_post_trials)
    per_subject_ct = _per_subject_choice_trajs(sessions, pre_post_trials)
    exp_fit = plo._fit_exp_to_choice_traj(choice_trajectories, p_1, pre_post_trials, last_n, double_exp=False)

    if return_fits:
        return {'p_1'     :p_1,
                'exp_fit' :exp_fit}
    else:
        colors = (('c','b'),('y','r'))[cols]
        plt.figure(fig_no)
        if clf:plt.clf()   
        plo._plot_mean_choice_trajectory(choice_trajectories, per_subject_ct, pre_post_trials, colors[0])
        plo._plot_exponential_fit(exp_fit, p_1, pre_post_trials, last_n, colors[1])
        plt.xlabel('Trials relative to block transition.')
        plt.ylabel('Fraction of choices to pre-reversal correct side.')
        print('Average block end choice probability: {}'.format(p_1))
        print('Tau: {}, P_0: {}'.format(exp_fit['tau'], exp_fit['p_0']))


def _end_of_block_p_correct(sessions, last_n = 5, stim_type = None, return_n_trials = False):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks.'
    n_correct, n_trials = (0, 0)
    for session in sessions:
        block_end_trials = np.zeros(session.n_trials, bool)
        for end_trial, rew_state in zip(session.blocks['end_trials'], session.blocks['reward_states']):
        	if rew_state != 1:
        		block_end_trials[end_trial - last_n:end_trial] = True
        if stim_type:
            assert stim_type in ['stim', 'non_stim'], 'Invalid stim_type argument.'
            if stim_type == 'stim':
                block_end_trials = block_end_trials &  session.stim_trials
            elif stim_type == 'non_stim':
                block_end_trials = block_end_trials & ~session.stim_trials
        n_trials += sum(block_end_trials)
        correct_choices = session.trial_data['choices'] != session.blocks['trial_rew_state'].astype(bool)
        n_correct += sum(correct_choices[block_end_trials])
    p_correct = n_correct / n_trials
    if return_n_trials:
        return p_correct, n_trials
    else:
        return p_correct

def _get_choice_trajectories(sessions, pre_post_trials):
    '''Evaluates choice trajectories around reversals.'''
    choice_trajectories = []
    n_trans_analysed = 0
    for session in sessions:
        blocks = session.blocks
        reversal_transitions = np.abs(blocks['reward_states'][1:] - 
        	                          blocks['reward_states'][:-1]) == 2
        n_trans_analysed +=sum(reversal_transitions)
        start_trials = np.array(blocks['start_trials'][1:])[reversal_transitions] # Start trials of blocks following selected transitions.
        end_trials = np.array(blocks['end_trials'][1:])[reversal_transitions]     # End trials of blocks following selected transitions.
        prev_start_trials = np.array(blocks['start_trials'][:-1])[reversal_transitions] # Start trials of blocks preceding selected transitions.
        reward_states = np.array(blocks['reward_states'][:-1])[reversal_transitions] # Reward state of blocks following selected transitions.

        for     start_trial,  end_trial,  prev_start_trial,  reward_state,  in \
            zip(start_trials, end_trials, prev_start_trials, reward_states):

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
            choice_trajectory = (choice_trajectory != bool(reward_state)).astype(float)
            if pad_start:
                choice_trajectory = np.hstack((ut.nans(pad_start), choice_trajectory))
            if pad_end:
                choice_trajectory = np.hstack((choice_trajectory, ut.nans(pad_end)))
            choice_trajectories.append(choice_trajectory)
    return np.vstack(choice_trajectories)

def _per_subject_choice_trajs(sessions, pre_post_trials):
    return np.array([np.nanmean(_get_choice_trajectories([s for s in sessions if s.subject_ID == ID],
                    pre_post_trials),0) for ID in set([s.subject_ID for s in sessions])])

def per_animal_end_of_block_p_correct(sessions, last_n = 5, fig_no = 1, col = 'b', clf = True):
    'Evaluate probabilty of correct choice in last n trials of non-neutral blocks on a per animals basis.'
    p_corrects = []
    for sID in sorted(set([s.subject_ID for s in sessions])):
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        p_corrects.append(_end_of_block_p_correct(subject_sessions, last_n))
    plt.figure(fig_no)
    if clf: plt.clf()
    n_sub = len(p_corrects)
    plt.scatter(0.2*np.random.rand(n_sub),p_corrects, s = 8,  facecolor= col, edgecolors='none', lw = 0)
    plt.errorbar(0.1, np.mean(p_corrects), np.std(p_corrects),linestyle = '', marker = '', linewidth = 2, color = col)
    plt.xlim(-1,1)
    plt.xticks([])
    plt.ylabel('Prob. correct choice')
    print(f'End of block probabiliy correct: {np.mean(p_corrects):.3f} + {np.std(p_corrects):3f}')
    return p_corrects
