# This script generates the imagaing data figures 3, 4 and 5.  To generate the
# figures, import the script and call the function for the relevant figure.

import os
import numpy as np
import pandas as pd
from scipy import sparse

from Two_step import *

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

def min_coverage(session):
    '''Returns the minimum number of trials for trial types defined by a specific 
    choice, second step and outcome.'''
    n_img_trials = session.calcium_data['aligned']['spikes'].shape[0]
    c = session.trial_data['choices'     ][:n_img_trials].astype(bool)
    s = session.trial_data['second_steps'][:n_img_trials].astype(bool)
    o = session.trial_data['outcomes'    ][:n_img_trials].astype(bool)
    coverage = [np.sum( c &  s &  o),
                np.sum( c &  s & ~o),
                np.sum( c & ~s &  o),
                np.sum( c & ~s & ~o),
                np.sum(~c &  s &  o),
                np.sum(~c &  s & ~o),
                np.sum(~c & ~s &  o),
                np.sum(~c & ~s & ~o)]
    return np.min(coverage)


def remove_missing_data_trials(session, start_frame, n_frames, median_latencies):
    '''Two sessions have a chunk of inscopix data missing in the middle of 
    the session. This function removes the trials during the missing data
    from  both the session.calcium_data['aligned']['spikes'] matrix and from
    the session.trial_data matricies so the remaining data can be used.
    '''
    # Remove frame times corresponding to missing data.
    session.calcium_data['frame_times'] = np.hstack( 
        [session.calcium_data['frame_times'][:start_frame], 
         session.calcium_data['frame_times'][start_frame+n_frames:]])
    # Rerun alignment code
    session.calcium_data['aligned'] = ia.align_activity(session, median_latencies, 20, 'auto')
    # Remove trials during missing data period from data matricies.
    bad_trials = np.where((session.times['scope_frame'][start_frame] < 
                           session.times['trial_start']) & 
                          (session.times['trial_start'] < 
                           session.times['scope_frame'][start_frame+n_frames]))[0]
    n_bad_trials = len(bad_trials)
    session.calcium_data['aligned']['spikes'] = np.delete(
        session.calcium_data['aligned']['spikes'], bad_trials, 0)
    for k in session.trial_data.keys():
        session.trial_data[k] = np.delete(session.trial_data[k], bad_trials)
    for k in ('trial_trans_state','trial_rew_state'):
        session.blocks[k] = np.delete(session.blocks[k], bad_trials)
    session.n_trials = len(session.trial_data['choices'])
    # Update start and end trials of blocks to reflect removed trials.
    post_bad_trials_blocks = session.blocks['start_trials'] > bad_trials[-1]
    for i, post in enumerate(post_bad_trials_blocks):
        if post:
            session.blocks['start_trials'][i] = session.blocks['start_trials'][i] - n_bad_trials
            session.blocks['end_trials'  ][i] = session.blocks['end_trials'  ][i] - n_bad_trials
    assert np.all(np.diff(session.blocks['start_trials']) > 0), 'Unable to remove missing data trials'

def export_session(session, save_dir):
    '''Export behavioural and calcium imaging data for a session to a folder named
    'save_dir/session.file_name' containing:
    trial_data.csv  : choice and outcome times +  choices, second steps and outcomes
    activity.npy    : Neuronal calcium activity [n_neurons, n_frames]
    frame_times.npy : Times of scope frames.
    '''
    activity = session.calcium_data['spikes']
    frame_times = 1000*session.calcium_data['frame_times']

    choice_time  = 1000*session.times['choice'][:session.n_trials]            
    outcome_time = 1000*session.times['outcome'][:session.n_trials] 

    choice, second_step, outcome = session.unpack_trial_data('CSO')

    trial_data = pd.DataFrame({'choice_time'  : choice_time,
                               'outcome_time' : outcome_time,
                               'choice'       : choice,
                               'second_step'  : second_step,
                               'outcome'      : outcome})

    session_path = os.path.join(save_dir, session.file_name.split('.')[0])
    os.mkdir(session_path)
    trial_data.to_csv(os.path.join(session_path,'trial_data.csv'))
    np.save(os.path.join(session_path,'frame_times.npy'),frame_times)
    sparse.save_npz(os.path.join(session_path,'activity.npz'), sparse.csr_matrix(activity))

def save_experiment():
    '''Save the experiment data as .pkl files - greatly speeds up 
    subsequent loading of experiments.'''
    exp_img.save()

# ---------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------

exp_img = di.Experiment('2016-02-10-ACC_endoscope')

CNMFe_folder = os.path.join(exp_img.path, 'CNMFe_npy')

behaviour_calcium_filemap = {# behaviour_file     : calcium data folder
                             'm455-2016-07-14.txt': 'm455\\20160714_144121',
                             'm455-2016-07-15.txt': 'm455\\20160715_133449',
                             'm455-2016-07-16.txt': 'm455\\20160716_153215',
                             'm455-2016-07-17.txt': 'm455\\20160717_143955',
                             'm455-2016-07-19.txt': 'm455\\20160719_150748',
                             'm455-2016-07-20.txt': 'm455\\20160720_160238',
                             'm455-2016-07-21.txt': 'm455\\20160721_172946',

                             'm456-2016-07-14.txt': 'm456\\20160714_152852',
                             'm456-2016-07-15.txt': 'm456\\20160715_152006',
                             'm456-2016-07-16.txt': 'm456\\20160716_162425',
                             'm456-2016-07-17.txt': 'm456\\20160717_152805',
                             'm456-2016-07-19.txt': 'm456\\20160719_160704',
                             'm456-2016-07-20.txt': 'm456\\20160720_165338',
                             'm456-2016-07-21.txt': 'm456\\20160721_182115',

                             'm457-2016-07-15.txt': 'm457\\20160715_162453',
                             'm457-2016-07-16.txt': 'm457\\20160716_172303',
                             'm457-2016-07-17.txt': 'm457\\20160717_162408',
                             'm457-2016-07-20.txt': 'm457\\20160720_174318',

                             'm458-2016-07-14.txt': 'm458\\20160714_172343',
                             'm458-2016-07-16.txt': 'm458\\20160716_182123',
                             'm458-2016-07-17.txt': 'm458\\20160717_172319'}

if not any([hasattr(session, 'calcium_data') for session in exp_img.sessions]):
    # Load calcium imaging data.

    for session in exp_img.sessions:
        if session.file_name in behaviour_calcium_filemap.keys():
            data_folder = os.path.join(CNMFe_folder, 
                behaviour_calcium_filemap[session.file_name])
            ia.add_spike_data_to_session(session, data_folder)
        else:
            session.calcium_data = None

    img_sessions = [session for session in exp_img.sessions if session.calcium_data]

    median_latencies = pl.trial_timings_analysis(img_sessions, fig_no=False)

    ia.add_aligned_activity_to_sessions(img_sessions, median_latencies)

    md_session_1 = next(s for s in exp_img.sessions if s.file_name == 'm456-2016-07-20.txt')
    remove_missing_data_trials(md_session_1, 4565, 828, median_latencies)
    md_session_2 = next(s for s in exp_img.sessions if s.file_name == 'm457-2016-07-20.txt')
    remove_missing_data_trials(md_session_2, 4389, 1005, median_latencies)

else: # Calcium data already stored on sessions.
    img_sessions = [session for session in exp_img.sessions if session.calcium_data]
    median_latencies = pl.trial_timings_analysis(img_sessions, fig_no=False)

bad_sessions = ['m455-2016-07-14.txt']  # Sync pulses missing.

# Sessions with sufficient coverage of different trial types.
trial_cov_sessions = [session for session in img_sessions if min_coverage(session) >= 2 
                 and session.file_name not in bad_sessions]

# Sessions with sufficient coverage of different states of transiton probs.
trans_cng_sessions = [s for s in trial_cov_sessions if min([np.sum(
    s.blocks['trial_trans_state']), np.sum(~s.blocks['trial_trans_state'])]) >= 40]

# Sessions with sufficient coverage of different states of reward probs.
reward_cng_sessions = [s for s in trial_cov_sessions if 
    np.sum([np.sum(s.blocks['trial_rew_state'] == i) >=40 for i in range(3)]) >=2]

# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------

def figure_3():
    # Generate the panels for figure 3.
    ia.event_rate_histogram(img_sessions, fig_no='3C')
    ia.ave_activity_across_trial(img_sessions, fig_no='3D')
    ia.regression_analysis(trial_cov_sessions, fig_no='3E')
    ia.second_step_representation_evolution(trial_cov_sessions, fig_no='4FG')
    ia.outcomes_AB_correlation(trial_cov_sessions, fig_no='4H')

def figure_4():
    # Generate the panels for figure 4.
    ia.trajectory_analysis(trial_cov_sessions, fig_no='4AB')
    ia.trajectory_analysis(trial_cov_sessions, fig_no='4C', PCs=[0,3,4])
    ia.decoding_analysis(trial_cov_sessions, fig_no='4E')

def figure_5():
    # Generate the panels for figure 5.
    ia.regression_analysis(trans_cng_sessions , predictors='trans_block' , fig_no='5A')
    ia.regression_analysis(reward_cng_sessions, predictors='reward_block', fig_no='5B')

def figure_S5():
    # Plot neurons with strong selectivity to trial events, a subset
    # of these are shown in figure S5.
    for session in trial_cov_sessions:
        ia.plot_selective_neurons(session, fig_no=session.file_name)