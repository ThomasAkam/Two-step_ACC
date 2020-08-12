import numpy as np
import os
from collections import namedtuple
from . import plotting as pl

Event = namedtuple('Event', ['time','name'])

class session:
    'Class containing data from a single session.'
    def __init__(self, file_name, data_path, IDs, file_type):
        print(file_name)
        self.file_name = file_name
        self.subject_ID =  int(file_name.split('-',1)[0][1:])
        self.date = file_name.split('-',1)[1].split('.')[0]
        self.IDs = IDs
        self.file_type = file_type

        # -------------------------------------------------------------------------------------------
        # Import data, extract timestamps and event codes.
        # -------------------------------------------------------------------------------------------

        with open(os.path.join(data_path, file_name), 'r') as data_file:
            split_lines = [line.strip().split(' ') for line in data_file]

        data_lines = [[int(i) for i in line] for line in split_lines if
                       len(line) > 1 and all([len(i)>0 and i[-1].isdigit() for i in line])]

        if self.file_type == 'pyControl_1': # Remove extra 0 from data lines to match format of other file types.
            [line.pop(1) for line in data_lines] # Remove zero element.
            
        event_lines       = [line for line in data_lines if line[1] in self.IDs.values()]
        block_start_lines = [line for line in data_lines if line[1] == -1] 
        stim_lines        = [line for line in data_lines if line[1] == -2] 

        #  Convert lines to numpy arrays of timestamps (in seconds) and event IDs.

        time_stamps = np.array([line[0] for line in event_lines]) / 1000.
        event_codes = np.array([line[1] for line in event_lines])

        ID2event = {v: k for k, v in self.IDs.items()} # Inverse of IDs dict.
        self.events = [Event(ts, ID2event[ec]) for (ts, ec) in zip(time_stamps, event_codes)]
        self.duration = time_stamps[-1]

        # -------------------------------------------------------------------------------------------
        # Make times dictionary: {'event_name': times_array}
        #--------------------------------------------------------------------------------------------

        self.times = {event_type: time_stamps[event_codes == self.IDs[event_type]]    
                            for event_type in self.IDs.keys()} # Dictionary of event names and times.

        self.times['choice'] = self.ordered_times(['left_select', 'right_select'])

        TsRwWp = [ev for ev in self.events if ev.name in 
                    ['trial_start', 'left_reward', 'right_reward', 'wait_for_poke_out']] + [Event(-1,'')]
        
        self.times['outcome'] =  np.array([TsRwWp[i+1].time for i,ev in enumerate(TsRwWp[:-1]) if ev.name == 'trial_start'])


        # -------------------------------------------------------------------------------------------
        # Make choice, outcome representation.
        #--------------------------------------------------------------------------------------------
        
        outcomes = [TsRwWp[i+1].name in ['left_reward', 'right_reward'] for
                        (i,ev) in enumerate(TsRwWp[:-1]) if ev.name == 'trial_start']

        choices = [ev.name == 'left_select' for ev in self.events if ev.name in ['left_select', 'right_select']]

        self.trial_data = {'choices'      : np.array(choices                , int), # 1 if high poke, 0 if low poke.
                   'outcomes'     : np.array(outcomes[:len(choices)], int)} # 1 if rewarded, 0 if not.

        self.n_trials = len(choices)
        self.rewards  = sum(outcomes)
        self.fraction_rewarded = self.rewards/self.n_trials

        #--------------------------------------------------------------------------------------------
        # Extract block information.
        #--------------------------------------------------------------------------------------------
   
        if len(block_start_lines) > 0:
            start_times = [line[0]/1000 for line in block_start_lines]
            reward_states = np.array([line[-1] for line in block_start_lines])
            start_trials = [np.searchsorted(self.times['trial_start'],st) for st in start_times]
            end_trials = start_trials[1:] + [self.n_trials]

            trial_rew_state   = np.zeros(self.n_trials, dtype = int)  # Integer array indicating state of rewared probabilities for each trial.
            for start_trial, end_trial, reward_state in zip(start_trials, end_trials, reward_states):
                trial_rew_state[start_trial:end_trial]   = reward_state   

            self.blocks = {'start_times'     : start_times,
                           'start_trials'    : start_trials,  # index of first trial of blocks, first trial of session is trial 0. 
                           'end_trials'      : end_trials,
                           'reward_states'   : reward_states, # 0 for right good, 1 for neutral, 2 for left good.
                           'trial_rew_state' : trial_rew_state}

        #--------------------------------------------------------------------------------------------
        # Extract stim information.    
        #--------------------------------------------------------------------------------------------

        # if len(stim_lines) > 0:
        #     stim_timestamps = (np.array([-1] + [line[0]  for line in stim_lines])) / 1000
        #     stim_state =                [0 ] + [line[-1] for line in stim_lines] # 1: onset, 0: offset.

        #     self.stim = {'on_times' : np.array([stim_timestamps[i]   for i,st in enumerate(stim_state)     if st == 1]),
        #                  'off_times': np.array([stim_timestamps[i+1] for i,st in enumerate(stim_state[1:]) if st == 0])}

        #     def _during_stim(event_timestamps): # Returns array of True/False for timestamps when stim is on/off.
        #         return np.array([stim_state[stim_timestamps.searchsorted(evt) - 1]
        #                          for evt in event_timestamps], bool)


        #     self.stim['trial_start'] = _during_stim(self.times['trial_start'])[:self.n_trials] # Trials which started during stim.  
        #     self.stim['choice'] = _during_stim(self.times['choice'] - 0.0005)[:self.n_trials]  # Trials where choice occured during stim. 
        #     self.stim['outcome'] = np.hstack([False, _during_stim(self.times['outcome'])[:self.n_trials-1]])  # Trials where outcome of previous trial was recieved during stim.
              

        if len(stim_lines) > 0:
            stim_timestamps = (np.array([line[ 0] for line in stim_lines])) / 1000
            stim_state =                [line[-1] for line in stim_lines] # 1: onset, 0: offset.

            self.times['stim_on' ] = np.array([stim_timestamps[i] for i,st in 
                                               enumerate(stim_state) if st == 1])
            self.times['stim_off'] = np.array([stim_timestamps[i] for i,st in 
                                               enumerate(stim_state) if st == 0])
            
            stim_trials = np.zeros(self.n_trials + 1, bool) 
            stim_trials[self.times['choice'].searchsorted(self.times['stim_on']
                        + 0.005)] = True
            self.stim_trials = stim_trials[:-1] # Boolean array indicating for each choice whether stim occured since previous choice.


    #------------------------------------------------------------------------------------------------

    def plot(self): pl.plot_session(self)


    def ordered_times(self,event_types):
        'Return a array of specified event times in order of occurence.'
        return np.sort(np.hstack([self.times[event_type] for event_type in event_types]))

    def get_IDs(self, event_list):
        return [self.IDs[val] for val in event_list]

    def unpack_CO(self, order = 'CO', dtype = int):
        'Return elements of CTSO dictionary in specified order and data type.'
        o_dict = {'C': 'choices', 'O': 'outcomes'}
        if dtype == int:
            return [self.trial_data[o_dict[i]] for i in order]
        else:
            return [self.trial_data[o_dict[i]].astype(dtype) for i in order]
