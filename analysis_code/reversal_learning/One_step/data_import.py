import os
import pickle
import json
import numpy as np
from pathlib import Path
from collections import namedtuple
from . import plotting as pl

data_path = os.path.join(Path(__file__).resolve().parents[3],
                         'data', 'reversal_learning_task')

#--------------------------------------------------------------------
# Experiment class
#--------------------------------------------------------------------

class Experiment:
    def __init__(self, exp_name, rebuild_sessions = False):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from subjects in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.start_date = exp_name[:10]  

        self.path = os.path.join(data_path, exp_name)

        self.data_path = os.path.join(self.path, 'data')

        with open(os.path.join(self.path, 'info.json')) as f:
            experiment_info = json.load(f) 
        self.IDs = experiment_info['event_IDs']
        self.info = experiment_info['info']
        self.file_type = experiment_info['file_type']

        self.sessions = []
        
        if not rebuild_sessions:
            try:
                exp_file = open(os.path.join(self.path, 'sessions.pkl'),'rb')
                self.sessions = pickle.load(exp_file)
                exp_file.close()
                print('Saved sessions loaded from: sessions.pkl')
            except IOError:
               pass

        self.import_data()

        if rebuild_sessions:
            self.save()

    def save(self):
        'Save sessions from experiment.'
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as f:
            pickle.dump(self.sessions, f)


    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        with open(self.path + file_name + '.pkl', 'wb') as f:
            pickle.dump(item, f)

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        f = open(self.path + item_name + '.pkl', 'rb')
        out = pickle.load(f)
        f.close()
        return out

    def import_data(self):
        '''Load new sessions as session class instances.'''

        old_files = [session.file_name for session in self.sessions]
        files = os.listdir(self.data_path)
        new_files = [f for f in files if f[0] == 'm' and f not in old_files]

        if len(new_files) > 0:
            print('Loading new data files...')
            new_sessions = []
            for file_name in new_files:
                try:
                    new_sessions.append(Session(file_name,self.data_path, self.IDs, self.file_type))
                except AssertionError as error_message:
                    print('Unable to import file: ' + file_name)
                    print(error_message)

            self.sessions = self.sessions + new_sessions  

        self.dates = sorted(list(set([session.date for session in self.sessions])))

        for session in self.sessions: # Assign day numbers.
            session.day = self.dates.index(session.date) + 1

        self.n_subjects = len(set([session.subject_ID for session in self.sessions]))
        self.n_days = max([session.day for session in self.sessions]) 
        self.subject_IDs= list(set([s.subject_ID for s in self.sessions]))

        
    def get_sessions(self, sIDs, days = [], dates = []):
        '''Return list of sessions which match specified subject ID and day numbers
        or dates. All days or subjects can be selected with input 'all'.
        The last n days can be selected with days = -n .
        '''
        if days == 'all':
            days = np.arange(self.n_days) + 1
        elif isinstance(days, int):
            if days < 0:
                days = list(range(self.n_days + 1 + days, self.n_days + 1))
            else: days = [days]
        if sIDs == 'all':
            sIDs = self.subject_IDs
        elif isinstance(sIDs, int):
            sIDs = [sIDs]
        valid_sessions = [s for s in self.sessions if 
            (s.day in days or s.date in dates) and s.subject_ID in sIDs]
        if len(valid_sessions) == 1: 
            valid_sessions = valid_sessions[0] # Don't return list for single session.
        return valid_sessions                


#--------------------------------------------------------------------------------------------
# Session
#--------------------------------------------------------------------------------------------

Event = namedtuple('Event', ['time','name'])

class Session:
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
