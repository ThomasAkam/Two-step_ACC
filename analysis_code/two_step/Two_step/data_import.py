import os
import json
import pickle
import numpy as np
from pathlib import Path
from collections import namedtuple
from . import plotting as pl

data_path = os.path.join(Path(__file__).resolve().parents[3],
                         'data', 'two_step_task')

#--------------------------------------------------------------------
# Experiment class
#--------------------------------------------------------------------

class Experiment:
    def __init__(self, exp_name, rebuild_sessions=False, missing_data_warning=False, exp_path=None):
        '''
        Instantiate an experiment object for specified group number.  Tries to load previously 
        saved sessions,  then loads sessions from data folder if they were not in
        the list of loaded sessions and are from subjects in the group.  rebuild sessions argument
        forces the sessions to be created directly from the data files rather than loaded.
        '''

        self.name = exp_name
        self.start_date = exp_name[:10]  

        if exp_path:
            self.path = exp_path
        else:
            self.path = os.path.join(data_path, exp_name)

        self.data_path = os.path.join(self.path, 'data')

        with open(os.path.join(self.path, 'info.json')) as f:
            experiment_info = json.load(f) 
        self.IDs = experiment_info['event_IDs']
        self.info = experiment_info['info']
        self.file_type = experiment_info['file_type']

        assert self.file_type in ['Arduino', 'pyControl_1', 'pyControl_2'], 'Invalid file type.'
        
        required_IDs = {'low_poke', 'high_poke', 'left_poke', 'right_poke', 'trial_start', 'left_reward',
                        'right_reward','left_active','right_active','ITI_start', 'wait_for_poke_out'}

        assert required_IDs <= self.IDs.keys(), 'IDs dictionary missing keys: ' + repr(list(required_IDs - set(self.IDs.keys())))

        self.sessions = []
        
        if not rebuild_sessions:
            try:
                with open(os.path.join(self.path, 'sessions.pkl'),'rb') as f:
                    self.sessions = pickle.load(f)
                print('Saved sessions loaded from: sessions.pkl')
            except IOError:
               pass

        self.import_data()

        if missing_data_warning: self.check_for_missing_data_files()
        if rebuild_sessions: self.save()

    def save(self):
        'Save sessions from experiment.'
        with open(os.path.join(self.path, 'sessions.pkl'),'wb') as f:
            pickle.dump(self.sessions, f)

    def save_item(self, item, file_name):
        'save an item to experiment folder using pickle.'
        with open(os.path.join(self.path, file_name + '.pkl'), 'wb') as f:
            pickle.dump(item, f)

    def load_item(self, item_name):
        'Unpickle and return specified item from experiment folder.'
        with open(os.path.join(self.path, item_name + '.pkl'), 'rb') as f:
                return pickle.load(f)

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
        or dates. 
        Select all days or subjects with:  days = 'all', sIDs = 'all'
        Select the last n days with     :  days = -n. 
        Select days from n to end with  :  days = [n, -1]
        '''
        if days == 'all':
            days = range(self.n_days + 1)
        elif isinstance(days, int):
            if days < 0:
                days = list(range(self.n_days + 1 + days, self.n_days + 1))
            else: days = [days]
        elif len(days) == 2 and days[-1] == -1:
            days = range(days[0], self.n_days + 1)
        if sIDs == 'all':
            sIDs = self.subject_IDs
        elif isinstance(sIDs, int):
            sIDs = [sIDs]
        valid_sessions = [s for s in self.sessions if 
            (s.day in days or s.date in dates) and s.subject_ID in sIDs]
        if len(valid_sessions) == 0:
            return None
        elif len(valid_sessions) == 1: 
            return valid_sessions[0] # Don't return list for single session.
        else:
            return valid_sessions                
                 
    def print_CSO_to_file(self, sIDs, days, file_name = 'sessions_CSO.txt'):
        f = open(file_name, 'w')
        sessions = self.get_sessions(sIDs, days)
        total_trials = sum([s.n_trials for s in sessions])
        f.write('Data from experiment "{}", {} sessions, {} trials.\n' 
                'Each trial is indicated by 3 numbers:\n'
                'First column : Choice      (1 = high poke, 0 = low poke)\n'
                'Second column: Second step (1 = left poke, 0 = right poke)\n'
                'Third column : Outcome     (1 = rewarded, 0 = not rewarded)\n'
                .format(self.name, len(sessions), total_trials))
        for (i,s) in enumerate(sessions):
            f.write('''\nSession: {0}, subject ID: {1}, date: {2}\n\n'''\
                    .format(i + 1, s.subject_ID, s.date))
            for c,sl,o in zip(s.trial_data['choices'], s.trial_data['second_links'], s.trial_data['outcomes']):
                f.write('{0:1d} {1:1d} {2:1d}\n'.format(c, sl, o))
        f.close()

    def check_for_missing_data_files(self):
        '''Identifies any days where there are data files for only a subset of subjects
        and reports missing sessions. Called on instantiation of experiment as a check 
        for any problems in the date transfer pipeline from rig to analysis.
        '''
        dates = sorted(set([s.date for s in self.sessions]))
        sessions_per_date = [len(self.get_sessions('all', dates = date)) for date in dates]
        if min(sessions_per_date) < self.n_subjects:
            print('Possible missing data files:')
            for date, n_sessions in zip(dates, sessions_per_date):
                if n_sessions < self.n_subjects:
                    subjects_run = [s.subject_ID for s in self.get_sessions('all', dates = date)]
                    subjects_not_run = set(self.subject_IDs) - set(subjects_run)
                    for sID in subjects_not_run:
                        print(('Date: ' + date + ' sID: {}'.format(sID)))

    def concatenate_sessions(self, days):
        ''' For each subject, concatinate sessions for specified days
        into single long sessions.
        '''
        concatenated_sessions = []
        for sID in self.subject_IDs:
            subject_sessions = self.get_sessions(sID, days)
            concatenated_sessions.append(ss.concatenated_session(subject_sessions))
        return concatenated_sessions

    # Plotting.

    def plot_day(self, day = -1): pl.plot_day(self, day)
    def plot_subject(self, sID, day_range = [0, np.inf]): pl.plot_subject(self, sID, day_range) 

# --------------------------------------------------------------
# Session class
# --------------------------------------------------------------

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

        # Import data, extract timestamps and event codes.
       
        with open(os.path.join(data_path, file_name), 'r') as data_file:
            split_lines = [line.strip().split(' ') for line in data_file]

        data_lines = [[int(i) for i in line] for line in split_lines if
                       len(line) > 1 and all([len(i)>0 and i[-1].isdigit() for i in line])]

        if self.file_type == 'pyControl_1': # Remove extra 0 from data lines to match format of other file types.
            [line.pop(1) for line in data_lines]
            
        event_lines       = [line for line in data_lines if line[1] in self.IDs.values()]
        block_start_lines = [line for line in data_lines if line[1] == -1] 
        stim_lines        = [line for line in data_lines if line[1] == -2] 

        #  Convert lines to numpy arrays of timestamps (in seconds) and event IDs.

        raw_time_stamps = np.array([line[0] for line in event_lines])
        event_codes     = np.array([line[1] for line in event_lines])

        if self.file_type == 'Arduino':

            if 'start_stop' in list(IDs.keys()):
                start_stop_inds = np.where(event_codes == IDs['start_stop'])[0]
                start_ind = start_stop_inds[0]
                stop_ind  = start_stop_inds[1]
            else:
                start_ind = np.where(event_codes == IDs['session_start'])[0][0]
                stop_ind  = np.where(event_codes == IDs['session_stop' ])[0][0]

            raw_time_stamps = raw_time_stamps[start_ind:stop_ind]
            time_stamps = (raw_time_stamps - raw_time_stamps[0])/1000
            event_codes = event_codes[start_ind:stop_ind]

        else: # File generated by pyControl behaviour system.
            time_stamps = raw_time_stamps / 1000.
            event_codes = event_codes

        ID2event = {v: k for k, v in self.IDs.items()} # Inverse of IDs dict.
        events = [Event(ts, ID2event[ec]) for (ts, ec) in zip(time_stamps, event_codes)]
        self.duration = time_stamps[-1]

        # Make times dictionary: {'event_name': times_array}
        
        self.times = {event_type: time_stamps[event_codes == self.IDs[event_type]]    
                      for event_type in self.IDs.keys()} # Dictionary of event names and times.

        self.times['choice'] = self.ordered_times(['left_active', 'right_active'])

        TsRwWp = [ev for ev in events if ev.name in 
                  ['trial_start', 'left_reward',
                   'right_reward', 'wait_for_poke_out']] + [Event(-1,'')]
        
        self.times['outcome'] =  np.array([TsRwWp[i+1].time for i,ev in 
                                 enumerate(TsRwWp[:-1]) if ev.name == 'trial_start'])

        # Make dictionary of choices, transitions, second steps and outcomes on each trial.
        
        second_steps = [ev.name == 'left_active' for ev in events
                        if ev.name in ['left_active', 'right_active']]
        
        outcomes = [TsRwWp[i+1].name in ['left_reward', 'right_reward'] for
                    (i,ev) in enumerate(TsRwWp[:-1]) if ev.name == 'trial_start']

        ChSs = [ev.name for ev in events if ev.name in 
                ['high_poke', 'low_poke', 'left_active', 'right_active']]

        choices = [ChSs[i] == 'high_poke' for (i,ev) in enumerate(ChSs[1:])
                   if ev in ['left_active', 'right_active']]

        self.trial_data = {'choices'     : np.array(choices                    , int), # 1 if high poke, 0 if low poke.
                           'second_steps': np.array(second_steps[:len(choices)], int), # 1 if left, 0 if right.
                           'outcomes'    : np.array(outcomes[:len(choices)]    , int)} # 1 if rewarded, 0 if not.

        self.trial_data['transitions'] = ((self.trial_data['choices'] ==               # 1 if high --> left or low --> right,
                                         self.trial_data['second_steps']).astype(int)) # 0 if high --> right or low --> left.
        
        self.n_trials = len(choices)
        self.rewards  = sum(outcomes)
        self.fraction_rewarded = self.rewards/self.n_trials

        # Extract block information.
        
        if len(block_start_lines) > 0:
            start_times = [(line[0]- raw_time_stamps[0])/1000 
                           for line in block_start_lines]
            start_trials = [np.searchsorted(self.times['trial_start'],st)
                            for st in start_times]
            reward_states     = np.array([line[-2] for line in block_start_lines])
            transition_states = np.array([line[-1] for line in block_start_lines])
            end_trials = start_trials[1:] + [self.n_trials]

            if self.file_type == 'Arduino': 
                reward_states   = 2 - reward_states # Arduino data used oposite coding of reward state.
                start_trials[0] = 0  # Timestamp for first block info follows first trial start in Arduino data files.

            trial_trans_state = np.zeros(self.n_trials, dtype = bool) # Boolean array indicating state of tranistion matrix for each trial.
            trial_rew_state   = np.zeros(self.n_trials, dtype = int)  # Integer array indicating state of rewared probabilities for each trial.
            end_trials = start_trials[1:] + [self.n_trials]
            for start_trial,end_trial, trans_state, reward_state in \
                    zip(start_trials, end_trials, transition_states, reward_states):
                trial_trans_state[start_trial:end_trial] = trans_state   
                trial_rew_state[start_trial:end_trial]   = reward_state   

            self.blocks = {'start_times'       : start_times,
                           'start_trials'      : start_trials, # index of first trial of blocks, first trial of session is trial 0. 
                           'end_trials'        : end_trials,
                           'reward_states'     : reward_states,      # 0 for left good, 1 for neutral, 2 for right good.
                           'transition_states' : transition_states,  # 1 for high --> left common, 0 for high --> right common.
                           'trial_trans_state' : trial_trans_state,
                           'trial_rew_state'   : trial_rew_state}   

        # Extract stim information.    
        
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

    def select_trials(self, selection_type, select_n = 20, first_n_mins = False,
                      block_type = 'all'):
        ''' Select specific trials for analysis.  

        The first selection step is specified by selection_type:

        'end' : Only final select_n trials of each block are selected.

        'xtr' : Exlude select_n trials following transition reversal.

        'xrr' : Exlude select_n trials following reward probability changes.

        'all' : All trials are included.

        The first_n_mins argument can be used to select only trials occuring within
        a specified number of minutes of the session start.

        The block_type argument allows additional selection for only 'neutral' or 'non_neutral' blocks.
        '''

        assert selection_type in ['end', 'xtr', 'all', 'xrr'], 'Invalid trial select type.'

        if selection_type == 'xtr': # Select all trials except select_n following transition reversal.
            trials_to_use = np.ones(self.n_trials, dtype = bool)
            trans_change = np.hstack((
                False, ~np.equal(self.blocks['transition_states'][:-1],
                                 self.blocks['transition_states'][1:])))
            start_trials = (self.blocks['start_trials'] + 
                            [self.blocks['end_trials'][-1] + select_n])
            for i in range(len(trans_change)):
                if trans_change[i]:
                    trials_to_use[start_trials[i]:start_trials[i] + select_n] = False

        elif selection_type == 'end': # Select only select_n trials before block transitions.
            trials_to_use = np.zeros(self.n_trials, dtype = bool)
            for b in self.blocks['start_trials'][1:]:
                trials_to_use[b - 1 - select_n:b -1] = True

        elif selection_type == 'xrr': # Exlude select_n trials after reward prob changes.
            trials_to_use = np.ones(self.n_trials, dtype = bool)
            rew_change = np.hstack((False, self.blocks['reward_states'][:-1] !=
                self.blocks['reward_states'][1:]))
            rew_change_trials = np.array(self.blocks['start_trials'])[rew_change]
            for r in rew_change_trials:
                trials_to_use[r:r+select_n] = False

        elif selection_type == 'all': # Use all trials.
            trials_to_use = np.ones(self.n_trials, dtype = bool)
            
        if first_n_mins:  #  Restrict analysed trials to only first n minutes. 
            time_selection = self.times['trial_start'][:self.n_trials] < (60*first_n_mins)
            trials_to_use = trials_to_use & time_selection

        if not block_type == 'all': #  Restrict analysed trials to blocks of certain types.
            if block_type == 'neutral':       # Include trials only from neutral blocks.
                block_selection = self.blocks['trial_rew_state'] == 1
            elif block_type == 'non_neutral': # Include trials only from non-neutral blocks.
                block_selection = self.blocks['trial_rew_state'] != 1
            trials_to_use = trials_to_use & block_selection

        return trials_to_use

    def plot(self, fig_no = 1): pl.plot_session(self, fig_no)

    def ordered_times(self,event_types):
        'Return a array of specified event times in order of occurence.'
        return np.sort(np.hstack([self.times[event_type] for event_type in event_types]))

    def get_IDs(self, event_list):
        return [self.IDs[val] for val in event_list]

    def unpack_trial_data(self, order = 'CTSO', dtype = int):
        'Return elements of trial_data dictionary in specified order and data type.'
        o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
        if dtype == int:
            return [self.trial_data[o_dict[i]] for i in order]
        else:
            return [self.trial_data[o_dict[i]].astype(dtype) for i in order]
