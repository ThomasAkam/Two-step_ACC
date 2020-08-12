from pyControl.utility import *
import hardware_definition as hw

#-----------------------------------------------------------------------------------------
# Outcome generator.
#-----------------------------------------------------------------------------------------

class Outcome_generator:
    # Determines trial outcome and when block transitions occur.

    def __init__(self, verbose = False):
        # Parameters
        self.verbose = verbose # Display user readable output.

        self.reward_state = int(withprob(0.5)) * 2 # 0 for left good, 1 for neutral, 2 for right good.

        self.settings = {'first_session':  False,
                         'high_contrast':  False}

        self.threshold = 0.75 
        self.tau = 8.  # Time constant of moving average.
        self.min_block_length = 15    # Minimum block length.
        self.min_trials_post_criterion = 10  # Number of trials after transition criterion reached before transtion occurs.
        self.mean_neutral_block_length = 25
        self.first_session_rewards = 40 # Number of trials in first session which automatically recieve reward.

        self.mov_ave = exp_mov_ave(tau = self.tau, init_value = 0.5)   # Moving average of agents choices.

    def reset(self):

        if self.settings['high_contrast']:
            self.reward_probs = [[0.9, 0.1],  # High contrast between good and bad options.
                                 [0.5, 0.5],
                                 [0.1, 0.9]]    
        else:
            self.reward_probs =[[0.75, 0.25],  # Low contrast between good and bad options.
                                [0.5 , 0.5 ],
                                [0.25, 0.75]]
        self.mov_ave.reset()
        self.block_trials = 0                       # Number of trials into current block.
        self.trial_number = 0                       # Current trial number.
        self.reward_number = 0                      # Current number of rewards.
        self.block_number = 0                       # Current block number.
        self.trans_crit_reached = False             # True if transition criterion reached in current block.
        self.trials_post_criterion = 0              # Current number of trials past criterion.    
        self.nb_hazard_prob = 1 / (self.mean_neutral_block_length # Prob. of block transition on each trial
                                   - self.min_block_length)       # after min block length in neutral blocks.

        self.print_block_info()
        
    def get_outcome(self, choice):
        # Update moving average.
        self.mov_ave.update(choice)

        self.block_trials += 1
        self.trial_number += 1
        
        if self.verbose: self.print_state()

        if self.settings['first_session'] and self.trial_number <= self.first_session_rewards:
            outcome = True # First trials of first session are all rewarded.
        else:
            outcome = withprob(self.reward_probs[self.reward_state][choice])
        
        if outcome:
            self.reward_number += 1
        
        if self.trans_crit_reached:
            self.trials_post_criterion +=1
        elif (((self.reward_state == 0) and (self.mov_ave.value < (1. - self.threshold))) or # left is good option and left treshold crossed.
              ((self.reward_state == 2) and (self.mov_ave.value > self.threshold))):         # right is good option and right threshold crossed.
                self.trans_crit_reached = True
                print('# Transition criterion reached.')

        # Check for block transition.
        if ((self.block_trials >= self.min_block_length) and                     # Transitions only occur after min block length trials..
             ((self.reward_state == 1 and withprob(self.nb_hazard_prob)) or      # Neutral block: transitions are stochastic.
              (self.trials_post_criterion >= self.min_trials_post_criterion))):  # Non-neutral block: transitions occur fixed
             # Block transition                                                  # number of trials after threshold crossing.
            self.block_number += 1
            self.block_trials  = 0
            self.trials_post_criterion = 0
            self.trans_crit_reached = False
            possible_next_states = list(set([0,1,2]) - set([self.reward_state]))
            self.reward_state = possible_next_states[withprob(0.5)]    
            self.print_block_info()
        
        return outcome

    def print_state(self):
        print('# Trial number: {}, Reward number: {}, Moving ave: {}'            
             .format(self.trial_number, self.reward_number, self.mov_ave.value))

    def print_block_info(self):
        print('-1 {}'.format(self.reward_state))
        if self.verbose:
            print('# Reward block    : ' + {0:'0 - Left good', 
                                            1:'1 - Neutral', 
                                            2:'2 - Right good'}[self.reward_state])

    def print_summary(self):
        print('$ Total trials    : {}'.format(self.trial_number))
        print('$ Total rewards   : {}'.format(self.reward_number))
        print('$ Completed blocks: {}'.format(self.block_number))

#-----------------------------------------------------------------------------------------
# State machine
#-----------------------------------------------------------------------------------------

states = ['center_active',
          'inter_trial',
          'side_active',
          'left_select',
          'right_select',
          'left_reward',
          'right_reward',
          'wait_for_poke_out']

events = ['left_poke', 
          'left_poke_out',
          'right_poke',
          'right_poke_out',
          'high_poke',
          'high_poke_out',
          'session_timer',
          'state_timer']

initial_state = 'center_active'

# Variables.
v.outcome_generator = Outcome_generator(verbose = True)
v.inter_trial_interval = 1 * second
v.session_duration = 1.5 * hour
v.reward_delivery_durations = [80, 80] # ms

# Run start and stop behaviour.

def run_start():
    hw.houselight.on()
    set_timer('session_timer', v.session_duration)
    print('Reward sizes: ' + repr(v.reward_delivery_durations))
    v.outcome_generator.reset()

def run_end():
    hw.off() # Turn off all outputs.
    v.outcome_generator.print_summary()

# State & event dependent behaviour.   

def center_active(event):
    if event == 'entry':
        hw.center_poke.LED.on() 
    elif event == 'exit':
        hw.center_poke.LED.off()
    elif event == 'high_poke':
        goto_state('side_active')
        
def side_active(event):
    if event == 'entry':
        hw.left_poke.LED.on()
        hw.right_poke.LED.on()
    elif event == 'exit':
        hw.left_poke.LED.off()
        hw.right_poke.LED.off()
    elif event == 'left_poke':
        goto_state('left_select')
    elif event == 'right_poke':
        goto_state('right_select')

def left_select(event):
    if event == 'entry':
        set_timer('state_timer', 5 * ms)
    elif event == 'state_timer':
        if v.outcome_generator.get_outcome(False):
            goto_state('left_reward')
        else:
            goto_state('wait_for_poke_out')

def right_select(event):
    if event == 'entry':
        set_timer('state_timer', 5 * ms)
    elif event == 'state_timer':
        if v.outcome_generator.get_outcome(True):
            goto_state('right_reward')
        else:
            goto_state('wait_for_poke_out')

def left_reward(event):
    if event == 'entry':
        hw.left_poke.SOL.on()
        set_timer('state_timer', v.reward_delivery_durations[0] * ms)
    elif event == 'exit':
        hw.left_poke.SOL.off()
    elif event == 'state_timer':
        goto_state('wait_for_poke_out')

def right_reward(event):
    if event == 'entry':
        hw.right_poke.SOL.on()
        set_timer('state_timer', v.reward_delivery_durations[1] * ms)
    elif event == 'exit':
        hw.right_poke.SOL.off()
    elif event == 'state_timer':
        goto_state('wait_for_poke_out')     
    
def wait_for_poke_out(event):
    if event == 'entry':
        if not (hw.left_poke.value() or hw.right_poke.value()):
            goto_state('center_active') # Subject already left poke.
    elif event in ['left_poke_out', 'right_poke_out']:
        goto_state('inter_trial')

def inter_trial(event):
    if event == 'entry':
        set_timer('state_timer', v.inter_trial_interval)
    elif event == 'state_timer':
        goto_state('center_active')

def all_states(event):
    if event == 'session_timer': # End session
        stop_framework()