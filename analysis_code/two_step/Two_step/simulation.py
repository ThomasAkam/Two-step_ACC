import os
import numpy as np
import pylab as plt

from random import randint, random
from copy import deepcopy
from functools import partial

from . import plotting as pl 
from . import RL_agents as rl 
from . import logistic_regression as lr 
from . import model_fitting as mf
from . import model_plotting as mp
from . import parallel_processing as pp
from .data_import import Session

#------------------------------------------------------------------------------------
# Two-step task.
#------------------------------------------------------------------------------------

class Extended_two_step:
    '''Two step task with reversals in both which side is good and the transition matrix.'''
    def __init__(self, neutral_reward_probs = False):
        # Parameters
        self.norm_prob = 0.8 # Probability of normal transition.
        self.neutral_reward_probs = neutral_reward_probs

        if neutral_reward_probs: 
            self.reward_probs = np.array([[0.4, 0.4],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.4, 0.4]])
        else:
            self.reward_probs = np.array([[0.2, 0.8],  # Reward probabilities in each reward block type.
                                          [0.4, 0.4],
                                          [0.8, 0.2]])
        self.threshold = 0.75 
        self.tau = 8.  # Time constant of moving average.
        self.min_block_length = 40       # Minimum block length.
        self.min_trials_post_criterion = 20  # Number of trials after transition criterion reached before transtion occurs.
        self.mov_ave = _exp_mov_ave(tau = self.tau, init_value = 0.5)   # Moving average of agents choices.
        self.reset()

    def reset(self, n_trials = 1000, stim = False):
        self.transition_block = _with_prob(0.5)      # True for A blocks, false for B blocks.
        self.reward_block =     randint(0,2)        # 0 for left good, 1 for neutral, 2 for right good.
        self.block_trials = 0                       # Number of trials into current block.
        self.cur_trial = 0                          # Current trial number.
        self.trans_crit_reached = False             # True if transition criterion reached in current block.
        self.trials_post_criterion = 0              # Current number of trials past criterion.
        self.trial_number = 1                       # Current trial number.
        self.n_trials = n_trials                    # Session length.
        self.mov_ave.reset()
        self.end_session   = False
        self.stim_trials = _get_stim_trials(n_trials+1) if stim else None # Trials on which stimulation is simulated.
        self.blocks = {'start_trials'      : [0],
                       'end_trials'        : [],
                       'reward_states'     : [self.reward_block],      # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : [self.transition_block]}  # 1 for A blocks, 0 for B blocks.

    def trial(self, choice):
        # Update moving average.
        self.mov_ave.update(choice)
        second_step = int((choice == _with_prob(self.norm_prob))
                           == self.transition_block)
        self.block_trials += 1
        self.cur_trial += 1
        outcome = int(_with_prob(self.reward_probs[self.reward_block, second_step]))
        # Check for block transition.
        block_transition = False
        if self.trans_crit_reached:
            self.trials_post_criterion +=1
            if (self.trials_post_criterion >= self.min_trials_post_criterion) & \
               (self.block_trials >= self.min_block_length):
               block_transition = True
        else: # Check if transition criterion reached.
            if self.reward_block == 1 or self.neutral_reward_probs: #Neutral block
                if (self.block_trials > 20) & _with_prob(0.04):
                    self.trans_crit_reached = True
            elif self.transition_block ^ (self.reward_block == 2): # High is good option
                if self.mov_ave.ave > self.threshold:
                    self.trans_crit_reached = True
            else:                                                  # Low is good option
                if self.mov_ave.ave < (1. -self.threshold):
                    self.trans_crit_reached = True                

        if block_transition:
            self.block_trials = 0
            self.trials_post_criterion = 0
            self.trans_crit_reached = False
            old_rew_block = self.reward_block
            if old_rew_block == 1:                      # End of neutral block always transitions to one side 
                self.reward_block = _with_prob(0.5) * 2  # being good without reversal of transition probabilities.
            else: # End of block with one side good, 50% chance of change in transition probs.
                if _with_prob(0.5): #Reversal in transition probabilities.
                    self.transition_block = not self.transition_block
                    if _with_prob(0.5): # 50% chance of transition to neutral block.
                        self.reward_block = 1
                else: # No reversal in transition probabilities.
                    if _with_prob(0.5):
                        self.reward_block = 1 # Transition to neutral block.
                    else:
                        self.reward_block = 2 - old_rew_block # Invert reward probs.
            self.blocks['start_trials'].append(self.cur_trial)
            self.blocks['end_trials'].append(self.cur_trial)
            self.blocks['reward_states'].append(self.reward_block)
            self.blocks['transition_states'].append(self.transition_block)

        if self.cur_trial >= self.n_trials: #End of session.
            self.end_session = True
            self.blocks['end_trials'].append(self.cur_trial + 1)

            self.blocks['trial_trans_state'] = np.zeros(self.n_trials, dtype = bool) #Boolean array indication state of tranistion matrix for each trial.
            self.blocks['trial_rew_state']   = np.zeros(self.n_trials, dtype = int)

            for start_trial,end_trial, trans_state, reward_state in \
                    zip(self.blocks['start_trials'],self.blocks['end_trials'], \
                        self.blocks['transition_states'], self.blocks['reward_states']):
                self.blocks['trial_trans_state'][start_trial - 1:end_trial-1] = trans_state   
                self.blocks['trial_rew_state'][start_trial - 1:end_trial-1]  = reward_state   

        if self.stim_trials is not None:
            return (second_step, outcome, self.stim_trials[self.cur_trial])
        else:
            return (second_step, outcome)

class _exp_mov_ave:
    'Exponential moving average class.'
    def __init__(self, tau=None, init_value=0., alpha = None):
        if alpha is None: alpha = 1 - np.exp(-1/tau)
        self._alpha = alpha
        self._m = 1 - alpha
        self.init_value = init_value
        self.reset()

    def reset(self, init_value = None):
        if init_value:
            self.init_value = init_value
        self.ave = self.init_value

    def update(self, sample):
        self.ave = (self.ave*self._m) + (self._alpha*sample)


def _with_prob(prob):
    'return true / flase with specified probability .'
    return random() < prob

def _get_stim_trials(n_trials, min_ISI=2, mean_TPST=6):
    ''' Generate pattern of stim trials disributed with min_ISI + exponential disribution
    of trials between stim trials to give mean_TPST trials per stim trial.'''
    stim_prob = 1. / (mean_TPST - min_ISI) 
    trials_since_last_stim = 0
    stim_trials = np.zeros(n_trials, bool)
    for i in range(n_trials):
        trials_since_last_stim += 1
        if ((trials_since_last_stim > min_ISI) and _with_prob(stim_prob)): 
            trials_since_last_stim = 0
            stim_trials[i] = True
    return stim_trials

#------------------------------------------------------------------------------------
# Simulation.
#------------------------------------------------------------------------------------

class simulated_session(Session):
    '''Stores agent parameters and simulated data, supports plotting as for experimental
    session class.
    '''
    def __init__(self, agent, params_T, n_trials = 1000, task = Extended_two_step()):
        '''Simulate session with current agent and task parameters.'''
        self.param_names = agent.param_names
        self.true_params_T = params_T
        self.subject_ID = -1 
        try: # Not possible for e.g. unit range params_T with value 0 or 1.
            self.true_params_U = mf.transTU(params_T, agent.param_ranges)
        except Exception: 
            self.true_params_U = None
        self.n_trials = n_trials
        choices, second_steps, outcomes = agent.simulate(task, params_T, n_trials)
        
        self.trial_data = {'choices'      : choices,
                           'transitions'  : (choices == second_steps).astype(int),
                           'second_steps' : second_steps,
                           'outcomes'     : outcomes}

        if hasattr(task,'blocks'):
            self.blocks = deepcopy(task.blocks)

        if task.stim_trials is not None:
            self.stim_trials = task.stim_trials[:-1]
 

def sim_sessions_from_pop_fit(agent, population_fit, n_ses=10, n_trials=1000,
                              task=Extended_two_step()):
    '''Simulate sessions using parameter values drawn from the population distribution specified
    by population_fit. alternatively a dictionary of means and variances for each paramter can be
    specified.'''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    _sim_func_ = partial(_sim_func, population_fit, agent, n_trials, task)
    sessions = pp.map(_sim_func_, range(n_ses))
    return sessions

def _sim_func(population_fit, agent, n_trials, task, i):
        params_T = mf._sample_params_T(population_fit)
        return simulated_session(agent, params_T, n_trials, task)

def sim_ses_from_pop_means(agent, population_fit, n_trials = 10000, task = Extended_two_step()):
    '''Simulate a single session with agent parameter values set to the mean
    values of the population level distribution.
    '''
    assert population_fit['param_names'] == agent.param_names, 'Agent parameters do not match fit.'
    params_T = mf._trans_UT(population_fit['pop_dists']['means'], agent.param_ranges)
    return simulated_session(agent, params_T, n_trials, task)

#---------------------------------------------------------------------------------------
# Regression fit to RL simulation.
#---------------------------------------------------------------------------------------

def RL_agent_behaviour_comparison(sessions, n_ses=4000, n_trials=500, save_dir=None):
    agents = [rl.MF_dec(['bs','ck']),rl.MB_dec(['bs','ck']),
              rl.MFmoMF_MB_dec(['bs','rb','ec','mc'])]
    LR1_model    = lr.config_log_reg('standard')
    LR_lag_model = lr.config_log_reg('lagged')
    fits = {}
    for i, agent in enumerate(agents):
        print('Fitting agent: ' + agent.name)
        RL_fit = mf.fit_population(sessions, agent)
        print('Simulating data: ' + agent.name)
        sim_sessions = sim_sessions_from_pop_fit(agent, RL_fit, n_ses=n_ses, n_trials=n_trials)
        print('Analysing simulated data')
        LR1_fit = mf.fit_population(sim_sessions, LR1_model)
        LR_lag_fit = mf.fit_population(sim_sessions, LR_lag_model)
        fits[agent.name] = {'sim_sessions' : sim_sessions,
                            'RL_fit'       : RL_fit,
                            'LR1_fit'      : LR1_fit,
                            'LR_lag_fit'   : LR_lag_fit}
    _plot_sim_fits(fits, save_dir)

def _plot_sim_fits(fits, save_dir):
    for i, agent_name in enumerate(fits.keys()):
        agent_fits = fits[agent_name]
        # Stay probability plot.
        pl.stay_probability_analysis(agent_fits['sim_sessions'], fig_no=i+1, title=agent_name)
        plt.ylim(0.55, 0.75)
        if save_dir: plt.savefig(os.path.join(save_dir, agent_name + '_stay_probabilities.pdf'))
        # One trial back regression.
        mp.model_fit_plot(agent_fits['LR1_fit']   , fig_no=i+11 , scatter=False, title=agent_name)
        f = plt.figure(i+11)
        f.set_size_inches(3.7,2.3)
        if save_dir: plt.savefig(os.path.join(save_dir, agent_name + '_LR1.pdf'))
        f.set_size_inches(2.09,2.3)
        plt.xlim(4,7)
        plt.ylim(-0.05,0.35)
        if save_dir: plt.savefig(os.path.join(save_dir, agent_name + '_LR1_zoom.pdf'))
        # Lagged regression.
        mp.lagged_fit_plot(agent_fits['LR_lag_fit'], fig_no=i+101, sub_MAP=False, title=agent_name)
        plt.ylim(-0.1,0.8)
        if save_dir: plt.savefig(os.path.join(save_dir, agent_name + '_LR_lag.pdf'))
        plt.ylim(-0.07,0.35)
        if save_dir: plt.savefig(os.path.join(save_dir, agent_name + '_LR_lag_zoom.pdf'))