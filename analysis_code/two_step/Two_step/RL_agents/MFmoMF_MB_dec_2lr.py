from ._RL_agent import *

class MFmoMF_MB_dec_2lr(RL_agent):
    '''Mixture agent with forgetting and motor level model free, seperate learning rates
    at first and second step.'''

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'MFmoMF_MB_dec_2lr'
        self.param_names  = ['alpQ1', 'decQ1','alpQ2', 'decQ2', 'lbd' , 'act', 'alpT', 'decT',
                             'G_td', 'G_mb']
        self.param_ranges = ['unit']*8 + ['pos']*2
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))

        # Unpack parameters.
        alpQ1, decQ1, alpQ2, decQ2, lbd, act, alpT, decT, G_td, G_mb = params_T[:10]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        P = np.zeros([2,2,session.n_trials]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o, ps) in enumerate(zip(choices[:-1], second_steps, outcomes, prev_sec_steps)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ1) # First step forgetting.
            P[:,:,i+1] = P[:,:,i] * (1.-decQ1) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ2) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            Q[c,i+1] = (1.-alpQ1)*Q[c,i] + alpQ1*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            P[c,ps,i+1] = (1.-alpQ1)*P[c,ps,i] + alpQ1*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ2)*V[s,i] + alpQ2*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        Q_td = (1.-act)*Q+act*P # Mixture of action and target model free values.
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_td*Q_td + G_mb*M    # Mixture of model based and model free values.
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net)