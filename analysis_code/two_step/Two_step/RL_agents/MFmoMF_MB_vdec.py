from ._RL_agent import *

class MFmoMF_MB_vdec(RL_agent):
    '''Mixture agent with value forgetting and motor level model free.'''

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'MFmoMF_MB_vdec'
        self.param_names  = ['alpQ', 'decQ', 'lbd','alpT', 'G_td','G_tdm','G_mb']
        self.param_ranges = ['unit']*4 + ['pos']*3
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))

        # Unpack parameters.
        alpQ, decQ, lbd, alpT, G_td, G_tdm, G_mb = params_T[:7]

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

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            P[:,:,i+1] = P[:,:,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i]  

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            P[c,ps,i+1] = (1.-alpQ)*P[c,ps,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_td*Q + G_tdm*P + G_mb*M    # Mixture of model based and model free values.
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q, M, P)
        else:       return session_log_likelihood(choices, Q_net)