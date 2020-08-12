from ._RL_agent import *

class best_RL_agent(RL_agent):
    '''Mixture agent with decays, optimised for speed by removing flexibility
    about which kernels are used.'''

    def __init__(self):
        self.name = 'best_RL_agent'
        self.param_names  = ['alpQ', 'decQ', 'lbd', 'act', 'alpT', 'decT', 'alpEC', 'alpMC',
                             'G_td', 'G_mb', 'bs', 'rb', 'ec', 'mc']

        self.param_ranges = ['unit']*8 + ['pos']*2 + ['unc']*4
        RL_agent.__init__(self, kernels = None)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))

        # Unpack parameters.
        alpQ, decQ, lbd, act, alpT, decT, alpCK, alpSK, G_td, G_mb, bs, sk, ec, sc = params_T

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        P = np.zeros([2,2,session.n_trials]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(session.n_trials) # Choice kernel.
        D = np.zeros([2,session.n_trials]) # Previous side dependent choice kernel.

        for i, (c, s, o, ps) in enumerate(zip(choices[:-1], second_steps, outcomes, prev_sec_steps)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.
            pr = 1 - ps # State not reached at second step on previous trial.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            P[:,:,i+1] = P[:,:,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            D[pr,i+1] = D[pr,i]

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            P[c,ps,i+1] = (1.-alpQ)*P[c,ps,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            C[i+1] = (1.-alpCK)*C[i] + alpCK*(c-0.5)
            D[ps,i+1] = (1.-alpSK)*D[ps,i] + alpSK*(c-0.5)

        # Evaluate net action values and likelihood. 

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        Q_td = (1.-act)*Q+act*P # Mixture of action and target model free values.
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = G_td*Q_td + G_mb*M  # Mixture of model based and model free values.
        D = D[prev_sec_steps,np.arange(session.n_trials)] 
        Q_net[1,:] += bs + sk*(prev_sec_steps-0.5) + ec*C + sc*D

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q, M)
        else:       return session_log_likelihood(choices, Q_net)