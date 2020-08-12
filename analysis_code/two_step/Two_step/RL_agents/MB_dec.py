from ._RL_agent import *

class MB_dec(RL_agent):
    'Model based agent with forgetting.'

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'MB_dec'
        self.param_names  = ['alpV', 'decV', 'alpT' , 'decT', 'iTemp']
        self.param_ranges = ['unit', 'unit', 'unit' , 'unit', 'pos'  ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpV, decV, alpT, decT, iTemp = params_T[:5]   

        # Variables.
        V = np.zeros([2,session.n_trials])  # Second step TD values.
        T = np.zeros([2,session.n_trials])  # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            # Update action values and transition probabilities.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            V[r,i+1] = V[r,i] * (1.-decV)         # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o  # Second step TD update.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = self.apply_kernels(M, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, None, M)
        else:       return session_log_likelihood(choices, Q_net, iTemp)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        alpV, decV, alpT, decT, iTemp = params_T[:5]  

        # Variables.
        V = np.zeros([2,n_trials+1]) # Model free second step action values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        Q_net = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)
        self.init_kernels_sim(params_T)
        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            V[r,i+1] = V[r,i] * (1.-decV)        # Second step forgetting.
            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o # Second step TD update.

            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.
            Q_net = self.apply_kernels_sim(M, c, s)

        return choices, second_steps, outcomes