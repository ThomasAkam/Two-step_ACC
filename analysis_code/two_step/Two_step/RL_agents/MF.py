from ._RL_agent import *

class MF(RL_agent):
    'Model-free agent.'

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'MF'
        self.param_names  = ['alp', 'iTemp', 'lbd']
        self.param_ranges = ['unit', 'pos' , 'unit' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alp, iTemp, lbd = params_T[:3]   # Q value decay parameter.

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            # Update action values. 

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            Q[n,i+1] = Q[n,i] 
            V[r,i+1] = V[r,i]

            Q[c,i+1] = (1.-alp)*Q[c,i] + alp*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alp)*V[s,i] + alp*o  # Second step TD update.

        # Evaluate net action values and likelihood. 

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        if get_DVs: return self.get_DVs(session, params_T, Q_net, Q)
        else:       return session_log_likelihood(choices, Q_net, iTemp)


    def simulate(self, task, params_T, n_trials):

        alp, iTemp, lbd = params_T[:3] 

        Q = np.zeros([2,n_trials+1])  ## First step TD values.
        V = np.zeros([2,n_trials+1])  # Model free second step action values.
        Q_net = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)
        self.init_kernels_sim(params_T)
        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            # Update action values.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step. 

            Q[n,i+1] = Q[n,i]
            V[r,i+1] = V[r,i]

            Q[c,i+1] = (1.-alp)*Q[c,i] + alp*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alp)*V[s,i] + alp*o           # Second step TD update.

            Q_net = self.apply_kernels_sim(Q[:,i+1], c, s)

        return choices, second_steps, outcomes