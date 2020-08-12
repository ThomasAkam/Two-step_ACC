from ._RL_agent import *

class MFmoMF_dec(RL_agent):
    '''Model free agent forgetting and motor level model free action values.'''

    def __init__(self, kernels = ['bs', 'ck', 'rb']):
        self.name = 'MFmoMF_dec'
        self.param_names  = ['alpQ', 'decQ', 'lbd' , 'act', 'iTemp']
        self.param_ranges = ['unit']*4 + ['pos']*1
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))

        # Unpack parameters.
        alpQ, decQ, lbd, act, iTemp = params_T[:5]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        P = np.zeros([2,2,session.n_trials]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,session.n_trials]) # Second step TD values.

        for i, (c, s, o, ps) in enumerate(zip(choices[:-1], second_steps, outcomes, prev_sec_steps)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            P[:,:,i+1] = P[:,:,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            P[c,ps,i+1] = (1.-alpQ)*P[c,ps,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

        # Evaluate net action values and likelihood. 

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        Q_td = (1.-act)*Q+act*P # Mixture of action and target model free values.
        Q_net = self.apply_kernels(Q_td, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net, iTemp)


    def simulate(self, task, params_T, n_trials):

        alpQ, decQ, lbd, act, iTemp = params_T[:5]

        Q = np.zeros([2,n_trials+1]) # First step TD values.
        P = np.zeros([2,2,n_trials+1]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,n_trials+1]) # Second step TD values.

        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)
        self.init_kernels_sim(params_T)

        ps = 0 # Previous second step
        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            P[:,:,i+1] = P[:,:,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            P[c,ps,i+1] = (1.-alpQ)*P[c,ps,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            Q_td = (1.-act)*Q[:,i+1]+act*P[:,s,i+1]

            Q_net = self.apply_kernels_sim(Q_td, c, s)

            ps = s

        return choices, second_steps, outcomes