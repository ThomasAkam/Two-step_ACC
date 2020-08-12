from ._RL_agent import *
from . stim_agents import _get_stim_data

from .. model_fitting import _trans_UT, _trans_TU

class stim_diff_RL_agent(RL_agent):
    '''Mixture agent with decays, optimised for speed by removing flexibility
    about which kernels are used.'''

    def __init__(self):
        self.name = 'best_RL_agent'
        self.param_names  = ['alpQ', 'decQ', 'lbd', 'alpT', 'decT', 'alpEC', 'alpMC',
                             'G_td', 'G_tdm', 'G_mb', 'bs', 'rb', 'ec', 'mc',
                             'alpQ_s','decQ_s', 'lbd_s', 'alpT_s', 'decT_s', 'alpEC_s', 'alpMC_s',
                             'G_td_s', 'G_tdm_s', 'G_mb_s', 'bs_s', 'rb_s', 'ec_s', 'mc_s' ]


        self.param_ranges = ['unit']*7 + ['pos']*3 + ['unc']*18
        RL_agent.__init__(self, kernels=None)

    @jit
    def session_likelihood(self, session, params_T, get_DVs = False):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.

        params_n = params_T[:14]
        params_s = _trans_UT(_trans_TU(params_T[:14], self.param_ranges[:14])+params_T[14:], self.param_ranges[:14])

        G_td_n, G_tdm_n, G_mb_n, bs_n, rb_n, ec_n, mc_n = params_n[7:]
        G_td_s, G_tdm_s, G_mb_s, bs_s, rb_s, ec_s, mc_s = params_s[7:]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        P = np.zeros([2,2,session.n_trials]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(session.n_trials) # Choice kernel.
        D = np.zeros([2,session.n_trials]) # Previous side dependent choice kernel.

        for i, (c, s, o, ps, su) in enumerate(zip(choices[:-1], second_steps, outcomes,
                                                  prev_sec_steps, s_updates)): # loop over trials.

            if su: # Stim update
                alpQ, decQ, lbd, alpT, decT, alpEC, alpMC = params_s[:7]
            else:
                alpQ, decQ, lbd, alpT, decT, alpEC, alpMC = params_n[:7]

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

            C[i+1] = (1.-alpEC)*C[i] + alpEC*(c-0.5)
            D[ps,i+1] = (1.-alpMC)*D[ps,i] + alpMC*(c-0.5)

        # Evaluate net action values and likelihood. 

        G_tdm = G_tdm_n*n_choices + G_tdm_s*s_choices
        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        rb = rb_n*n_choices + rb_s*s_choices
        ec = ec_n*n_choices + ec_s*s_choices
        mc = mc_n*n_choices + mc_s*s_choices

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q + G_tdm*P + G_mb*M  # Mixture of model based and model free values.
        D = D[prev_sec_steps,np.arange(session.n_trials)] 
        Q_net[1,:] += bs + rb*(prev_sec_steps-0.5) + ec*C + mc*D

        return session_log_likelihood(choices, Q_net)