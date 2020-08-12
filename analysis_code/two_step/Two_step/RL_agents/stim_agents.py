import numpy as np
from numba import jit
from ._RL_agent import session_log_likelihood, choose, softmax
from .. import utility as ut

#------------------------------------------------------------------------------------
# _Stim_agent base class
#------------------------------------------------------------------------------------

class _Stim_agent():
    '''Base class for agents whose parameters can be configured to take seperate values
    on stim and non-stim trials. The stim_params argument is used to specify which 
    parameters take seperate values on stim and non-stim trials.'''

    def __init__(self, stim_params = 'all', kernels = True): 

        self.name += '_' + ''.join(stim_params)

        if kernels:
            self.bp_names  += [ 'bs', 'ck' , 'rb']
            self.bp_ranges += [ 'unc' , 'unc', 'unc']

        if stim_params == 'all':
            stim_params = self.bp_names   

        assert all([p in self.bp_names for p in stim_params]), 'Invalid stim_param name.'  

        self.param_names, self.param_ranges, self.np_ind, self.sp_ind  = ([],[],[],[])

        for i, param in enumerate(self.bp_names):
            self.param_names.append(param)
            self.np_ind.append(len(self.param_names) - 1)
            self.param_ranges.append(self.bp_ranges[i])
            if param in stim_params:
                self.param_names.append(param + '_s')
                self.param_ranges.append(self.bp_ranges[i])
            self.sp_ind.append(len(self.param_names) - 1)

        self.n_params = len(self.param_names)
        self.calculates_gradient = False
        self.type = 'RL_stim'

    def apply_kernels(self, Q_pre, choices, second_steps,
                      s_choices, n_choices, params_T):
        'Apply modifier to Q values due to kernels.'             
        p_names = self.param_names
        bias_n = params_T[p_names.index('bs')]   if 'bs'   in p_names else 0.
        ck_n   = params_T[p_names.index('ck')]     if 'ck'     in p_names else 0.
        ssk_n  = params_T[p_names.index('rb')]    if 'rb'    in p_names else 0.
        bias_s = params_T[p_names.index('bs_s')] if 'bs_s' in p_names else bias_n
        ck_s   = params_T[p_names.index('ck_s')]   if 'ck_s'   in p_names else ck_n
        ssk_s  = params_T[p_names.index('sk_s')]  if 'sk_s'  in p_names else ssk_n
        kernel_Qs = np.zeros((2,len(choices)))
        kernel_Qs[0,:] -= s_choices * bias_s + n_choices * bias_n
        kernel_Qs[0,1:] += ((0.5 - choices[:-1]) * 
                            (s_choices * ck_s + n_choices * ck_n)[1:])
        kernel_Qs[0,1:] += ((0.5 - second_steps[:-1]) * 
                            (s_choices * ssk_s + n_choices * ssk_n)[1:])
        return Q_pre + kernel_Qs

def _get_stim_data(session):
    s_choices = session.stim_trials.astype(float)
    n_choices = 1. - s_choices
    s_updates = session.stim_trials[1:]
    return s_choices, n_choices, s_updates

#------------------------------------------------------------------------------------
# Stim agents
#------------------------------------------------------------------------------------

class Stim_agent_1lr(_Stim_agent):
    
    def __init__(self, stim_params = 'all'):      
        self.name = 'Stim_agent_1lr'
        self.bp_names  = ['alpQ', 'decQ', 'lbd' , 'alpT', 'decT', 'G_td', 'G_mb']
        self.bp_ranges = ['unit', 'unit', 'unit', 'unit', 'unit', 'pos' , 'pos' ]
        _Stim_agent.__init__(self, stim_params)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Parameters
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])
        G_td_n, G_mb_n = params_n[5:7]
        G_td_s, G_mb_s = params_s[5:7]

        #Variables.
        Q = np.zeros([2,session.n_trials])# First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T  = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o, su) in enumerate(zip(choices[:-1], second_steps, outcomes, s_updates)): # loop over trials.

            if su: # Stim update
                alpQ, decQ, lbd, alpT, decT = params_s[:5]   
            else:
                alpQ, decQ, lbd, alpT, decT = params_n[:5]

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)* V[0,:] # Model based action values.
        Q_net = (G_td_n*Q + G_mb_n*M)*n_choices + (G_td_s*Q + G_mb_s*M)*s_choices
        Q_net = self.apply_kernels(Q_net, choices, second_steps, s_choices, n_choices, params_T)
        return session_log_likelihood(choices, Q_net)

#------------------------------------------------------------------------------------

class Stim_agent_k_only(_Stim_agent):

    def __init__(self, stim_params = 'all'):      
        self.name = 'Stim_agent_k_only'
        self.bp_names  = []
        self.bp_ranges = []
        _Stim_agent.__init__(self, stim_params)

    @jit
    def session_likelihood(self, session, params_T):
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        s_choices, n_choices, s_updates = _get_stim_data(session)
        Q_net = self.apply_kernels(np.zeros([2,session.n_trials]), choices, second_steps, s_choices, n_choices, params_T)
        return session_log_likelihood(choices, Q_net)

#------------------------------------------------------------------------------------

class Log_reg_emulation(_Stim_agent):

    def __init__(self, stim_params = 'all'):      
        self.name = 'Stim_agent_k_only'
        self.bp_names  = ['out', 'trans', 'out_x_trans']
        self.bp_ranges = ['unc', 'unc'  ,  'unc']
        _Stim_agent.__init__(self, stim_params)

    @jit
    def session_likelihood(self, session, params_T):
        choices, second_steps, transitions, outcomes = session.unpack_trial_data('CSTO')
        s_choices, n_choices, s_updates = _get_stim_data(session)
        trial_trans_state = session.blocks['trial_trans_state'].astype(int)

        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        Q_net = np.zeros([2,session.n_trials])

        for i, (c, s, t, o, su) in enumerate(zip(choices[:-1], second_steps, transitions, outcomes, s_updates)): # loop over trials.
            
            if su: # Stim update
                out, trans, out_x_trans = params_s[:3]   
            else:
                out, trans, out_x_trans = params_n[:3]

            common_transition = int(t == trial_trans_state[i])
            interaction = common_transition == o

            Q_net[c,i+1] += (o-0.5)*out
            Q_net[c,i+1] += (common_transition-0.5)*trans
            Q_net[c,i+1] += (interaction-0.5)*out_x_trans

        Q_net = self.apply_kernels(Q_net, choices, second_steps, s_choices, n_choices, params_T)
        return session_log_likelihood(choices, Q_net)

#------------------------------------------------------------------------------------

class stim_agent_ec_sc_mMF(_Stim_agent):
    '''Mixture agent with decays, optimised for speed by removing flexibility
    about which kernels are used.'''

    def __init__(self, stim_params = 'all'):
        self.name = 'stim_agent_ec_sc_mMF'
        self.bp_names  = ['alpQ', 'decQ', 'lbd', 'act', 'alpT', 'decT', 'alpEC', 'alpMC', 'k',
                          'iTemp', 'bs', 'rb', 'ec', 'mc']
        self.bp_ranges = ['unit']*9 + ['pos']*1 + ['unc']*4
        _Stim_agent.__init__(self, stim_params, kernels = False)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        k_n, iTemp_n, bs_n, sk_n, ec_n, sc_n = params_n[8:]
        k_s, iTemp_s, bs_s, sk_s, ec_s, sc_s = params_s[8:]

        G_mb_n =     k_n *iTemp_n
        G_td_n = (1.-k_n)*iTemp_n
        G_mb_s =     k_s *iTemp_s
        G_td_s = (1.-k_s)*iTemp_s

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
                alpQ, decQ, lbd, act_s, alpT, decT, alpCK, alpSK = params_s[:8]
            else:
                alpQ, decQ, lbd, act_n, alpT, decT, alpCK, alpSK = params_n[:8]

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

        act = act_n*n_choices + act_s*s_choices
        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        sk = sk_n*n_choices + sk_s*s_choices
        ec = ec_n*n_choices + ec_s*s_choices
        sc = sc_n*n_choices + sc_s*s_choices

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        Q_td = (1.-act)*Q+act*P # Mixture of action and target model free values.
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q_td + G_mb*M  # Mixture of model based and model free values.
        D = D[prev_sec_steps,np.arange(session.n_trials)] 
        Q_net[1,:] += bs + sk*(prev_sec_steps-0.5) + ec*C + sc*D

        return session_log_likelihood(choices, Q_net)


    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        P = np.zeros([2,2,n_trials+1]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(n_trials+1) # Choice kernel.
        D = np.zeros([2,n_trials+1]) # Previous side dependent choice kernel.
        ps = 0 # State reached at second step on previous trial.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials, stim=True)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o, stim = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            params = params_s if stim else params_n
            alpQ, decQ, lbd, act, alpT, decT, alpCK, alpSK, G_td, G_mb, bs, sk, ec, sc = params

            n = 1 - c   # Action not chosen at first step.
            r = 1 - s   # State not reached at second step.
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

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.
        
            Q_td = Q[:,i+1]*(1.-act) + P[:,s,i+1]*act

            Q_net = G_td*Q_td + G_mb*M  # Mixture of model based and model free values.
            Q_net[1] += bs + sk*(s-0.5) + ec*C[i+1] + sc*D[s,i+1] # Apply kernels.

            ps = s

        return choices, second_steps, outcomes


#------------------------------------------------------------------------------------

class stim_agent_ec_sc_mMF2(_Stim_agent):
    '''Mixture agent with decays, optimised for speed by removing flexibility
    about which kernels are used.'''

    def __init__(self, stim_params = 'all'):
        self.name = 'stim_agent_ec_sc_mMF'
        self.bp_names  = ['alpQ', 'decQ', 'lbd', 'alpT', 'decT', 'alpEC', 'alpMC',
                          'G_td', 'G_tdm', 'G_mb', 'bs', 'rb', 'ec', 'mc']
        self.bp_ranges = ['unit']*7 + ['pos']*3 + ['unc']*4
        _Stim_agent.__init__(self, stim_params, kernels = False)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        G_td_n, G_tdm_n, G_mb_n, bs_n, sk_n, ec_n, sc_n = params_n[7:]
        G_td_s, G_tdm_s, G_mb_s, bs_s, sk_s, ec_s, sc_s = params_s[7:]

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
                alpQ, decQ, lbd, alpT, decT, alpCK, alpSK = params_s[:7]
            else:
                alpQ, decQ, lbd, alpT, decT, alpCK, alpSK = params_n[:7]

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

        G_tdm = G_tdm_n*n_choices + G_tdm_s*s_choices
        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        sk = sk_n*n_choices + sk_s*s_choices
        ec = ec_n*n_choices + ec_s*s_choices
        sc = sc_n*n_choices + sc_s*s_choices

        P = P[:,prev_sec_steps,np.arange(session.n_trials)]
        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q + G_tdm*P + G_mb*M  # Mixture of model based and model free values.
        D = D[prev_sec_steps,np.arange(session.n_trials)] 
        Q_net[1,:] += bs + sk*(prev_sec_steps-0.5) + ec*C + sc*D

        return session_log_likelihood(choices, Q_net)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        P = np.zeros([2,2,n_trials+1]) # Prev. side dep. first step TD values. (c,ps,t)
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(n_trials+1) # Choice kernel.
        D = np.zeros([2,n_trials+1]) # Previous side dependent choice kernel.
        ps = 0 # State reached at second step on previous trial.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials, stim=True)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o, stim = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            params = params_s if stim else params_n
            alpQ, decQ, lbd, alpT, decT, alpCK, alpSK, G_td, G_tdm, G_mb, bs, sk, ec, sc = params

            n = 1 - c   # Action not chosen at first step.
            r = 1 - s   # State not reached at second step.
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

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.

            Q_net = G_td*Q[:,i+1] + G_tdm*P[:,s,i+1] + G_mb*M  # Mixture of model based and model free values.
            Q_net[1] += bs + sk*(s-0.5) + ec*C[i+1] + sc*D[s,i+1] # Apply kernels.

            ps = s

        return choices, second_steps, outcomes

#--------------------------------------------------------------------------------

class stim_agent_ec_sc(_Stim_agent):
    '''Exponential choice kernel, motor choice kernel..'''

    def __init__(self, stim_params = 'all'):
        self.name = 'stim_agent_ec_sc'
        self.bp_names  = ['alpQ', 'decQ', 'lbd', 'alpT', 'decT', 'alpEC', 'alpMC', 'G_td', 'G_mb', 'bs', 'rb', 'ec', 'mc']
        self.bp_ranges = ['unit']*7 + ['pos']*2 + ['unc']*4
        _Stim_agent.__init__(self, stim_params, kernels = False)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        G_td_n, G_mb_n, bs_n, sk_n, ec_n, sc_n = params_n[7:]
        G_td_s, G_mb_s, bs_s, sk_s, ec_s, sc_s = params_s[7:]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(session.n_trials) # Choice kernel.
        D = np.zeros([2,session.n_trials]) # Previous side dependent choice kernel.

        for i, (c, s, o, ps, su) in enumerate(zip(choices[:-1], second_steps, outcomes,
                                                  prev_sec_steps, s_updates)): # loop over trials.

            if su: # Stim update
                alpQ, decQ, lbd, alpT, decT, alpCK, alpSK = params_s[:7]
            else:
                alpQ, decQ, lbd, alpT, decT, alpCK, alpSK = params_n[:7]

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.
            pr = 1 - ps # State not reached at second step on previous trial.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            D[pr,i+1] = D[pr,i]

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            C[i+1] = (1.-alpCK)*C[i] + alpCK*(c-0.5)
            D[ps,i+1] = (1.-alpSK)*D[ps,i] + alpSK*(c-0.5)

        # Evaluate net action values and likelihood. 

        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        sk = sk_n*n_choices + sk_s*s_choices
        ec = ec_n*n_choices + ec_s*s_choices
        sc = sc_n*n_choices + sc_s*s_choices

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q + G_mb*M  # Mixture of model based and model free values.
        D = D[prev_sec_steps,np.arange(session.n_trials)] 
        Q_net[1,:] += bs + sk*(prev_sec_steps-0.5) + ec*C + sc*D

        return session_log_likelihood(choices, Q_net)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(n_trials+1) # Choice kernel.
        D = np.zeros([2,n_trials+1]) # Previous side dependent choice kernel.
        ps = 0 # State reached at second step on previous trial.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials, stim=True)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o, stim = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            params = params_s if stim else params_n
            alpQ, decQ, lbd, alpT, decT, alpCK, alpSK, G_td, G_mb, bs, sk, ec, sc = params

            n = 1 - c   # Action not chosen at first step.
            r = 1 - s   # State not reached at second step.
            pr = 1 - ps # State not reached at second step on previous trial.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.
            D[pr,i+1] = D[pr,i]

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            C[i+1] = (1.-alpCK)*C[i] + alpCK*(c-0.5)
            D[ps,i+1] = (1.-alpSK)*D[ps,i] + alpSK*(c-0.5)

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.
        
            Q_net = G_td*Q[:,i+1] + G_mb*M  # Mixture of model based and model free values.
            Q_net[1] += bs + sk*(s-0.5) + ec*C[i+1] + sc*D[s,i+1] # Apply kernels.

            ps = s

        return choices, second_steps, outcomes

#------------------------------------------------------------------------------


class stim_agent_ec(_Stim_agent):
    '''Exponential choice kernel, no motor choice kernel or RL.'''

    def __init__(self, stim_params = 'all'):
        self.name = 'stim_agent_ec'
        self.bp_names  = ['alpQ', 'decQ', 'lbd', 'alpT', 'decT', 'alpEC', 'G_td', 'G_mb', 'bs', 'rb', 'ec']
        self.bp_ranges = ['unit']*6 + ['pos']*2 + ['unc']*3
        _Stim_agent.__init__(self, stim_params, kernels = False)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        prev_sec_steps = np.hstack((0,second_steps[:-1]))
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        G_td_n, G_mb_n, bs_n, sk_n, ec_n = params_n[6:]
        G_td_s, G_mb_s, bs_s, sk_s, ec_s = params_s[6:]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(session.n_trials) # Choice kernel.

        for i, (c, s, o, su) in enumerate(zip(choices[:-1], second_steps, outcomes, s_updates)): # loop over trials.

            if su: # Stim update
                alpQ, decQ, lbd, alpT, decT, alpCK = params_s[:6]
            else:
                alpQ, decQ, lbd, alpT, decT, alpCK = params_n[:6]

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            C[i+1] = (1.-alpCK)*C[i] + alpCK*(c-0.5)

        # Evaluate net action values and likelihood. 

        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        sk = sk_n*n_choices + sk_s*s_choices
        ec = ec_n*n_choices + ec_s*s_choices

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q + G_mb*M  # Mixture of model based and model free values.
        Q_net[1,:] += bs + sk*(prev_sec_steps-0.5) + ec*C

        return session_log_likelihood(choices, Q_net)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        C = np.zeros(n_trials+1) # Choice kernel.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials, stim=True)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o, stim = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            params = params_s if stim else params_n
            alpQ, decQ, lbd, alpT, decT, alpCK, G_td, G_mb, bs, sk, ec = params

            n = 1 - c   # Action not chosen at first step.
            r = 1 - s   # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            C[i+1] = (1.-alpCK)*C[i] + alpCK*(c-0.5)

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.
        
            Q_net = G_td*Q[:,i+1] + G_mb*M  # Mixture of model based and model free values.
            Q_net[1] += bs + sk*(s-0.5) + ec*C[i+1] # Apply kernels.

        return choices, second_steps, outcomes

#------------------------------------------------------------------------------


class stim_agent_basic(_Stim_agent):
    '''Exponential choice kernel, no motor choice kernel or RL.'''

    def __init__(self, stim_params = 'all'):
        self.name = 'stim_agent_basic'
        self.bp_names  = ['alpQ', 'decQ', 'lbd', 'alpT', 'decT', 'G_td', 'G_mb', 'bs', 'rb', 'ck']
        self.bp_ranges = ['unit']*5 + ['pos']*2 + ['unc']*3
        _Stim_agent.__init__(self, stim_params, kernels = False)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')
        s_choices, n_choices, s_updates = _get_stim_data(session)

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        G_td_n, G_mb_n, bs_n, sk_n, ck_n = params_n[5:]
        G_td_s, G_mb_s, bs_s, sk_s, ck_s = params_s[5:]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o, su) in enumerate(zip(choices[:-1], second_steps, outcomes, s_updates)): # loop over trials.

            if su: # Stim update
                alpQ, decQ, lbd, alpT, decT = params_s[:5]
            else:
                alpQ, decQ, lbd, alpT, decT = params_n[:5]

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        G_td = G_td_n*n_choices + G_td_s*s_choices
        G_mb = G_mb_n*n_choices + G_mb_s*s_choices
        bs = bs_n*n_choices + bs_s*s_choices
        sk = sk_n*n_choices + sk_s*s_choices
        ck = ck_n*n_choices + ck_s*s_choices

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        
        Q_net = G_td*Q + G_mb*M  # Mixture of model based and model free values.
        Q_net[1,:] += (bs + sk*(np.hstack((0.5,second_steps[:-1]))-0.5)
                          + ck*(np.hstack((0.5,choices[:-1]))-0.5))

        return session_log_likelihood(choices, Q_net)

    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        params_n, params_s = (params_T[self.np_ind], params_T[self.sp_ind])

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials, stim=True)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o, stim = task.trial(c)
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            params = params_s if stim else params_n
            alpQ, decQ, lbd, alpT, decT, G_td, G_mb, bs, sk, ck = params

            n = 1 - c   # Action not chosen at first step.
            r = 1 - s   # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] * (1.-decQ) # First step forgetting.
            V[r,i+1] = V[r,i] * (1.-decQ) # Second step forgetting.
            T[n,i+1] = T[n,i] - decT*(T[n,i]-0.5) # Transition prob. forgetting.

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            M = T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1] # Model based action values.
        
            Q_net = G_td*Q[:,i+1] + G_mb*M  # Mixture of model based and model free values.
            Q_net[1] += bs + sk*(s-0.5) + ck*(c-0.5) # Apply kernels.

        return choices, second_steps, outcomes