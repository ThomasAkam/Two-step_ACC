import os
import scipy.io as scio
import pylab as plt
import numpy as np
from scipy.special import erf
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
from collections import OrderedDict, Counter
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.mplot3d import Axes3D
# from dPCA import dPCA

# Data import ------------------------------------------------------------------

def add_spike_data_to_session_old(session, data_path):
    'Add the spike data in file data_path to session object.'
    data = scio.loadmat(data_path)
    if 'neuron_' in data.keys():
        spikes = data['neuron_']['S'][0][0]
    elif 'results' in data.keys():
        spikes = data['results'][0][0]['S'].toarray()
    else:
        print('Data type not recognised')
        return
    frame_times = session.times['scope_frame'][:spikes.shape[1]]
    fs = len(frame_times)/frame_times[-1]
    session.calcium_data = {'spikes':spikes, 'frame_times':frame_times,'fs':fs}

def add_spike_data_to_session(session, data_folder, verbose=False):
    spikes = np.load(os.path.join(data_folder, 'S.npy'))
    frame_times = session.times['scope_frame']
    fs = 1/np.mean(np.diff(frame_times))
    if verbose:
        print(session.file_name + ' First frame time: {:.2f}'.format(frame_times[0]))
    if len(frame_times) >= spikes.shape[1]:
        if verbose:
            print (session.file_name + ': {} too many imaging frames for frame_times.'
                   .format(len(frame_times) - spikes.shape[1]))
        frame_times = session.times['scope_frame'][:spikes.shape[1]]
    session.calcium_data = {'spikes':spikes, 'frame_times':frame_times,'fs':fs}

def add_aligned_activity_to_sessions(sessions, median_latencies, fs=20, smooth_SD='auto'):
    '''For each session in sessions calculate the trial aligned neuronal activity 
    and store results in session.calcium_data.'''
    for i, ses in enumerate(sessions):
        if ses.calcium_data:
            print('Aligning session {} of {}'.format(i+1,len(sessions)))
            ses.calcium_data['aligned'] = align_activity(ses, median_latencies, fs, smooth_SD)

# View raw traces ------------------------------------------------

def view_raw_traces(data_folder, fs=10, t_max=300):
    spikes = np.load(os.path.join(data_folder, 'S.npy')) # [n_neurons, n_frames]
    calcium = np.load(os.path.join(data_folder, 'C.npy'))          # [n_neurons, n_frames]
    calcium_raw = np.load(os.path.join(data_folder, 'C_raw.npy'))   # [n_neurons, n_frames]
    t = np.arange(0,t_max,1/fs)
    plt.figure(1, figsize=[10,3])
    for n in range(spikes.shape[0]):
        plt.clf()
        ax = plt.subplot(2,1,1)
        plt.plot(t,calcium_raw[n,:len(t)])
        plt.plot(t,calcium[n,:len(t)])
        plt.xlim(t[0],t[-1])
        ax.xaxis.set_ticklabels([])
        plt.ylabel('ΔF\n(arb. unit)', multialignment='center')
        plt.subplot(2,1,2)
        plt.plot(t,spikes[n,:len(t)])
        plt.xlim(t[0],t[-1])
        plt.ylabel('Spikes\n(arb. unit)', multialignment='center')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.pause(0.05)
        if input("Neuron #: {}, press enter for next neuron, 'x' to exit:".format(n)) == 'x':
            break

# Event rate histogram ---------------------------------------------------------

def event_rate_histogram(sessions, fig_no=1, save_dir=None):
    '''Plot histogram of event rates across cell population where an event is 
    any frame with spike count greater than 1'''
    cell_average_rates = []
    for session in sessions:
        spikes = session.calcium_data['spikes']
        duration = spikes.shape[1]/session.calcium_data['fs']
        cell_average_rates.append(np.sum(spikes.astype(bool),1)/duration)
    cell_average_rates = np.hstack(cell_average_rates)
    plt.figure(fig_no, figsize=[3.8,2.7]).clf()
    plt.hist(cell_average_rates,20)
    plt.ylabel('Number of cells')
    plt.xlabel('Calcium event rate (Hz)')
    plt.tight_layout()
    print(f'Median rate: {np.median(cell_average_rates):.2f}')
    if save_dir: 
        plt.savefig(os.path.join(save_dir,'event rate historgram.pdf'))

# Timing alignment -------------------------------------------------------------

def align_activity(session, median_latencies, fs_out=20, smooth_SD='auto', plot_warp=False,
                   align_times = [('C',-1000),('C',0),('O',0), ('O',1000)]):
    '''Time stretch neuronal activity for each trial to align event times to the median 
    non-rewarded trial timings.  Function outputs neuronal activity at regularly spaced timepoints in the aligned
    trial. The activity at each timepoint is evaluated by linear interpolation between
    input samples followed by Gaussian smoothing around output sample timepoints.  This is 
    implemented using the analytical solution to the overlap integral between the linearly
    interpolated input activity and Gaussian smoothing kernel around the output sample
    Arguments: 
    session: Session with imaging data attached.
    median_latencies: Median latencies between trial events (generate by pl.trial_timings_analysis)
    fs_out: The sampling rate (Hz) of the aligned output activity.
    smooth_SD: The standard deviation (ms) of the Gaussian smoothing applied to output 
               activity. If set to 'auto', smooth_SD is set to 1000/fs_out.
    plot_warp:  If True the input and output activity are plotted for the most active 
    neurons for each trial.
    align_times: Time points within trial used for alignment, must be a list of tuples 
    whose first element is 'C', 'O' or 'N' indicating whether the time is defined relative
    to the choice, outcome or next choice, and second element is a number of ms, e.g.
    ('C',-1000) indicates 1000ms before choice.'''

    if smooth_SD=='auto': smooth_SD = 1000/fs_out

    median_times = {'C':0,'O':median_latencies['CO'],'N':median_latencies['CC_non']}

    target_times = np.array([median_times[at[0]]+at[1] for at in align_times])

    t_out = np.arange(target_times[0], target_times[-1], 1000/fs_out) # Timepoints of output samples relative to choice.

    align_samples = [np.argmin(np.abs(t_out-tt)) for tt in target_times] # Output sample numbers corresponding to alignment points.

    # Times of trial events in session.
    tC  = 1000*session.times['choice'][:-1]        # Choices.
    tO  = 1000*session.times['outcome'][:len(tC)]  # Outcomes.
    tN  = 1000*session.times['choice'][1:]         # Next choice.

    trial_times = np.array([{'C':tC,'O':tO,'N':tN}[at[0]]+at[1] for at in align_times]).T

    # Extend alignment start and end points to prevent edge effects.
    target_times[0 ]  -= 200 
    target_times[-1]  += 200
    trial_times[:, 0] -= 200
    trial_times[:,-1] += 200

    assert not np.any(np.diff(trial_times,1)<0), 'Supplied align_times give negative time differences'

    align_sec_durs = np.diff(target_times)  # Duration of trial sections in aligned activity (ms).
    trial_sec_durs = np.diff(trial_times,1) # Actual duration of trial sections for each trial (ms).

    frame_times = 1000*session.calcium_data['frame_times']

    aligned_activity = []

    for i in np.arange(len(tC)):
        if trial_times[i,-1] > frame_times[-1]:
            break # This and subsequent trials occured after recording finshed.

        # Timestretch input frame times to match median inter event intervals.
        trial_frames =  ((trial_times[i,0] <= frame_times) & 
                         (frame_times < trial_times[i,-1]))
        t0 = frame_times[trial_frames]  # Trial frame times before timestretching
        t1 = np.zeros(len(t0))          # Trial frame times relative to choice after timestretching.
        for j in range(align_sec_durs.shape[0]):
            mask = (trial_times[i,j] <= t0) & (t0 < trial_times[i,j+1])
            t1[mask] = (t0[mask]-trial_times[i,j])*align_sec_durs[j]/trial_sec_durs[i,j] + target_times[j]

        trial_activity = session.calcium_data['spikes'][:,np.where(trial_frames)[0]]

        aligned_activity.append(np.sum(_int_norm_lin_prod(trial_activity[:,:-1],
            trial_activity[:,1:],t1[:-1],t1[1:],t_out[:,None,None],s=smooth_SD),2).T)

        if plot_warp: # Plot input and output activity for the most active neurons.
            most_active = np.argsort(np.mean(trial_activity,1))[-5:]
            plt.figure(2, figsize=[10,3.2]).clf()
            plt.subplot2grid((1,3),(0,0))
            plt.plot(t0-tC[i],t1,'.-')
            plt.xlim(-1000,t0[-1]-tC[i])
            plt.ylabel('Aligned trial time\n(ms relative to choice)')
            plt.xlabel('True trial time (ms relative to choice)')
            plt.subplot2grid((2,3),(0,1), colspan=2)
            plt.plot(t0-tC[i], trial_activity[most_active,:].T,'.-')
            y = plt.ylim()
            for x in trial_times[i,:]-tC[i]:
                plt.plot([x,x],y,'k:')
            plt.xlim(-1000,t0[-1]-tC[i])
            plt.ylabel('Activity')
            plt.xlabel('True trial time (ms relative to choice)')
            plt.subplot2grid((2,3),(1,1), colspan=2)
            plt.plot(t_out,aligned_activity[-1][most_active,:].T,'.-')
            y = plt.ylim()
            for x in target_times:
                plt.plot([x,x],y,'k:')
            plt.xlim(-1000,t0[-1]-tC[i])
            #plt.xlim(t_out[0], t_out[-1])
            plt.ylabel('Activity')
            plt.xlabel('Aligned trial time (ms relative to choice)')
            plt.tight_layout()
            plt.pause(0.05)
            if input("Press enter for next trial, 'x' to exit:") == 'x':
                break

    # Make labels for plotting alignment points.
    align_labels = []
    for at in align_times:
        label = {'C':'Choice', 'O': 'Outcome', 'N': 'Choice'}[at[0]]
        if at[1] > 0:
            label += ' + {}ms'.format(at[1])
        elif at[1] < 0:
            label += ' - {}ms'.format(abs(at[1]))
        align_labels.append(label)
    target_times[0], target_times[-1]  = (t_out[0],t_out[-1])

    return {'spikes'          : np.array(aligned_activity), # Aligned neural activity
            't_out'           : t_out,                      # Time of aligned activity relative to choice (ms).
            'align_sec_durs'  : align_sec_durs,             # Duration of trial sections in aligned activity (ms).
            'trial_sec_durs'  : trial_sec_durs,             # Actual duration of trial sections for each trial (ms).
            'median_latencies': median_latencies,           # Median trial timings used for alignment.
            'align_times'     : target_times,               # Output alignment point times
            'align_samples'   : align_samples,              # Output alignment point sample numbers.
            'align_labels'    : align_labels}               # Labels for output alignment timepoints.

r2pi = np.sqrt(2*np.pi) # Constants used in _int_norm_lin_prod
r2   = np.sqrt(2) 

def _int_norm_lin_prod(a,b,v,t,u,s):
    '''Evaluate the integral of (a+(b-a)*(x-v)/(t-v))*Npdf(u,s) from v to t where 
    Npdf is the probability density function of the Normal distribution. Wolfram: 
    integrate ((a+(b-a)*(x-v)/(t-v))/(s*sqrt(2*pi)))*exp(-((x-u)^2)/(2*s^2)) from v to t'''
    return (1/(2*r2pi*(t-v)))*(r2pi*(a*(t-u)+b*(u-v))*(erf((t-u)/(r2*s))-erf((v-u)/(r2*s)))+
            2*s*(a-b)*(np.exp(-((t-u)**2)/(2*s**2))-np.exp(-((v-u)**2)/(2*s**2))))

# Aligned activity plots -----------------------------------------------------------

def ave_activity_across_trial(sessions, normalise=False, log=True, fig_no=1, 
                              vmax=False, save_dir=None):
    '''Plot the mean activity of each neuron across all trials as a heatmap with the 
    neurons ordered by their time of peak activity. If normalise=True the activity of 
    each neuron is normalised to the same mean value.'''
    if not type(sessions) == list: sessions = [sessions]
    mean_activity = []
    for session in sessions:
            aligned_activity = session.calcium_data['aligned']
            aligned_spikes = aligned_activity['spikes']
            if log:
                aligned_spikes = np.log2(aligned_spikes+0.01)-np.log2(0.01)
            mean_activity.append(np.mean(aligned_spikes,0))
    mean_activity = np.vstack(mean_activity)
    t = aligned_activity['t_out']
    ordering_end_time = aligned_activity['median_latencies']['CC_non']-500
    if normalise: 
        mean_activity = mean_activity/np.mean(mean_activity[:,t<ordering_end_time],1)[:,None]
    if not vmax: vmax=np.percentile(mean_activity,98)
    # Plot heatmap.
    _ordered_heatmap(mean_activity, aligned_activity, fig_no, cmap=plt.cm.plasma, vmax=vmax)
    plt.colorbar()
    plt.tight_layout()
    if save_dir: 
        plt.savefig(os.path.join(save_dir,'average activity across trial.pdf'))
    # Scatter plot of average activity pre outcome vs post outcome.
    pre_ind, post_ind, choice_ind, outcome_ind = _get_pre_post_outcome_inds(sessions[0])
    plt.figure(fig_no+100, figsize=[4,4]).clf()
    plt.axhline(0,linestyle=':',color='k')
    plt.axvline(0,linestyle=':',color='k')
    plt.plot(mean_activity[:,pre_ind], mean_activity[:, post_ind],'.')
    plt.title(f'R-squared: {_rsquared(mean_activity[:,pre_ind], mean_activity[:, post_ind]):.4f}')    
    
def trial_by_trial_heatmap(session, sort_trials=False, fig_no=1):
    '''For each neuron plot a heatmap showing the activity across trial number and
    time.  Optionally, trials can be sorted by trial events (choice, second_step,
    outcome) rather than trial order.
    '''
    aligned_spikes = session.calcium_data['aligned']['spikes']
    t = session.calcium_data['aligned']['t_out']
    if sort_trials: # Sort trials by events.
        osc = np.vstack(session.unpack_trial_data('OSC'))[:,:aligned_spikes.shape[0]]
        ordering = np.lexsort(osc)
        aligned_spikes = aligned_spikes[ordering,:,:]
        cso = np.vstack(session.unpack_trial_data('CSO'))[:,:aligned_spikes.shape[0]].T
        cso_sorted = cso[ordering,:]
    n_h, n_v = 5,6 # Number of subplots horizontally and vertically.
    plot_per_fig = (n_h*n_v)
    f = fig_no
    plt.figure(f, figsize=[25.6,13.06]).clf()
    p = 0
    for n in range(aligned_spikes.shape[1]):
        f_new = fig_no + n // plot_per_fig
        if f_new != f: # New figure.
            plt.tight_layout()
            plt.figure(f_new, figsize=[25.6,13.06]).clf()
            f = f_new
            p = 0
        p += 1
        ax = plt.subplot(n_h, n_v, p)
        neuron_spikes = aligned_spikes[:,n,:]
        if sort_trials: # Add extra columns showing value of choices, sec_steps, outcomes.
            neuron_spikes = np.hstack([neuron_spikes,cso_sorted*np.max(neuron_spikes)])
        ax.imshow(neuron_spikes, aspect='auto', extent=[t[0],t[-1],aligned_spikes.shape[0], 1])
        _aligned_x_ticks(session.calcium_data['aligned'], tick_labels='short')
    plt.tight_layout() 

def _ordered_heatmap(aligned_data, aligned_activity, fig_no, cmap=plt.cm.plasma, vmax=None,
                     vmin=None, order='max', title=None, xtick_labels=True, return_order=False):
    assert (type(order) == np.ndarray) or (order in ['phase', 'max']), \
        "order must be 'phase', 'max' or a 1D numpy array of size n_neurons."
    med_lat = aligned_activity['median_latencies']
    t = aligned_activity['t_out']
    ordering_end_time = med_lat['CC_non']-500
    n_neurons = aligned_data.shape[0]
    if type(order)==np.ndarray:
        ordering = np.argsort(order)
    elif order == 'max': # Order by position of maximum value.
        ordering = np.argsort(np.argmax(aligned_data[:,t<ordering_end_time],1))
    else:
        phase_vec = -np.exp(1j*(2*np.pi*(t[t<ordering_end_time]-t[0]) / (ordering_end_time-t[0])))
        ordering  = np.argsort(np.angle(np.sum(aligned_data[:,t<ordering_end_time]*phase_vec,1)))
    ordered_activity = aligned_data[ordering,:]
    if fig_no: plt.figure(fig_no, figsize=[4.8,5]).clf()
    plt.imshow(ordered_activity,aspect='auto',vmax=vmax, vmin=vmin,
               extent=[t[0],t[-1],1,n_neurons], cmap=cmap)
    for x in aligned_activity['align_times'][1:-1]:
            plt.plot([x,x], [1,n_neurons],'k:')
    _aligned_x_ticks(aligned_activity, xtick_labels)
    plt.ylabel('Neuron #')
    plt.xlim(t[0],t[-1])
    plt.ylim(1,n_neurons)
    if title: plt.title(title)
    if return_order: return ordering

def _aligned_x_ticks(aligned_activity, tick_labels=True):
    ha, rotation, labels = ('left', 0, [])
    if tick_labels:
        if tick_labels == 'short':
            labels = [(l[0] if not 'ms' in l else '') 
                      for l in aligned_activity['align_labels']]
            ha = 'center'
        else:
            labels = aligned_activity['align_labels']
            rotation = -45
    plt.xticks(aligned_activity['align_times'],labels, rotation=rotation, ha=ha)

def plot_selective_neurons(session, fig_no=1, n_neurons=10, save_dir=None):

    def _plot_neuron_cond(neuron_ind, mask, color, label=None):
        '''Plot the mean +/- SEM activity for trials where mask is true
        for neuron number neuron_ind'''
        activity = aligned_spikes[mask,neuron_ind,:]
        m = np.mean(activity, axis=0)
        s = sem(activity, axis=0)
        plt.fill_between(t, m-s, m+s, alpha=0.4, color=color, lw=0)
        plt.plot(t, m, color=color, label=label, linewidth=1)

    def _get_selective_neurons(masks, n_neurons):
        '''Return the indices of those neurons that most strongly discriminate
        between trials where masks are true and false. '''
        if type(masks) != list: masks = [masks]
        d = []
        for i, mask in enumerate(masks):
            activity_A = aligned_spikes[ mask,:,:]
            activity_B = aligned_spikes[~mask,:,:]
            mean_A = np.mean(activity_A, axis=0)
            mean_B = np.mean(activity_B, axis=0)
            var_A =  np.var(activity_A, axis=0)+0.1
            var_B =  np.var(activity_B, axis=0)+0.1
            d.append(np.abs(mean_A-mean_B)/np.sqrt(var_A+var_B))
        return np.argsort(np.max(np.prod(d,axis=0), axis=1))[-n_neurons:]  

    def _format_axis():
        ax.axvline(ct, color='k', linestyle=':', linewidth=0.75)
        ax.axvline(ot, color='k', linestyle=':', linewidth=0.75)
        plt.text(0.02, 0.8, 'n'+str(i), transform=ax.transAxes)
        _aligned_x_ticks(session.calcium_data['aligned'], p+1 == n_neurons)
        plt.xlim(t[0],t[-1])

    aligned_spikes = session.calcium_data['aligned']['spikes']
    n_trials = aligned_spikes.shape[0]
    c = session.trial_data['choices'     ][:n_trials].astype(bool)
    s = session.trial_data['second_steps'][:n_trials].astype(bool)
    o = session.trial_data['outcomes'    ][:n_trials].astype(bool)
    # Plotting
    plt.figure(fig_no, figsize=[16,9], clear=True)
    t = session.calcium_data['aligned']['t_out']
    ct, ot = session.calcium_data['aligned']['align_times'][1:3]
            
    # Plot choice selective neurons split by choice and second-step.
    neuron_inds = _get_selective_neurons(c, n_neurons)
    for p, i in enumerate(reversed(neuron_inds)):
        ax = plt.subplot(n_neurons,5,5*p+1)
        _plot_neuron_cond(i,  c &  s, color='C0', label='C:T SS:L')
        _plot_neuron_cond(i, ~c &  s, color='C1', label='C:B SS:L')
        _plot_neuron_cond(i,  c & ~s, color='C2', label='C:T SS:R')
        _plot_neuron_cond(i, ~c & ~s, color='C6', label='C:B SS:R')
        _format_axis()
    plt.legend(bbox_to_anchor=(-0.7, 0.9, 0.5, 0.5))
    # Plot choice & second step selective neurons split by choice and second-step.
    neuron_inds = _get_selective_neurons([c,s], n_neurons)
    for p, i in enumerate(neuron_inds):
        ax = plt.subplot(n_neurons,5,5*p+2)
        _plot_neuron_cond(i,  c &  s, color='C0')
        _plot_neuron_cond(i, ~c &  s, color='C1')
        _plot_neuron_cond(i,  c & ~s, color='C2')
        _plot_neuron_cond(i, ~c & ~s, color='C6')
        _format_axis()
    # Plot second step selective neurons split by choice and second-step.
    neuron_inds = _get_selective_neurons(s, n_neurons)
    for p, i in enumerate(neuron_inds):
        ax = plt.subplot(n_neurons,5,5*p+3)
        _plot_neuron_cond(i,  c &  s, color='C0')
        _plot_neuron_cond(i, ~c &  s, color='C1')
        _plot_neuron_cond(i,  c & ~s, color='C2')
        _plot_neuron_cond(i, ~c & ~s, color='C6')
        _format_axis()
    # Plot second-step and outcome selective neurons split by second-step and outcome.
    neuron_inds = _get_selective_neurons([s,o], n_neurons)
    for p, i in enumerate(neuron_inds):
        ax = plt.subplot(n_neurons,5,5*p+4)
        _plot_neuron_cond(i,  o &  s, color='C3')
        _plot_neuron_cond(i, ~o &  s, color='C9')
        _plot_neuron_cond(i,  o & ~s, color='C4')
        _plot_neuron_cond(i, ~o & ~s, color='C7')
        _format_axis()
    # Plot outcome selective neurons split by second-step and outcome.
    neuron_inds = _get_selective_neurons(o, n_neurons)
    for p, i in enumerate(neuron_inds):
        ax = plt.subplot(n_neurons,5,5*p+5)
        plt.text(0.02, 0.8, 'n'+str(i), transform=ax.transAxes)
        _plot_neuron_cond(i,  o &  s, color='C3', label='SS:L O:rew')
        _plot_neuron_cond(i, ~o &  s, color='C9', label='SS:L O:non')
        _plot_neuron_cond(i,  o & ~s, color='C4', label='SS:R O:rew')
        _plot_neuron_cond(i, ~o & ~s, color='C7', label='SS:R O:non')
        _format_axis()
    plt.legend(bbox_to_anchor=(1.05, 0.9, 0.5, 0.5))
    plt.suptitle(session.file_name)
    if save_dir: 
        plt.savefig(os.path.join(save_dir, session.file_name + '.pdf'))



def condition_average_rates(session, fig_no=1, log=False, sort_neurons=True):
    '''Plot average firing rates for every neuron for a set of different trial event
    conditions.'''
    aligned_spikes = session.calcium_data['aligned']['spikes']
    if log:
        aligned_spikes = np.log2(aligned_spikes+0.1)-np.log2(0.1)
    if sort_neurons: # Sort neurons in decending order of mean activity.
        ordering = np.argsort(np.mean(aligned_spikes, axis=(0,2)))
        aligned_spikes = aligned_spikes[:,ordering[::-1],:]
    n_img_trials = aligned_spikes.shape[0]
    choices, second_steps, outcomes = session.unpack_trial_data('CSO',bool)
    cond_trials_cs = {'C:1 S:1':  choices &  second_steps, # Trials matching each condition.
                      'C:1 S:0':  choices & ~second_steps,
                      'C:0 S:1': ~choices &  second_steps,
                      'C:0 S:0': ~choices & ~second_steps}
    cond_style_cs = {'C:1 S:1': 'r',  # Plotting line style for each condition.
                     'C:1 S:0': 'b',
                     'C:0 S:1': 'r--',
                     'C:0 S:0': 'b--'}
    cond_trials_so = {'S:1 O:1':  second_steps &  outcomes,
                      'S:0 O:1': ~second_steps &  outcomes,
                      'S:1 O:0':  second_steps & ~outcomes,
                      'S:0 O:0': ~second_steps & ~outcomes}
    cond_style_so = {'S:1 O:1': 'r-.',
                     'S:0 O:1': 'b-.',
                     'S:1 O:0': 'r:',
                     'S:0 O:0': 'b:'}
    # Evaulate mean activity and number of trials in each condition.
    cond_n_trials_cs = {cond: np.sum(cond_trials_cs[cond]) for cond in cond_trials_cs.keys()}
    cond_means_cs = {cond: np.mean(aligned_spikes[cond_trials_cs[cond][:n_img_trials],:,:],0)
                     for cond in cond_trials_cs.keys()} 
    cond_n_trials_so = {cond: np.sum(cond_trials_so[cond]) for cond in cond_trials_so.keys()}
    cond_means_so = {cond: np.mean(aligned_spikes[cond_trials_so[cond][:n_img_trials],:,:],0)
                     for cond in cond_trials_so.keys()} 
    # Plotting

    def _plot_cond_aves(cond_means, cond_style, cond_n_trials, sharey=None):
        ax = plt.subplot(n_h, n_v, p, sharey=sharey)
        for cond in cond_means:
            plt.plot(t, cond_means[cond][n,:], cond_style[cond], 
                     label=cond+', trials: {}'.format(cond_n_trials[cond]))
        plt.text(0.02, 0.8, 'n'+str(n), transform=ax.transAxes)
        _aligned_x_ticks(session.calcium_data['aligned'], 'short')
        plt.xlim(t[0],t[-1])
        return ax

    t = session.calcium_data['aligned']['t_out']
    n_h, n_v = 10,6 # Number of subplots horizontally and vertically.
    plot_per_fig = (n_h*n_v)# - 1
    f = fig_no
    plt.figure(f, figsize=[25.6,13.06]).clf()
    p = 0
    for n in range(aligned_spikes.shape[1]):
        f_new = fig_no + 2*n // plot_per_fig
        if f_new != f: # New figure.
            plt.tight_layout()
            plt.figure(f_new, figsize=[25.6,13.06]).clf()
            f = f_new
            p = 0
        p += 1
        ax1 = _plot_cond_aves(cond_means_cs, cond_style_cs, cond_n_trials_cs)
        p += 1
        ax2 = _plot_cond_aves(cond_means_so, cond_style_so, cond_n_trials_so, ax1)
    plt.tight_layout()
    ax1.legend(loc=(0,-1.5))
    ax2.legend(loc=(0,-1.5))

# -------------------------------------------------------------------------------
# Regression analyses
# -------------------------------------------------------------------------------

def regression_analysis(sessions, log=True, fig_no=1, vmax=False, order='phase', 
        perm=False, heatmaps=False, save_dir=None, return_betas=False, 
        predictors='current_trial', partition=None):
    '''Linear regression analysis predicting firing rates as a function of trial events.
    The predictor loadings are plotted for each neuron in the population as heat maps with the 
    neurons ordered to by phase of loading across trial.  Neuron orderings are not the same
    for different predictors.'''
    if not type(sessions) == list: sessions = [sessions] # Handle individual sessions.
    if type(predictors) == str:
        ct_preds = ['choice','second step','outcome','ch_x_ss','ch_x_out','ss_x_out']
        predictors = {
            'current_trial': ct_preds,
            'trans_block'  : ct_preds + ['common_trans', 'ch_x_t_block', 'ss_x_t_block', 'trans_state'],
            'reward_block' : ct_preds + ['tr_x_r_block', 'ch_x_r_block', 'ss_x_r_block', 'reward_state'],
            'both_block'   : ct_preds + ['common_trans', 'ch_x_t_block', 'ss_x_t_block', 'trans_state',
                                         'tr_x_r_block', 'ch_x_r_block', 'ss_x_r_block', 'reward_state']
            }[predictors]
    betas = [] # To strore predictor loadings for each session.
    cpd   = []   # To strore cpd for each session.
    if perm:
        betas_perm = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm   = [[] for i in range(perm)] # To store permuted cpd for each session.
    for session in tqdm(sessions):
        aligned_spikes = session.calcium_data['aligned']['spikes']
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape
        if log:
            aligned_spikes = np.log2(aligned_spikes+0.01)
        X = _get_session_predictors(session, n_trials, predictors) # Predictor matrix [n_trials, n_predictors]  
        n_predictors = X.shape[1]
        y = aligned_spikes.reshape([n_trials,-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        if partition: # Select only even or odd trials for cross validation purposes.
            assert partition in ['even', 'odd'], "partition must be 'even', 'odd' or None."
            i = 0 if partition == 'even' else 1
            X = X[i::2,:]
            y = y[i::2,:]
        ols = LinearRegression()
        ols.fit(X,y)
        betas.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        if perm:
            for i in range(perm):
                X = np.roll(X,np.random.randint(n_trials), axis=0)
                ols.fit(X,y)
                betas_perm[i].append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
                cpd_perm[i].append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

    betas = np.concatenate(betas,0)
    #betas2 = np.sum(betas**2,0) # Population sum of squared betas.
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons - nanmean handles neuron-timepoints where y has no variance.
    
    if return_betas: return betas

    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        #betas2_perm = np.stack([np.sum(np.concatenate(betas_i,0)**2,0) for betas_i in betas_perm],0)
        #betas2_p_value = np.mean(betas2_perm > betas2,0)
        cpd_p_value    = np.mean(cpd_perm    > cpd   ,0)

    # Plot CPD timecourses

    plt.figure(fig_no, figsize=[7,8 if perm else 6]).clf()
    ax1 = plt.subplot(2,1,1)
    plt.ylabel('Coef. partial determination (%)')
    _aligned_x_ticks(session.calcium_data['aligned'], tick_labels=False)
    ax2 = plt.subplot(2,1,2)
    plt.ylabel('Coef. partial determination (%)')
    _aligned_x_ticks(session.calcium_data['aligned'], tick_labels=True)
    t = session.calcium_data['aligned']['t_out']
    for i, predictor in enumerate(predictors):
        if predictor in ct_preds:
            ax1.plot(t, 100*cpd[:,i], label=predictor, color='C{}'.format(i))
        else:
            ax2.plot(t, 100*cpd[:,i], label=predictor, color='C{}'.format(i))
    if perm:
        # Indicate significance level for each predictor.
        _plot_P_values(cpd_p_value, t, ax1, 4, multi_comp=True)
    
    for ax in (ax1, ax2):
        ax.set_xlim(t[0], t[-1])
        #ax.set_ylim(bottom=0.3)
        ax.legend(bbox_to_anchor=(1, 1))
        for x in session.calcium_data['aligned']['align_times'][1:-1]:
            ax.axvline(x, color='k', linestyle=':')
    ax1.set_ylim(0.3, 4+2*bool(perm))
    ax2.set_ylim(0.3, 1.1)
    plt.tight_layout()
    if save_dir: 
        plt.savefig(os.path.join(save_dir,'regression cpd.pdf'))

    # Plot heatmaps of betas.
    if heatmaps:
        if order == 'TSNE':
            order = manifold.TSNE(1, perplexity=100, init='random', metric= 'correlation'
                ).fit_transform(np.reshape(betas[:,:,:3], [betas.shape[0],-1], order='F')).squeeze()
        plt.figure(fig_no+100, figsize=[9,6.5]).clf()
        if vmax:
            vmin = - vmax
        else:
            vmin, vmax = (-1,1) if log else (-4,4) 
        for i, predictor in enumerate(predictors):
            plt.subplot(3,np.ceil(len(predictors)/3),i+1)
            _ordered_heatmap(betas[:,:,i], session.calcium_data['aligned'], False,  
                             order=order, vmax=vmax, vmin=vmin, cmap=plt.cm.bwr, title=predictor)
        plt.tight_layout()

def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression()
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd


def _plot_P_values(p_values, t, ax, y0, multi_comp):
    '''Indicate significance levels with dots of different sizes above plot.'''
    n_pvals = p_values.shape[1]
    for i in range(n_pvals):
        y = y0*(1+0.04*(n_pvals-i))
        p_vals = p_values[:,i]
        if multi_comp: # Benjamini-Hochberg multiple comparison correction.
            p_vals = multipletests(p_vals, method='fdr_bh')[1]
        t05 = t[(p_vals < 0.05) & (p_vals >= 0.01 )]
        t01 = t[(p_vals < 0.01) & (p_vals >= 0.001)]
        t00 = t[p_vals < 0.001]
        ax.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color='C{}'.format(i))
        ax.plot(t01, np.ones(t01.shape)*y, '.', markersize=6, color='C{}'.format(i))
        ax.plot(t00, np.ones(t00.shape)*y, '.', markersize=9, color='C{}'.format(i))

def _get_session_predictors(session, n_trials, predictors):
        '''Get predictors used in regression analysis for one session.'''
        # Current trial predictors.
        choices, trans_AB, second_steps, outcomes = session.unpack_trial_data('CTSO', bool)
        outcome_A = (outcomes-0.5)*second_steps
        outcome_B = (outcomes-0.5)*(~second_steps)
        ch_x_ss  = choices == second_steps
        ch_x_out = choices == outcomes
        ss_x_out = second_steps == outcomes
        # Previous trail predictors.
        prev_choice  = np.roll(choices,1)
        prev_secstep = np.roll(second_steps,1)
        prev_outcome = np.roll(outcomes,1)
        rep_s = prev_secstep == second_steps
        rep_c = prev_choice  == choices 
        rep_o = prev_outcome == outcomes
        # Transtion block predictors.
        trans_state = session.blocks['trial_trans_state']
        not_post_trans_rev = session.select_trials('xtr', 20).astype(float)
        common_trans = ((trans_AB == trans_state)-0.5)*not_post_trans_rev
        ch_x_t_block = ((choices == trans_state)-0.5)*not_post_trans_rev
        ss_x_t_block = ((second_steps == trans_state)-0.5)*not_post_trans_rev
        # Reward block predictors.
        reward_state = session.blocks['trial_rew_state'] - 1.
        not_post_rew_cng = session.select_trials('xrr', 20).astype(float)
        ch_x_r_block = ((choices     -0.5)*reward_state)*not_post_rew_cng
        ss_x_r_block = ((second_steps-0.5)*reward_state)*not_post_rew_cng
        trans_x_r_block = ((trans_AB == reward_state)-0.5)*not_post_rew_cng
        all_predictors = {
                      # Current trail predictors
                      'choice'       : choices     ,
                      'second step'  : second_steps,
                      'outcome'      : outcomes    ,
                      'ch_x_ss'      : ch_x_ss     ,
                      'ch_x_out'     : ch_x_out    ,
                      'ss_x_out'     : ss_x_out    ,
                      'outcome_A'   : outcome_A   ,
                      'outcome_B'   : outcome_B   ,
                      # Previous trial predictors
                      'prev_choice' : prev_choice , 
                      'prev_secstep': prev_secstep, 
                      'prev_outcome': prev_outcome,
                      'rep_s'      : rep_s      , 
                      'rep_c'      : rep_c      , 
                      'rep_o'      : rep_o      , 
                      # Transition block predictors
                      'trans_state'  : trans_state,
                      'common_trans' : common_trans,
                      'ch_x_t_block' : ch_x_t_block,
                      'ss_x_t_block' : ss_x_t_block,
                      # Reward block predictors
                      'reward_state' : reward_state,
                      'ch_x_r_block' : ch_x_r_block,
                      'ss_x_r_block' : ss_x_r_block,
                      'tr_x_r_block' : trans_x_r_block}
        X = np.vstack([all_predictors[p] for p in predictors]).T[:n_trials,:].astype(float)
        pred_rank = np.linalg.matrix_rank(np.hstack([X,np.ones([X.shape[0],1])]))
        assert pred_rank == X.shape[1] + 1, f'Predictor matrix rank deficient: {session.file_name}'
        return X

def predictor_corrlations(sessions, predictors, fig_no=1):
    '''Plot the correlation matrix for a set of predictors used in the regression 
    analysis.'''
    X = []
    for session in sessions:
        n_trials = session.calcium_data['aligned']['spikes'].shape[0]
        try:
            X.append(_get_session_predictors(session, n_trials, predictors))
        except AssertionError as e:
            print(e)
    X = np.concatenate(X,0)
    C = np.corrcoef(X.T)
    plt.figure(fig_no, clear=True)
    plt.imshow(C, vmin=-1,vmax=1, cmap='RdBu')
    plt.colorbar()
    plt.xticks(range(len(predictors)), predictors, rotation=90)
    plt.yticks(range(len(predictors)), predictors)
    plt.tight_layout()

# ----------------------------------------------------------------------

def second_step_representation_evolution(sessions, fig_no=1):
    '''Plot the correlation between the representation of the second step
    state before and after the outcome is recieved, and how the strength of 
    these two representations changes across the trial.'''
    predictors = ['choice','second step','outcome','ch_x_ss','ch_x_out','ss_x_out']
    _representation_evolution(sessions, predictors, 'second step', fig_no)
    plt.figure(fig_no)
    plt.xlabel('Second step regression weight pre-outcome')
    plt.ylabel('Second step regression weight post-outcome')
    plt.tight_layout()

def second_step_by_trans_block_representation_evolution(sessions, fig_no=1):
    '''Plot the correlation between the representation of the second step
    state interacted with transtion block before and after the outcome is 
    recieved, and how the strength of these two representations changes
    across the trial.'''
    predictors=['choice','second step','outcome','ch_x_ss','ch_x_out','ss_x_out',
                'common_trans','ch_x_t_block','ss_x_t_block','trans_state']
    _representation_evolution(sessions, predictors, 'ss_x_t_block', fig_no)
    plt.figure(fig_no)
    plt.xlabel('Sec. step x trans. prob.\nregression weight pre-outcome')
    plt.ylabel('Sec. step x trans. prob.\nregression weight post-outcome')
    plt.tight_layout()

def _representation_evolution(sessions, predictors, predictor, fig_no):
    betas = regression_analysis(sessions, return_betas=True, predictors=predictors)
    betas = betas[:,:, predictors.index(predictor)]
    pre_ind, post_ind, choice_ind, outcome_ind = _get_pre_post_outcome_inds(sessions[0])
    betas_pre  = betas[:,pre_ind]
    betas_post = betas[:,post_ind]
    # Plot correlations between predictors.
    plt.figure(fig_no, figsize=[4,4]).clf()
    plt.axhline(0,linestyle=':',color='k')
    plt.axvline(0,linestyle=':',color='k')
    plt.plot(betas_pre, betas_post,'.')
    plt.title(f'R-squared: {_rsquared(betas_pre, betas_post):.4f}')    
    # Evaluate correlation only for abs(betas) > 1 at either time point.
    mask = np.logical_or(np.abs(betas_pre) > 1, np.abs(betas_post) > 1) 
    print(f'Fraction of neurons with |beta|>1: {np.mean(mask):.3f}')
    print(f'R-squared for |beta|>1: {_rsquared(betas_pre[mask], betas_post[mask]):.4f}')
    # Plot evolution of second step representation over time. Two-fold cross validation
    # is used to avoid selection bias.
    betas_1 = regression_analysis(sessions, return_betas=True, predictors=predictors, 
                partition='even')[:,:, predictors.index('second step')]
    betas_2 = regression_analysis(sessions, return_betas=True, predictors=predictors, 
                partition='odd' )[:,:, predictors.index('second step')]
    proj_vec_pre_1  = betas_1[:,pre_ind]  / np.linalg.norm(betas_1[:,pre_ind])
    proj_vec_pre_2  = betas_2[:,pre_ind]  / np.linalg.norm(betas_2[:,pre_ind])
    proj_vec_post_1 = betas_1[:,post_ind] / np.linalg.norm(betas_1[:,post_ind])
    proj_vec_post_2 = betas_2[:,post_ind] / np.linalg.norm(betas_2[:,post_ind])
    pre_projection  = (betas_1.T @ proj_vec_pre_2 ) + (betas_2.T @ proj_vec_pre_1 )
    post_projection = (betas_1.T @ proj_vec_post_2) + (betas_2.T @ proj_vec_post_1)
    plt.figure(str(fig_no)+'_', figsize=[4.3,4.6]).clf()
    _aligned_x_ticks(sessions[0].calcium_data['aligned'], tick_labels=True)
    t = sessions[0].calcium_data['aligned']['t_out']
    plt.ylabel('Projection')
    plt.plot(t, pre_projection , 'r', label='Pre')
    plt.plot(t, post_projection, 'b', label='Post')
    plt.axvline(t[choice_ind] ,linestyle=':',color='k')
    plt.axvline(t[outcome_ind],linestyle=':',color='k')
    ymin, ymax = plt.ylim()
    plt.scatter(t[pre_ind ], ymax, marker='v', color='r')
    plt.scatter(t[post_ind], ymax, marker='v', color='b')
    plt.xlim(t[0],t[-1])
    plt.legend()
    plt.tight_layout()

def _rsquared(X,Y):
    ols = LinearRegression().fit(X[:,None], Y[:,None])
    return ols.score(X[:,None], Y[:,None])

def _get_pre_post_outcome_inds(session):
    '''Get timepoint indicies corresponding to the mid point between
    choice and outcome and 250ms after outcome.'''
    align_samples = session.calcium_data['aligned']['align_samples']
    align_labels = session.calcium_data['aligned']['align_labels']
    choice_ind = align_samples[align_labels.index('Choice')]
    outcome_ind = align_samples[align_labels.index('Outcome')]
    pre_ind = int((choice_ind + outcome_ind)/2)
    post_ind = outcome_ind + int((outcome_ind - choice_ind)/2)
    return pre_ind, post_ind, choice_ind, outcome_ind
# ----------------------------------------------------------------------

def outcomes_AB_correlation(sessions, fig_no=1):
    '''Plot correlation between betas for outcomes obtained in in second steps A and B.''' 
    predictors = ['choice','second step','ch_x_ss','ch_x_out','outcome_A', 'outcome_B']
    betas = regression_analysis(sessions, return_betas=True, predictors=predictors)
    # Evalute outcome betas in 500ms post outcome.
    align_times = sessions[0].calcium_data['aligned']['align_times']
    align_labels = sessions[0].calcium_data['aligned']['align_labels']
    outcome_time = align_times[align_labels.index('Outcome')]
    t = sessions[0].calcium_data['aligned']['t_out']
    outcome_ind = np.argmin(np.abs(t - (outcome_time + 250)))
    # Plot correlations between predictors.
    betas_outcome_A = betas[:,outcome_ind, predictors.index('outcome_A')]
    betas_outcome_B = betas[:,outcome_ind, predictors.index('outcome_B')]
    plt.figure(fig_no, figsize=[4,4]).clf()
    plt.axhline(0,linestyle=':',color='k')
    plt.axvline(0,linestyle=':',color='k')
    plt.plot(betas_outcome_A, betas_outcome_B,'.')
    plt.xlabel('left outcome regression weight')
    plt.ylabel('right outcome regression weight')
    # Calculate r-squared
    ols = LinearRegression().fit(betas_outcome_A[:,None], betas_outcome_B[:,None])
    r2 = ols.score(betas_outcome_A[:,None], betas_outcome_B[:,None])
    plt.title(f'R-squared: {r2:.4f}')    
    plt.tight_layout()
    # Check for axis alignment.
    n_rota=1000
    min_d = 0.2
    d = np.sqrt(betas_outcome_A**2 + betas_outcome_B**2)
    included = d>min_d
    betas_outcome_A = betas_outcome_A[included]
    betas_outcome_B = betas_outcome_B[included]
    true_metric = _far_on_near_axis_distance(betas_outcome_A, betas_outcome_B)
    rota_metric = np.zeros(n_rota) # Array to store values under random rotations.
    for i, a in enumerate(2*np.pi*np.arange(0,1,1/n_rota)):
        rota_betas_A = np.cos(a)*betas_outcome_A - np.sin(a)*betas_outcome_B
        rota_betas_B = np.sin(a)*betas_outcome_A + np.cos(a)*betas_outcome_B
        rota_metric[i] = _far_on_near_axis_distance(rota_betas_A, rota_betas_B)
    print(f'Axis alignment P value: {np.mean(true_metric<rota_metric)}')

def _far_on_near_axis_distance(x, y):
    '''For a set of points defined by x and y coordinates, return the average 
    (over points) distance to the farther axis divided by the average distance
    to the closer axis.'''
    abs_xy = np.vstack([np.abs(x), np.abs(y)])
    return np.mean(np.max(abs_xy,0)/np.min(abs_xy,0))

# -------------------------------------------------------------------------------
# dPCA analysis 
# -------------------------------------------------------------------------------

def dPCA_analysis(session, labels='CS', regularizer=0, normalise=False):
    trial_labels = np.array([l for l in session.unpack_trial_data(labels)]).T.squeeze()
    aligned_spikes = session.calcium_data['aligned']['spikes']
    if normalise: 
        aligned_spikes = aligned_spikes/np.mean(aligned_spikes, axis=(0,2))[None,:,None]
    n_trials, n_neurons, n_timepoints = aligned_spikes.shape
    N = np.zeros([2]*len(labels)) # number of trials in each condition   
    X = np.zeros([n_neurons, n_timepoints] + [2]*len(labels))
    for trial_aligned_spikes, trial_label in zip(aligned_spikes,trial_labels):
        trial_label = tuple(trial_label) if type(trial_label) is np.ndarray else (trial_label,)
        X[(slice(None),slice(None))+trial_label] += trial_aligned_spikes
        N[trial_label] += 1
    X = X/N # Per condition average activity.
    # Center activity.
    ave_X = np.mean(X.reshape((X.shape[0],-1)),1)
    for i in range(1+len(labels)):
        ave_X = np.expand_dims(ave_X,1)
    X -= ave_X
    dpca = dPCA.dPCA(labels='T'+labels,regularizer=regularizer)
    Z = dpca.fit_transform(X)
    return Z, N

# -------------------------------------------------------------------------------
# Representational similarity analysis
# -------------------------------------------------------------------------------

def representational_similarity(sessions, fig_no=1, vmax=1, normalise=False, log=True, 
                                perm=False, save_dir=None):
    # Make masks for different trial epochs.  Note:  All sessions must use same time alignment.
    t = sessions[0].calcium_data['aligned']['t_out']
    median_latencies = sessions[0].calcium_data['aligned']['median_latencies']
    epoch_masks = { # Binary masks for different time epochs across trial.
        'choice'  : (-500 < t) & (t < 0),
        'secstep' : (   0 < t) & (t < median_latencies['CO']),
        'outcome' : (median_latencies['CO'] < t) & (t < median_latencies['CO'] + 500),
        'late'    : (median_latencies['CO']+500 < t) & (t < median_latencies['CO'] + 1000)}#,
        #'next_ch' : (median_latencies['CC_non']-500 < t) & (t < median_latencies['CC_non'])}
    cond_ave_activity =  {epoch: [] for epoch in epoch_masks.keys()} # To store condition average activity.
    if perm: # Create data structure to store permuted condition average activity.
        cond_ave_permuted = [{epoch: [] for epoch in epoch_masks.keys()} for i in range(perm)]
    for session in sessions:
        aligned_spikes = session.calcium_data['aligned']['spikes']
        n_trials = aligned_spikes.shape[0]
        if log:
            aligned_spikes = np.log2(aligned_spikes+0.01)
        # Make mask for different conditions, each condition is a combination of trial events.
        c, s, o = session.unpack_trial_data('CSO', bool)
        binary_conds = [o,s,c] # Trial events used combinatorially to construct conditions.
        cond_masks = []        # [n_condition, n_trials] indicating which trials correspond to each condition.
        for i in range(2**len(binary_conds)):
            cond_masks.append(np.all(np.vstack(
                [bool(i & 1<<j) == binary_conds[j] for j in range(len(binary_conds))]),0))
        cond_masks = np.vstack(cond_masks)[:,:n_trials]
        _session_cond_ave(cond_ave_activity, aligned_spikes, cond_masks, epoch_masks, normalise)
        if perm:
            for cond_ave_perm in cond_ave_permuted:
                np.random.shuffle(aligned_spikes) # Shuffle axis 0 only (trials)
                # aligned_spikes = np.roll(aligned_spikes,np.random.randint(n_trials), axis=0)
                _session_cond_ave(cond_ave_perm, aligned_spikes, cond_masks, epoch_masks, normalise)
    RSA = _RSA(cond_ave_activity, _RSM_regressors())
    # Permutation test for significance of predictor loadings.
    if perm:
        RSA_perm = [_RSA(cond_ave_perm, _RSM_regressors()) 
                    for cond_ave_perm in cond_ave_permuted]
        for epoch in epoch_masks:
            true_coefs = RSA[epoch]['coefs']
            perm_coefs = np.array([RSA_p[epoch]['coefs'] for RSA_p in RSA_perm])
            p_values = np.mean(perm_coefs > true_coefs,0) # One sided
            print('\nCoefficient P values for {} epoch:'.format(epoch))
            name_len = max([len(name) for name in RSA[epoch]['codes']])
            for i, name in enumerate(RSA[epoch]['codes']):
                print('  ' + name.ljust(name_len) + ': {:.3f}'.format(p_values[i+1]))
    # Plotting
    plt.figure(fig_no, figsize=[9,9]).clf()
    for i, epoch in enumerate(RSA):
        # Plot similarity matricies
        plt.subplot(3,4,i+1)
        plt.imshow(RSA[epoch]['rsm'], vmax=vmax, vmin=-vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(epoch)
        if i == 3:
            pos = plt.gca().get_position()
            cbar =  plt.colorbar(cax=plt.axes([pos.x1+0.02, pos.y0, 0.015, pos.height]))
        # Plot regression coeficients.
        plt.subplot(3,4,i+9)
        x = np.arange(len(RSA[epoch]['coefs'])-1)+1
        plt.bar(x, RSA[epoch]['coefs'][1:])
        plt.xticks(x, RSA[epoch]['codes'].keys(), rotation=-45, ha='left')
        plt.gca().axhline(0, color='k', linewidth=0.5)
        if i == 0: plt.ylabel('Regression loading')
    for i, (code_name, code) in enumerate(RSA[epoch]['codes'].items()):
        # Plot regressors.
        plt.subplot(3,len(RSA[epoch]['codes']), i+len(RSA[epoch]['codes'])+1)
        plt.imshow(code)
        plt.xticks([])
        plt.yticks([])
        plt.title(code_name)
    if save_dir: 
        plt.savefig(os.path.join(save_dir,'RSA.pdf'))

def _session_cond_ave(cond_ave_activity, aligned_spikes, cond_masks, epoch_masks, normalise):
    '''Evaluate the condition average activity for a session.'''
    for epoch in epoch_masks:
        epoch_activity = np.mean(aligned_spikes[:,:,epoch_masks[epoch]],2)                # [n_trials, n_neurons]
        epoch_cond_ave = np.vstack([np.mean(epoch_activity[cm,:],0)for cm in cond_masks]) # [n_conditions, n_neurons]
        if normalise: # Normalise each neurons mean and standard devation across conditions.
             epoch_cond_ave = (epoch_cond_ave-np.mean(epoch_cond_ave,0))/np.std(epoch_cond_ave,0)
        else:
            epoch_cond_ave = epoch_cond_ave-np.mean(epoch_cond_ave,0)
        cond_ave_activity[epoch].append(epoch_cond_ave)

def _RSA(cond_ave_activity, codes):
    '''Given the average activity in each condition and trial epoch calculate the similarity
    matrix for each epoch and regress them onto a set of candiate neural codes.'''
    # Setup regression model.
    X = np.vstack([p.flatten().astype(float) for p in [np.eye(8)] + list(codes.values())]).T # Predictor matrix [n_conditions**2, n_predictors]
    ols = LinearRegression()
    # Evaluate RSA matrix and regression coefficients for each epoch.
    RSA = {} # To store similarity matrices and regression coefs for each condition.
    for epoch in cond_ave_activity.keys():
        # Evaluate similarity matrix
        similarity_matrix = np.corrcoef(np.hstack(cond_ave_activity[epoch]))
        # Regress similarity matrix onto candidate codes.
        y = similarity_matrix.flatten() # [n_conditions**2]
        ols.fit(X,y)
        RSA[epoch] = {'rsm': similarity_matrix,
                      'coefs': ols.coef_,
                      'codes': codes}
    return RSA

def _RSM_regressors():
    '''Make a set of regressors each expressing the expected similarity matrix for a
    candidate neural code'''
    # Choice second step and outcome in each condition.
    c_mask = np.array([bool(i & 1<<2) for i in range(2**3)])
    s_mask = np.array([bool(i & 1<<1) for i in range(2**3)])
    o_mask = np.array([bool(i & 1<<0) for i in range(2**3)])
    # Choice, second step and outcome codes.
    c_code = c_mask[:,None] == c_mask[None,:]
    s_code = s_mask[:,None] == s_mask[None,:]
    o_code = o_mask[:,None] == o_mask[None,:]
    # Conjunctive codes.
    cs_code = c_code & s_code
    co_code = c_code & o_code
    so_code = s_code & o_code
    return OrderedDict(
        zip(['choice', 'sec. step', 'outcome','ch. & ss.', 'ch. & out.', 'ss. & out.'],
            [c_code  , s_code     , o_code   , cs_code   , co_code     , so_code     ]))

# -------------------------------------------------------------------------------
# Trajectory analysis
# -------------------------------------------------------------------------------

def trajectory_analysis(sessions, log=True, fig_no=1, PCs=[0,1,2], remove_nonspecific=True,
                        condition_independent=False, save_dir=None):
    '''Plot trajectories showing the average activity for each trial type defined by 
    choice, transition and outcome in a low dimensional space obtained by PCA on the
    data matrix [n_neurons, n_trial_types*n_timepoints].  If the remove_nonspecific 
    argument is True, the cross condition mean activity is subtrated from each conditions
    activity before PCA so only variance across trial types remains.
    '''
    # Extract average activity for each trial type.
    condition_ave_activity = []
    for session in sessions:
        aligned_spikes = session.calcium_data['aligned']['spikes']
        n_trials = aligned_spikes.shape[0]
        if log:
            aligned_spikes = np.log2(aligned_spikes+0.01)
        c = session.trial_data['choices'     ][:n_trials].astype(bool)
        s = session.trial_data['second_steps'][:n_trials].astype(bool)
        o = session.trial_data['outcomes'    ][:n_trials].astype(bool)
        trial_type_masks = [ c &  s &  o,
                             c &  s & ~o,
                             c & ~s &  o,
                             c & ~s & ~o,
                            ~c &  s &  o,
                            ~c &  s & ~o,
                            ~c & ~s &  o,
                            ~c & ~s & ~o]
        ses_cond_aves = np.concatenate([np.mean(aligned_spikes[ttm,:,:],0, keepdims=True)
                                        for ttm in trial_type_masks], 0) 
        condition_ave_activity.append(ses_cond_aves)
    condition_ave_activity = np.concatenate(condition_ave_activity,1)  # [trial_type, n_neurons, n_timepoint]
    if condition_independent: # Use mean across conditions as data matrix.
        X = np.mean(condition_ave_activity,0)
        
    else:
        if remove_nonspecific: # Subtract mean across conditions from each condition.
            condition_ave_activity = condition_ave_activity - np.mean(condition_ave_activity,0)
        X = np.hstack([caa for caa in condition_ave_activity]) # [n_neurons, n_timepoints*n_trial_types]
    # Do PCA.
    pca = PCA(n_components=12)
    pca.fit(X.T)
    # Plot trajectories for each trial type.
    fig = plt.figure(fig_no, figsize=[12, 12], clear=True)
    ax3D = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
    ax2Da = fig.add_subplot(3, 3, 3)
    ax2Db = fig.add_subplot(3, 3, 6)
    ax2Dc = fig.add_subplot(3, 3, 9)
    ax3D.set_xlabel('PC{}'.format(PCs[0]))
    ax3D.set_ylabel('PC{}'.format(PCs[1]))
    ax3D.set_zlabel('PC{}'.format(PCs[2]))
    ax2Da.set_xlabel('PC{}'.format(PCs[0]))
    ax2Da.set_ylabel('PC{}'.format(PCs[1]))
    ax2Db.set_xlabel('PC{}'.format(PCs[0]))
    ax2Db.set_ylabel('PC{}'.format(PCs[2]))
    ax2Dc.set_xlabel('PC{}'.format(PCs[1]))
    ax2Dc.set_ylabel('PC{}'.format(PCs[2]))
    labels = ['C:1 S:1 O:1', 'C:1 S:1 O:0', 'C:1 S:0 O:1', 'C:1 S:0 O:0',
              'C:0 S:1 O:1', 'C:0 S:1 O:0', 'C:0 S:0 O:1', 'C:0 S:0 O:0']
    colors = ['b','b','r','r','c','c','orange','orange']
    styles = ['-','--','-','--','-','--','-','--']
    mksn = session.calcium_data['aligned']['align_samples'] # Sample numbers in trajectory for event markers.
    for i, caa in enumerate([X] if condition_independent else condition_ave_activity):
        #3D plot
        traj_x = caa.T @ pca.components_[PCs[0],:]
        traj_y = caa.T @ pca.components_[PCs[1],:]
        traj_z = caa.T @ pca.components_[PCs[2],:]
        ax3D.plot3D(traj_x, traj_y, traj_z, color=colors[i], linestyle=styles[i], label=labels[i])
        # 2D projections.
        ax2Da.plot(traj_x, traj_y, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Db.plot(traj_x, traj_z, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Dc.plot(traj_y, traj_z, color=colors[i], linestyle=styles[i], label=labels[i])
        # Event markers.
        for j, sn, m in zip(range(4), mksn, ['$S$','$C$','$O$','$E$']):
            ax3D.scatter3D(traj_x[sn],traj_y[sn],traj_z[sn], color=colors[i], marker=m, s=80)
            ax2Da.scatter(traj_x[sn],traj_y[sn], color=colors[i], marker=m, s=80)
            ax2Db.scatter(traj_x[sn],traj_z[sn], color=colors[i], marker=m, s=80)
            ax2Dc.scatter(traj_y[sn],traj_z[sn], color=colors[i], marker=m, s=80)
    ax3D.legend()
    fig.tight_layout()
    if save_dir: 
        plt.savefig(os.path.join(save_dir,'trajectories, cond_ind={}, PCs={}.pdf'.format(
                                           condition_independent, PCs)))
    if condition_independent:
        plt.figure(fig_no+1, figsize=[6,8], clear=True)
        t = session.calcium_data['aligned']['t_out']
        for i in range(12):
            ax = plt.subplot(12,1,i+1)
            traj = X.T @ pca.components_[i,:]
            traj = (traj-np.mean(traj))
            plt.plot(t,traj)
            yl = ax.get_ylim()
            plt.xlim(t[0],t[-1])
            for x in session.calcium_data['aligned']['align_times'][1:-1]:
                ax.plot([x,x], yl,'k:')
            plt.ylabel('PC{}'.format(i+1))
            plt.xticks([])
        _aligned_x_ticks(session.calcium_data['aligned'], tick_labels=True)
        if save_dir: 
            plt.savefig(os.path.join(save_dir,'PC timecourses, PCs={}.pdf'.format(PCs)))

# -------------------------------------------------------------------------------
# Decoding analysis
# -------------------------------------------------------------------------------

def decoding_analysis(sessions, C=0.01, n_rep=10, fig_no=1):
    '''Assess how accurately different locations in the task state-action
    space can be decoded from the population activity.  The analysis defines
    10 different locations differentiated by the time relative to trial 
    events (pre choice, between choice and outcome, post outcome), and the 
    trial's choice, second-step state and outcome.  Over the course of each
    trial the subject moves through 3 of these locations. For each session
    a location vector 'l' and neuronal activity matrix 'A' are computed. To 
    combine neurons across session, we randomly select 10 visits to each
    location from each session, and contatenate activity from different
    sessions for the same locations. Location is decoded from the activity
    using multinomial regression, with L2 regularisation whose strength is
    determined by C. The analysis reports cross validated prediction accuracy
    using stratified Kfold cross validation with 10 folds. The cross 
    validated confusion matrix is plotted.  As the random sampling of 10
    visits to each location from each session introduces stochasiticity, the
    analysis is run n_rep times with different random samples and averaged.
    '''
    # Extract actitity from each session.
    ses_acts = [] # Neuronal activity matricies for each session.
    ses_locs = [] # State ID vectors for each session.
    for session in sessions:
        aligned_spikes = session.calcium_data['aligned']['spikes']
        n_trials = aligned_spikes.shape[0]
        n_neurons = aligned_spikes.shape[1]
        c = session.trial_data['choices'     ][:n_trials].astype(int)
        s = session.trial_data['second_steps'][:n_trials].astype(int)
        o = session.trial_data['outcomes'    ][:n_trials].astype(int)
        # Choice, second-step and outcome window inds.
        win_ch = [15,20]
        win_ss = [23,28]
        win_oc = [31,36]
        # Make data matricies
        l = np.zeros(n_trials*3, int)         # State space locations
        A = np.zeros([n_trials*3, n_neurons]) # Neuronal activity
        for t in range(n_trials):
            # Choice windows state and activity.
            A[3*t,:] = aligned_spikes[t, :, np.arange(*win_ch)].sum(axis=0)
            l[3*t] = c[t]
            # Second-step window state and activity.
            A[3*t+1,:] = aligned_spikes[t, :, np.arange(*win_ss)].sum(axis=0)
            l[3*t+1] = 2+c[t]+2*s[t]
            # Outcome window state and activity.
            A[3*t+2,:] = aligned_spikes[t, :, np.arange(*win_oc)].sum(axis=0)
            l[3*t+2] = 6+2*s[t]+o[t]
        A = scale(A) # Normalise and demean the activity
        ses_acts.append(A)
        ses_locs.append(l)
    # Combine neurons across sessions and run decoding analysis.
    state_counts = np.array([list(Counter(S).values()) for S in ses_locs])
    include_session = np.min(state_counts,1) >= 10
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',
            penalty='l2', C=C, fit_intercept=False, max_iter=1000)
    xval_scores = []
    xval_conf_mat = []
    cl = np.hstack([np.ones(10, int) * i for i in range(10)]) # Locations corresponding to the combined activity matrix.
    for i in range(n_rep): # Repeat analysis to average over random sampling of location visits.
        cA = [] # Combined activity matrix from multiple sessions.
        for i, (A, l) in enumerate(zip(ses_acts, ses_locs)):
            if include_session[i]:
                sA = [] # Randomly selected examples of each state for one session.
                for j in range(10): 
                    m = np.where(l==j)[0]
                    np.random.shuffle(m)
                    sA.append(A[m[:10],:])
                cA.append(np.vstack(sA))
        cA = np.hstack(cA)
        xval_scores.append(cross_val_score(clf, cA, cl, cv=10))
        cl_pred = cross_val_predict(clf, cA, cl, cv=10)
        xval_conf_mat.append(confusion_matrix(cl, cl_pred, normalize='true'))
    mean_scores = np.mean(xval_scores,0)
    xval_conf_mat = np.mean(np.stack(xval_conf_mat),0)
    print(f'Included {cA.shape[1]} neurons from the {np.sum(include_session)}'
           ' sessions with at least 10 visits to each state-space location')
    print(f'Cross val score: {np.mean(mean_scores):.2f} ± {sem(mean_scores):.2f}')
    # Plot confusion matrix.
    labels = ['ch._B.','ch._T.','ss_B.R.','ss_T.R.','ss_B.L.','ss_T.L.',
              'out_R.non.','out_R.rew.','out_L.non.','out_L.rew.']
    plt.figure(fig_no, figsize=[6,4], clear=True)
    ax = plt.axes()
    disp = ConfusionMatrixDisplay(xval_conf_mat, display_labels=labels)
    disp.plot(include_values=False, cmap='viridis', xticks_rotation='vertical', ax=ax)
    plt.tight_layout()
