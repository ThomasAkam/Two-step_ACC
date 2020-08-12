import numpy as np
import pylab as plt
from functools import partial

from . import utility as ut
from . import model_fitting as mf
from . import parallel_processing as pp

def BIC_model_comparison(sessions, agents, n_draws=1000, n_repeats=1, fig_no=1,
                         file_name=None, log_Y=False):
    ''' Compare goodness of different fits using integrated BIC.'''    
    if n_repeats > 1: 
        fit_func = partial(mf.repeated_fit_population, sessions, n_draws=n_draws, n_repeats=n_repeats)
        fiterator = map(fit_func, agents) # Save parallel processessing for repeated fits of same agent.
    else:
        fit_func = partial(mf.fit_population, sessions, eval_BIC={'n_draws':n_draws})
        fiterator = pp.imap(fit_func, agents, ordered=False) # Use parallel processing for seperate agents.
    population_fits = []
    for i,fit in enumerate(fiterator):
        print('Fit {} of {}, agent: '.format(i+1, len(agents)) + fit['agent_name'])
        population_fits.append(fit)
        if file_name: ut.save_item(population_fits, file_name)
    if fig_no: BIC_comparison_plot(population_fits, fig_no, log_Y)
    return population_fits

def BIC_comparison_plot(population_fits, fig_no=1, log_Y=False, plot_rep_fits=False):
    '''Plot the results of a BIC model comparison'''
    sorted_fits = sorted(population_fits, key = lambda fit: fit['iBIC']['score']) 
    print('BIC_scores:')
    for fit in sorted_fits:
        s =   '{:.3f} : '.format(fit['iBIC']['best_prob']) if 'best_prob' in fit['iBIC'].keys() else ''
        print('{:.0f} : '.format(round(fit['iBIC']['score'])) + s + fit['agent_name'])
    print('The best fitting model is: ' + sorted_fits[0]['agent_name'])
    if fig_no:
        BIC_scores = np.array([fit['iBIC']['score'] for fit in sorted_fits])
        BIC_deltas = BIC_scores - BIC_scores[0]
        agent_names = [fit['agent_name'] for fit in sorted_fits]
        x = np.flipud(np.arange(1,len(agent_names)+1))
        if 'BIC_95_conf' in fit['iBIC'].keys():
            ebars = np.array([np.abs(fit['iBIC']['BIC_95_conf'] - fit['iBIC']['score'])
                              for fit in sorted_fits]).T
        elif 'lik_95_conf'in fit['iBIC'].keys(): 
            ebars = -2*np.array([np.abs(fit['iBIC']['lik_95_conf'] - fit['iBIC']['int_lik'])
                                for fit in sorted_fits]).T
        else:
            ebars = None
        plt.figure(fig_no).clf()
        plt.bar(x, BIC_deltas, color = 'k')
        plt.errorbar(x, BIC_deltas, ebars, color = 'r', linestyle = '', elinewidth = 2)
        if 'repeated_fits' in population_fits[0].keys() and plot_rep_fits: # Scatter plot repeated fits.
            for fit, xi in zip(sorted_fits, x):
                rep_fit_iBICs = np.array([f['iBIC']['score'] for f in fit['repeated_fits']])
                plt.scatter(xi+0.4+np.linspace(-0.2,0.2,len(rep_fit_iBICs)), rep_fit_iBICs - BIC_scores[0])
        plt.xticks(x + 0.6/len(agent_names), agent_names, rotation = -45, ha = 'left')
        plt.xlim(0.25,len(agent_names)+1)
        plt.ylim(0, BIC_deltas[-1]*1.2)
        plt.ylabel('∆ BIC')
        plt.figtext(0.13,0.92,'Best BIC score: {}'.format(int(BIC_scores[0])))
        plt.tight_layout()
        if log_Y:
            plt.gca().set_yscale('log')
            plt.ylim(10,plt.ylim()[1])
     
#---------------------------------------------------------------------------------------
# Model calibration - evaluate real vs predicted choice probabilities.
#---------------------------------------------------------------------------------------

def eval_calibration(sessions, agent, population_fit, use_MAP=True, n_bins=10,
                     fixed_widths=False, to_plot=False):
    '''Caluculate real choice probabilities as function of model choice probabilities.'''

    session_fits = population_fit['session_fits']

    assert len(session_fits[0]['params_T']) == agent.n_params, 'agent n_params does not match population_fit.'
    assert len(sessions) == len(session_fits), 'Number of fits does not match number of sessions.'
    assert population_fit['agent_name'] == agent.name, 'Agent name different from that used for fits.'

    # Create arrays containing model choice probabilites and true choices for each trial.
    session_choices, session_choice_probs = ([],[])
    for fit, session in zip(session_fits, sessions):
        if use_MAP:
            params_T = fit['params_T']
        else:
            params_T = mf._sample_params_T(population_fit) 
        session_choices.append(session.trial_data['choices'])
        DVs = agent.session_likelihood(session, params_T, get_DVs = True)
        session_choice_probs.append(DVs['choice_probs'])

    choices = np.hstack(session_choices)
    choice_probs = np.hstack(session_choice_probs)[1,:]

    # Calculate true vs model choice probs.
    true_probs  = np.zeros(n_bins)
    model_probs = np.zeros(n_bins)
    if fixed_widths: # Bins of equal width in model choice probability.
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else: # Bins of equal trial number.
        choices = choices[np.argsort(choice_probs)]
        choice_probs.sort()
        bin_edges = choice_probs[np.round(np.linspace(0,len(choice_probs)-1,
                                 n_bins+1)).astype(int)]
        bin_edges[0] = bin_edges[0] - 1e-6 
    for b in range(n_bins):
        true_probs[b] = np.mean(choices[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
        model_probs[b] = np.mean(choice_probs[np.logical_and(
                            bin_edges[b] < choice_probs,
                            choice_probs <= bin_edges[b + 1])])
    
    calibration = {'true_probs': true_probs, 'choice_probs': model_probs}
    
    if to_plot: 
        calibration_plot(calibration, fig_no = to_plot) 

    print(('Fraction correct: {}'.format(sum((choice_probs > 0.5) == choices.astype(bool)) / len(choices))))
    chosen_probs = np.hstack([choice_probs[choices == 1], 1. - choice_probs[choices == 0]])
    print(('Geometric mean choice prob: {}'.format(np.exp(np.mean(np.log(chosen_probs))))))
    
    return calibration

def calibration_plot(calibration, clf=True, fig_no=1):
    if 'calibration' in list(calibration.keys()): #Allow population_fit to be passed in.
        calibration = calibration['calibration']
    plt.figure(fig_no)
    if clf:plt.clf()
    plt.plot(calibration['true_probs'], calibration['choice_probs'], 'o-')
    plt.plot([0,1],[0,1],'k',linestyle =':')
    plt.xlabel('True choice probability')
    plt.ylabel('Model choice probability')