import pylab as plt
import numpy as np

from . import model_fitting as mf

#--------------------------------------------------------------------------------------
# Plotting model fits.
#--------------------------------------------------------------------------------------

def model_fit_plot(population_fit, fig_no=1, clf=True, col='b', x_offset=0., scatter=True,
                   ebars='pm95', title=None, half_height=False, sub_medians=True):
    ''' Plot the results of a population fit, for logistic regression fits all predictor
    loadings are plotted on single axis.  For RL fits where parameters are transformed
    to enforce constraints, seperate axes are used for parameters with different ranges.
    '''

    plt.figure(fig_no, figsize=[7,2.3])
    if clf:plt.clf()
    title = population_fit['agent_name'] if title is None else title
    plt.suptitle(title)

    def _plot(y, yerr, MAP_params, rng, param_names):
        n_ses, n_params = MAP_params.shape
        if ebars:
            plt.errorbar(np.arange(n_params)+0.5+x_offset, y, yerr, linestyle='',
                     linewidth=2, color=col, marker='_', markersize=10)
        else:
            plt.plot(np.arange(n_params)+0.5+x_offset, y, linestyle='', marker='.',
                     markersize=6, color=col)
        if scatter:
            for i in range(n_params):
                plt.scatter(i+0.4+x_offset+0.2*np.random.rand(n_ses), MAP_params[:,i],
                            s=4,  facecolor=col, edgecolors='none', lw=0)

        plt.xlim(0,n_params)
        plt.xticks(np.arange(n_params)+0.5, param_names, rotation=-45, ha='left')

        if rng == 'unc':
            plt.plot(plt.xlim(),[0,0],'k', linewidth=0.5)
        elif rng == 'unit': 
            plt.ylim(0,1)
        elif rng == 'pos':
            plt.ylim(0, max(plt.ylim()[1], np.max(MAP_params) * 1.2))

        plt.locator_params('y', nbins=5)

    pop_dists = population_fit['pop_dists']

    if sub_medians: # Plot subject median MAP fits rather than all session MAP fits.
        MAP_params = _sub_median_MAP(population_fit)
    else:
        MAP_params = np.array([sf['params_T'] for sf in population_fit['session_fits']])

    if ebars == 'SD': # Use population level distribution SDs for errorbars.
        ebars_U = pop_dists['SDs']
    elif ebars == 'pm95': # Use 95% confidence intervals of population distribution means.
        ebars_U = 1.96*np.sqrt(-1/population_fit['iBIC']['means_hessian'])
    else:
        ebars_U = np.zeros(len(pop_dists['SDs'])) 

    if (population_fit['param_ranges'][0] == 'all_unc' or
        all([pr == 'unc' for pr in population_fit['param_ranges']])): # Logistic regression fit.
        if half_height: plt.subplot(2,1,1)
        _plot(pop_dists['means'], ebars_U, MAP_params, 'unc', population_fit['param_names'])
        plt.ylabel('Log odds')

    else: # Reinforcement learning model fit.
        param_ranges = population_fit['param_ranges']

        #Transform parameters into constrained space.
        means_T = mf._trans_UT(pop_dists['means'], param_ranges)
        upp_1SD = mf._trans_UT(pop_dists['means'] + ebars_U, param_ranges)
        low_1SD = mf._trans_UT(pop_dists['means'] - ebars_U, param_ranges)
        yerr_T = np.vstack([means_T - low_1SD, upp_1SD - means_T])

        axes, ax_left = ([], 0.125)
        ax_bottom, ax_height = (0.53, 0.4) if half_height else (0.1, 0.8)
        for rng in ['unit', 'pos', 'unc']:
            rng_mask = np.array([r == rng for r in param_ranges])
            param_names = [p_name for p_name, rm in zip(population_fit['param_names'], rng_mask) if rm]
            subplot_MAP_params = MAP_params[:,rng_mask]   
            ax_width = np.mean(rng_mask) * 0.655  
            if rng == 'unc': ax_left += 0.02
            axes.append(plt.axes([ax_left, ax_bottom, ax_width, ax_height]))
            _plot(means_T[rng_mask], yerr_T[:,rng_mask], subplot_MAP_params, rng, param_names)
            ax_left = ax_left + ax_width + 0.05

        axes[0].set_ylabel('Parameter value')

def _sub_median_MAP(population_fit):
    'Return array of median MAP session fits per subject.'
    subject_IDs = set([sf['sID'] for sf in population_fit['session_fits']])
    subject_medians = np.zeros([len(subject_IDs), len(population_fit['param_names'])])
    for i,subject_ID in enumerate(subject_IDs):
        sub_session_fits = [sf for sf in population_fit['session_fits'] if sf['sID'] == subject_ID]
        subject_medians[i,:] = np.median([sf['params_T'] for sf in sub_session_fits],0)
    return subject_medians


def lagged_fit_plot(fit, fig_no=1, ebars='pm95', linestyle='-', xo=0,
                    cm=plt.cm.jet, clf=True, sub_MAP=True, title=None):
    'Model fit plot for logistic regression model with lagged predictors.'
    plt.figure(fig_no, figsize=[2.5,2.3])
    if clf: plt.clf()
    param_means = fit['pop_dists']['means']
    param_lags = np.array([int(pn.split('_lag_')[1].split('_')[0]) if len(pn.split('_lag_')) > 1 else -1
                           for pn in fit['param_names']])
    if ebars == 'SD': # Use population level distribution SDs for errorbars.
        ebars = fit['pop_dists']['SDs']
    elif ebars == 'pm95': # Use 95% confidence intervals of population distribution means.
        ebars = 1.96*np.sqrt(-1/fit['iBIC']['means_hessian'])
    subject_medians = _sub_median_MAP(fit)
    param_base = [pn.split('_lag_')[0] for pn in fit['param_names']]
    base_params = sorted(list(set(param_base)), key = param_base.index)
    color_idx = np.linspace(0, 1, len(base_params))
    for i, base_param in zip(color_idx, base_params):
        p_mask = np.array([pb == base_param for pb in param_base])
        if sum(p_mask) > 1:
            plt.errorbar(-param_lags[p_mask]+xo, param_means[p_mask], ebars[p_mask],
                         label=base_param, color=cm(i), linestyle=linestyle)
            sub_meds = subject_medians[:,p_mask]
            if sub_MAP:
                for j,lag in enumerate(-param_lags[p_mask]):
                    plt.scatter(xo+lag-0.1+0.2*np.random.rand(
                        sub_meds.shape[0]), sub_meds[:,j], s=4, 
                        facecolor=cm(i), edgecolors='none', lw=0)
    plt.plot([-max(param_lags) - 0.5, -0.5], [0,0], 'k', linewidth=0.5)
    plt.xlim([-max(param_lags) - 0.5, -0.5])
    plt.xticks(-param_lags[p_mask])
    plt.xlabel('Lag (trials)')
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0., fontsize = 'small')
    if title: plt.title(title)