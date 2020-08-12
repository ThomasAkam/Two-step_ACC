from . import parallel_processing as pp
from . import utility as ut
from . import RL_agents as rl 
from . import plotting as pl 
from . import model_plotting as mp
from . import model_fitting as mf
from . import data_import as di
from . import logistic_regression as lr 
from . import model_comparison as mc 
from . import simulation as sm 
from . import stim_analysis as sa
from . import group_comparison as gc
from . import imaging_analysis as ia

pl.plt.ion()
pl.plt.rcParams['pdf.fonttype'] = 42
pl.plt.rc("axes.spines", top=False, right=False)