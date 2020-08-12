import os
import sys
import pylab as plt
from pathlib import Path

two_step_path = os.path.join(Path(__file__).resolve().parents[2], 'two_step')
sys.path.append(two_step_path)

import Two_step.model_plotting as mp
import Two_step.model_fitting as mf
import Two_step.parallel_processing as pp

from . import stim_analysis as sa
from . import logistic_regression as lr
from . import plotting as pl
from . import data_import as di

plt.ion()
plt.rcParams['pdf.fonttype'] = 42