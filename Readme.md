This repository contains analysis code and pyControl task definition code for the manuscript:

**The anterior cingulate cortex predicts future states to mediate model-based action selection.**

*Thomas Akam, Ines Rodrigues-Vaz, Ivo Marcelo, Xiangyu Zhang, Michael Pereira, Rodrigo Freire Oliveira, Peter Dayan, Rui M. Costa*. 

Neuron 2020. 

---

The analysis code folders contain scripts with functions to generate manuscript figures.  All analysis code is in Python 3, all required packages should come by default with the Anaconda Python distribution.

The pyControl task definition files specify the the two-step and probabilistic reversal learning tasks. For more information see: <http://pycontrol.readthedocs.io>

To use the analysis code, download or clone this repository to obtain the folder *analysis_code*.   Download the file *data.zip* from [OSF](https://osf.io/8jwhm/) and unzip to obtain the folder *data*.  The analysis code expects the folders *analysis_code* and *data* to be in the same directory, giving the following folder structure:

- analysis code:
  
  - reversal_learning          :  Analysis code for the reversal learning task.
  - two_step                         : Analysis code for the two-step task.
  
- data: 

  - reversal_learning_task :  Behavioural data for the reversal learning task.
  - two_step_task                :  Behavioural and imaging data (CNMFe output) for two-step task.

  

  

