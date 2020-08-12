This project contains data, analysis code and pyControl task definition code for manuscript:

**Anterior cingulate cortex represents action-state predictions and causally mediates model-based reinforcement learning in a two-step decision task.**

*Thomas Akam, Ines Rodrigues-Vaz, Ivo Marcelo, Xiangyu Zhang, Michael Pereira, Rodrigo Freire Oliveira, Peter Dayan, Rui M. Costa*

---

The analysis code folders contain scripts with functions to generate manuscript figures.  All analysis code is in Python 3, all required packages should come by default with the Anaconda Python distribution.

The pyControl task definition files specify the the two-step and probabilistic reversal learning tasks. For more information see: <http://pycontrol.readthedocs.io>

**Folder structure:**

- analysis code:
  
  - reversal_learning          :  Analysis code for the reversal learning task.
  - two_step                         : Analysis code for the two-step task.
  
- data: 

  - reversal_learning_task :  Behavioural data for the reversal learning task.
  - two_step_task                :  Behavioural and imaging data (CNMFe output) for two-step task.

- pyControl_code                     : pyControl task definition files for both tasks.   

  

