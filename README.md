## Special Participation C — Refactor Summary

This refactor modularizes the original notebook into clean, reusable components while preserving all teaching value.

Structure
src/
 ├─ models.py       # Defines MLP and ScaledMLP architectures
 ├─ optimizers.py   # Implements SimpleAdam, SimpleAdamMuP, and SimpleShampoo
 └─ training.py     # Contains training loops, visualization utilities, and LR sweeps

models.py — contains the neural network definitions (MLP, ScaledMLP) with optional deterministic initialization and preserved TODO hooks for μP experiments.
optimizers.py — includes lightweight optimizer implementations (SimpleAdam, SimpleAdamMuP, SimpleShampoo) following the PyTorch API, with TODO sections intentionally left blank for assignment tasks.
training.py — encapsulates all training routines, evaluation helpers, and visualization functions (activation deltas, matrix norms, learning-rate sweeps) using the baseline Adam optimizer only.

Solution notebook is `q_mup_coding_sol.ipynb` that utilises files from `src_sol/` folder. 

### Key Improvements

Code quality: Added type hints, docstrings, and input validation (PEP 8/257).

Modularity: Separated models, optimizers, and training for reuse and testing.

Reproducibility: Centralized seeding, deterministic init options, safe device handling.

Pedagogy preserved: Same outputs, same hooks (μP / Shampoo TODOs), cleaner structure.

### Imports

from src.models import MLP, ScaledMLP
from src.optimizers import SimpleAdamMuP, SimpleShampoo, SimpleShampooScaled
from src.training import train_with_lr, set_seed


### References:
PEP 8 / 257 • Glorot & Bengio (2010) • He et al. (2015) • PyTorch docs on nn.Module, reset_parameters, manual_seed


# SpecialParticipationC
Because the code in our problems was evolved till it worked with specific deep-learning-concept related learning objectives, it is often not good code from the perspective of being exemplary from a software engineering  point of view. For example, it is often not very pythonic, etc. You can, ideally with AI assistance that you document carefully vis-a-vis process, take one of the coding problems/demos and refactor it as well as update the code to follow good documented software engineering and ML Engineering processes. Here, we expect you to give citations to the relevant points of good style and document your changes in a report. The constraint is that the problem code shouldn't lose any of its teaching value --- just be transformed to have good coding practices and style. As always, deconfliction is a must however this can be a group effort by up to three people. 
