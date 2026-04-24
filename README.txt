This repository contains the Python source code to replicate the plots of the model reported in "Evolutionary tuning of molecular charge state of UBP24 shapes responses to high temperature", by Shao-Li Yang et al. 

For a model description please see the Supplementary Information section of the paper.

Stomata_model_Shao_Li_Yang_et_al_updated_paramAnalysis.py can be run in a python environment. The standard python libraries 'numpy', 'scipy', 'os', 'matplotlib' are needed. Python installation in a normal computer takes between 5 and 15 minutes.


In the code a "Stomata" function is defined, containing ordinary differential equations describing how the activity of SNRK2S, TOT3, PP2C, AHA, UPB24 and SLAC1 are regulated (see details in Supp. Information in Yang et al). The parameters for the simulations are sampled from a uniform distribution within a 10% range of the predefined values. The Stomata function is used to run the following simulations: 

1) Wt-ost1-variants - Stomata aperture in control (21)and heat (42 degrees) conditions for Wt, ost1 loss of function mutant, and Wt variants without OST1->UPB24 regulation or OST1->TOT3 inactivation. 

2) Temperature x Stomata - Stomata aperture at 21, 28, 35, and 42 degrees. 

3) Heat_x_ABA - Stomata aperture in control, heat (H), ABA, and combined Heat and ABA conditions for Wt, ost1 and slac1 mutants.

Output: the output images are saved in an "output" folder in both png and svg formats.

The simulations reported in the manuscript were performed in a Linux machine (Ubuntu 24.04), and have also been run on MacOS operating system (Ventura 13.5). The model takes 10014.83 seconds (2 hours, 46 minutes) to run. 
