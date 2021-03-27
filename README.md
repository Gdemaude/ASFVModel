# ASFVModel
Numba accelerated and multiprocessed simulation of the African Swine Fever Virus using Gillespie Algorithm implemented in the same way as this [project](https://github.com/Gdemaude/Gillespie).

This model is a replication of the model explained in [1] with a few differences: 

- Correction of an error in the original paper: the transition rate for exposed pigs is Beta * S * (I + epsilon * C) / N  instead of Beta * S * (I + epsilon * C) , with N the total of live pigs.
- No modelling of a vaccination plan for ASFV, only biosecurity measures are taken into account.
- Emphasis on the speed of the implementation to be able to run the model on a larger population (tens of thousands of pigs instead of 500) within a reasonnable amount of time.

#Implementation
To run the model on a larger scale while maintaining a reasonnable runtime, a few keypoints where necessary. Firstly, using Numba to accelerate the gillespie_direct() and the propensity function. This step alone allows the model to run 10 time faster than a normal python implementation.
