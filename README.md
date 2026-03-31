# meci-without-nac-testing
Testing python interface to Gaussian16 for a MECI optimizer without the use of nonadiabatic coupling (NAC), using either only gradients and Lagrange Multipliers OR only gradients and numerical branching space vectors [Gonon, JCP, 2017]

## Available methods 
- SLM, [Sanz García, JCTC, 2024] using Lagrang Multiplier and updating the BS numerically
- Gradient descent with composed gradient, originally inspired from [Bearpark, CPL, 1994] and [Harvey, TCA, 1998]
    - Branching-Space Vectors are evaluated numerically using energy derivatives only [Gonon, JCP, 2017]
    - Tentatives toward Composed steps [Sicilia, JCTC, 2008] and Double Newton-Raphson [Ruiz-Barragan, JCTC, 2013]
## TODO-list
- [X] add home-code with quasi-Newton + numerical branching-space
- [ ] sort and clean-up most recent and tested versions [diff. between MECI_Optimization.py and MECISearch_FUNCTION.py]
- [ ] try implementing ALM version from [Sanz García, JCTC, 2024]
- [ ] merge SLM with home-code of quasi-Newton + numerical éranching-space
- [ ] upload tests cases [sym. and nosym. dummy molecules, m22, m23]
- [ ] interface with openMolcas for (MS,XMS,RMS)-CASPT2 energies and analytical gradients
- [ ] update references

## References
- [Sanz García, JCTC, 2024] Juan Sanz García, Rosa Maskri, Alexander Mitrushchenkov, and Loïc Joubert-Doriol, J. Chem. Theory Comput. 2024, 20, 13, 5643–5654, [https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00326][pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00326]
