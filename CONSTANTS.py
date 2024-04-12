"""
This module defines project-level constants.
"""

PLANCK_CONSTANT=6.62607015e-34
LIGHT_SPEED=299792458

HARTREE_TO_EV=27.2114
BOHR_TO_ANGSTROM=0.529177210903 # bohr to angstrom, angstrom per bohr
AMU_TO_ME=1822.888486209 # amu to mass of the electron, me per amu (Dalton)
EV_TO_JOULE=1.602176634e-19
EV_TO_RCM=8065.545 # from IUPAC 2006
RNM_TO_RCM=10**7

EV_TO_NM=PLANCK_CONSTANT*LIGHT_SPEED/EV_TO_JOULE*10**9
CURVATURE_TO_FREQUENCY=HARTREE_TO_EV/EV_TO_NM*RNM_TO_RCM
HARTREE_TO_RCM=HARTREE_TO_EV/EV_TO_NM*RNM_TO_RCM

# BACKUP definitions of MECI.py
# amu2me=1822.888486209 # amu to mass of the electron, me per amu (Dalton)
# bohr2angstrom=0.529177210903 # bohr to angstrom, angstrom per bohr
# planckConstant=6.62607015e-34
# lightSpeed=299792458
# ev2joule=1.602176634e-19
# ev2nm=planckConstant*lightSpeed/ev2joule*10**9
# hartree2ev=27.2114
# rnm2rcm=10**7
# hartree2rcm=hartree2ev/ev2nm*rnm2rcm
# MWCurvature2Frequency=hartree2ev/ev2nm*rnm2rcm

