#!/bin/bash
# Required parameters:
molecule_name="without_symmetry"
deltaEThreshold=5e-5
deltaESwitch=0.0
optCriteria="default"
# --calcAllFreq set all frequency calculations for all steps; is default 
# --no-calcAllFreq set not all frequency calculations for all steps;

# Check for more documentation with:
# > python3 MECISearch_FUNCTION.py --help
# Required files:
# - step_0_A.com, the com file of the initial point.
# - step_0_B.com, the com file of the initial point.
# - step_0_A.fchk, the fchk file of the initial point.
# - step_0_B.fchk, the fchk file of the initial point.

# To launch the python interface process:
nohup python3 ./MECISearch_FUNCTION.py --molecule $molecule_name --deltaEThreshold $deltaEThreshold --deltaESwitch $deltaESwitch --optCriteria $optCriteria --calcAllFreq --fromGeometry &
# nohup allows for the python script to run even if the terminal is closed or the ssh-link ended.
# TODO The script copy/paste its version at the time it is launched in the working directory.

# To kill the python interface process...
# Look for the PID related to python process.
# > top -u {user}
# Find PID of your optimization:
# > pwdx {PID1} {PID2}
# If PID1 is in the directory of your optimization, kill it.
# > kill {PID1} 

