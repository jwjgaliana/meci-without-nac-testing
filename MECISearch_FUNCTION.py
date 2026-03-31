import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import sys,os
import time

import MECI as meci
import CONSTANTS as CST
import MECISearch_MODULE

def add_bool_arg(parser,name,default=False,doc="Bool type"):
    group=parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--'+name,dest=name,action='store_true',help=doc)
    group.add_argument('--no-'+name,dest=name,action='store_false',help=doc)
    parser.set_defaults(**{name:default})

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        """
        Program interfacing Gaussian Single-Point TD-DFT calculations with
        a (tentative) MECI optimizer
        Important features
        - Initial Optimizer uses Composed Gradient (of Intersection Space and Gradient Difference)
          but efforts toward Composed Step [Sicilia, JCTC, 2008] and Double Newton-Raphson [Ruiz-Barragan, JCTC, 2013]
          are ongoing
        - Branching-Space Vectors are evaluated numerically using energy derivatives only 
          [Gonon, JCP, 2017]
        """
        )
    parser.add_argument("--molecule",metavar="molecule",type=str,required=True,
                        help="Name/tag of the Molecule")
    parser.add_argument("--deltaEThreshold",metavar="deltaEThreshold",type=float,required=True,
                        help="Convergence Criterion for Energy Difference ΔE=0.5*(EB-EA) (Hartree)")
    parser.add_argument("--deltaESwitch",metavar="deltaESwich",type=float,required=False,default=0.0,
                        help="Switch for Energy Difference ΔE=0.5*(EB-EA) for Sicilia-like algorithm [Sicilia, JCTC, 2008]")
    parser.add_argument("--optCriteria",metavar="optCriteria",type=str,required=False,
                        choices=['tight','verytight','default','loose'],default='default',
                        help="Convergence Criteria Pre-Sets for Gradient and Displacements (similar to Gaussian16)")
    parser.add_argument("--trustRadius",metavar="trustRadius",type=float,required=False,default=1.0,
                        help="Trust Radius for Quasi-Newton Search Step")
    parser.add_argument("--rootA",metavar="rootA",type=int,required=False,default=1,
                        help="Index of excited state A")
    parser.add_argument("--rootB",metavar="rootB",type=int,required=False,default=2,
                        help="Index of excited state B")
    add_bool_arg(parser,'fromGeometry',default=False,doc="Switch on/off initialization from geometry only")
    add_bool_arg(parser,'calcAllFreq',default=True,doc="Switch on/off freq calculation at each step")
    add_bool_arg(parser,'bohr',default=True,doc="Switch on/off bohr units for Hessian update")
    add_bool_arg(parser,'massWeighting',default=False,doc="Switch on/off mass weighting for Hessian update")
    add_bool_arg(parser,'trustRegion',default=False,doc="Switch on/off trust radius update")

    args=parser.parse_args()

    molecule=args.molecule
    energyDifferenceThs=args.deltaEThreshold
    energyDifferenceSwitch=args.deltaESwitch
    optCriteria=args.optCriteria
    trust_radius=args.trustRadius
    rootA=args.rootA
    rootB=args.rootB
    roots=[rootA,rootB]

    fromGeometry=args.fromGeometry
    calcAllFreq=args.calcAllFreq
    bohr=args.bohr
    mw=args.massWeighting
    trust_region=args.trustRegion

    ####################################################
    # INTERFACE PARAMETERS                             #
    ####################################################
    sleep_buffer=5 # seconds 

    ####################################################
    # CALCULATION TYPE                                 #
    # Should ideally be defined withing ArgumentParser #
    ####################################################
    mw_initial_hessian=False
    if not calcAllFreq:
        energyThsForFreqUpdate=1e-3
        cfreq=5

    ####################################################
    # OPTIMIZATION PARAMETERS                          #
    ####################################################
    if optCriteria=='default':
        # for Gaussian/Opt equivalent, CAREFUL UNITS
        gradientMaxThs=4.5e-4              # Should be in Hartree/Bohr
        gradientRMSThs=3.0e-4              # Should be in Hartree/Bohr
        coordinatesDifferenceMaxThs=1.8e-3 # Should be in Bohr?
        coordinatesDifferenceRMSThs=1.2e-3 # Should be in Bohr?
    elif optCriteria=='tight':
        # for Gaussian/Opt=Tight equivalent, CAREFUL UNITS
        gradientMaxThs=1.5e-5              # Should be in Hartree/Bohr
        gradientRMSThs=1.0e-5              # Should be in Hartree/Bohr
        coordinatesDifferenceMaxThs=6.0e-5 # Should be in Bohr?
        coordinatesDifferenceRMSThs=4.0e-5 # Should be in Bohr?
    elif optCriteria=='verytight':
        # for Gaussian/Opt=VeryTight equivalent, CAREFUL UNITS
        gradientMaxThs=1.5e-6              # Should be in Hartree/Bohr
        gradientRMSThs=1.0e-6              # Should be in Hartree/Bohr
        coordinatesDifferenceMaxThs=6.0e-6 # Should be in Bohr?
        coordinatesDifferenceRMSThs=4.0e-6 # Should be in Bohr?
    elif optCriteria=='loose':
        # for Gaussian/Opt=VeryTight equivalent, CAREFUL UNITS
        gradientMaxThs=1.5e-3              # Should be in Hartree/Bohr
        gradientRMSThs=1.0e-3              # Should be in Hartree/Bohr
        coordinatesDifferenceMaxThs=6.0e-3 # Should be in Bohr?
        coordinatesDifferenceRMSThs=4.0e-3 # Should be in Bohr?
    if not bohr:
        gradientMaxThs/=CST.BOHR_TO_ANGSTROM
        gradientRMSThs/=CST.BOHR_TO_ANGSTROM
        coordinatesDifferencMaxThs*=CST.BOHR_TO_ANGSTROM
        coordinatesDifferencRMSThs*=CST.BOHR_TO_ANGSTROM
    convergenceTestThs=np.array([energyDifferenceThs,
                            gradientMaxThs,
                            gradientRMSThs,
                            coordinatesDifferenceMaxThs,
                            coordinatesDifferenceRMSThs,])
    cmax=np.inf

    ####################################################
    # HEADER FOR GAUSSIAN INPUT (INTERFACE)            #
    ####################################################
    header_template=[]
    header_template.append("%chk=%FILE_TAG%_%STEP_NUMBER%_%ROOT_TAG%.chk")
    header_template.append("%mem=16GB")
    header_template.append("%nprocshared=16")
    header_template.append("# pop=(full) %DERIVATIVE_CALCULATION% cam-b3lyp/6-31+g(d) integral=grid=ultrafine scf=(conver=10,novaracc) td=(root=%ROOT_NUMBER%,nstates=10) nosym")
    header_template.append("")
    header_template.append("%TITLE%")
    header_template.append("")
    header_template.append("0 1")
    header_template="\n".join(header_template)
    header_template=header_template.replace("%FILE_TAG%","step")
    ####################################################
    # INITIALIZATION                                   #
    ####################################################
    energyDifferenceFlag=False
    gradientMaxFlag=False
    gradientRMSFlag=False
    coordinatesDifferenceMaxFlag=False
    coordinatesDifferenceRMSFlag=False
    convergenceTest=np.array([energyDifferenceFlag,
                            gradientMaxFlag,
                            gradientRMSFlag,
                            coordinatesDifferenceMaxFlag,
                            coordinatesDifferenceRMSFlag,])
    c=0
    currentCoordinatesDifference=None
    currentGradient=None
    currentHessian=None

    pflog=open("ProgressionFile.log","w")
    hessianlog=open("HessianFile.log","w")
    pflog.write("Energy Difference Convergence Criterion {}".format(energyDifferenceThs)+"\n")
    pflog.write("Energy Difference Switching   Threshold {}".format(energyDifferenceSwitch)+"\n")
    pflog.write("Maximum Force Convergence Criterion     {}".format(gradientMaxThs)+"\n")
    pflog.write("RMS Force Convergence Criterion         {}".format(gradientRMSThs)+"\n")
    pflog.write("Maximum Disp. Convergence Criterion     {}".format(coordinatesDifferenceMaxThs)+"\n")
    pflog.write("RMS Disp. Convergence Criterion         {}".format(coordinatesDifferenceRMSThs)+"\n")
    if mw:
        trust_radius*=np.sqrt(CST.AMU_TO_ME)
    pflog.write("Trust Radius for Quasi-Newton Search Step {}\n".format(trust_radius))
    ####################################################
    # INITIAZILATION FROM GEOMETRY                     #
    # If starting point has no freq calc               #
    ####################################################
    if fromGeometry:
        pflog.write("Only initial geometry is given\n")
        pflog.write("Computing single-point at initial geometry\n")
        comfilenames=["step_0_A.com","step_0_B.com"]

        header_A=header_template
        header_A=header_A.replace("%STEP_NUMBER%","0")
        header_A=header_A.replace("%ROOT_TAG%","A")
        header_A=header_A.replace("%ROOT_NUMBER%",str(rootA))
        header_A=header_A.replace("%DERIVATIVE_CALCULATION%","freq=(savenm,hpmodes)")
        header_A=header_A.replace("%TITLE%","step 0 frequency calculation for state A")
        MECISearch_MODULE.writeComFileFromXYZ(comfilenames[0],header_A,"step_0.xyz")

        header_B=header_template
        header_B=header_B.replace("%STEP_NUMBER%","0")
        header_B=header_B.replace("%ROOT_TAG%","B")
        header_B=header_B.replace("%ROOT_NUMBER%",str(rootB))
        header_B=header_B.replace("%DERIVATIVE_CALCULATION%","freq=(savenm,hpmodes)")
        header_B=header_B.replace("%TITLE%","step 0 frequency calculation for state B")
        MECISearch_MODULE.writeComFileFromXYZ(comfilenames[1],header_B,"step_0.xyz")
       
        pflog.write("Computing step_{}".format(c)+"\n")
        pflog.write("com files: "+"\t".join(comfilenames)+"\n")
        os.system("subg16a03 {}".format(comfilenames[0]))
        os.system("subg16a03 {}".format(comfilenames[1]))
        finishedA=False
        finishedB=False
        logfilenames=[comfilenames[0].split(".")[0]+".log",comfilenames[1].split(".")[0]+".log"]
        pflog.write("log files: "+"\t".join(logfilenames)+"\n")

        pflog.close()
        pflog=open("ProgressionFile.log","a+")

        while not finishedA or not finishedB:
            time.sleep(sleep_buffer)
            if os.path.isfile(logfilenames[0]):
                with open(logfilenames[0],"r") as f:
                    last_line=f.readlines()[-1].split()
                    if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                        finishedA=True
            if os.path.isfile(logfilenames[1]):
                with open(logfilenames[1],"r") as f:
                    last_line=f.readlines()[-1].split()
                    if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                        finishedB=True
            time.sleep(sleep_buffer)
        chkfilenames=[comfilenames[0].split(".")[0]+".chk",comfilenames[1].split(".")[0]+".chk"]
        pflog.write("chk files: "+"\t".join(chkfilenames)+"\n")
        os.system("formchk16 {}".format(chkfilenames[0]))
        os.system("rm {}".format(chkfilenames[0]))
        os.system("formchk16 {}".format(chkfilenames[1]))
        os.system("rm {}".format(chkfilenames[1]))
        fchkfilenames=[comfilenames[0].split(".")[0]+".fchk",comfilenames[1].split(".")[0]+".fchk"]
        # print(stop)

    ####################################################
    # SYSTEM DEFINITION                                # 
    ####################################################
    NAtoms,NCoords,atomicNumbers,initialCoordinates=meci.fchk2coordinates("step_0_A.fchk") 
    initialCoordinates=initialCoordinates.reshape(NAtoms,3)
    pflog.write("Initial Coordinates (Angstrom)\n")
    to_print=pd.DataFrame(initialCoordinates,index=atomicNumbers)
    to_print.columns=['x','y','z']
    pflog.write(to_print.to_string())
    pflog.write("\n")
    pflog.close()
    hessianlog.close()
    ####################################################
    # OPTIMIZATION LOOP                                #
    ####################################################
    while not np.all(convergenceTest) and c<cmax:
        pflog=open("ProgressionFile.log","a+")
        hessianlog=open("HessianFile.log","a+")
        pflog.write("step {}".format(c)+"\n")

        #################################################
        # GET SYSTEM INFORMATION AND ENERGY DERIVATIVES #
        #################################################
        ## Modifiers are:
        ## - freq
        ## - mw
        NAtoms,NCoords,atomicNumbers,currentCoordinates=meci.fchk2coordinates("step_{}_A.fchk".format(c))
        atomicNumbers3=np.array([[_]*3 for _ in atomicNumbers]).flatten()
        atomicMasses3=meci.fchk2derivatives("step_{}_A.fchk".format(c),freq=False)[2]
        atomicMasses3*=CST.AMU_TO_ME
        if not calcAllFreq:
            if c==0:
                freq=True
            elif c!=0 and (c%cfreq==0 or currentEnergyDifference<energyThsForFreqUpdate):
                freq=True
            elif c%cfreq!=0:
                freq=False
        elif calcAllFreq:
            freq=True
        if freq:
            pflog.write("step {} was a Freq Calculation".format(c)+"\n")
            currentEnergyA,currentGradientA,currentHessianA=MECISearch_MODULE.getStateDerivatives("step_{}_A.fchk".format(c),mw=mw,freq=freq)
            currentEnergyB,currentGradientB,currentHessianB=MECISearch_MODULE.getStateDerivatives("step_{}_B.fchk".format(c),mw=mw,freq=freq)
        else:
            pflog.write("step {} was a Force-only Calculation".format(c)+"\n")
            currentEnergyA,currentGradientA=MECISearch_MODULE.getStateDerivatives("step_{}_A.fchk".format(c),mw=mw,freq=freq)
            currentEnergyB,currentGradientB=MECISearch_MODULE.getStateDerivatives("step_{}_B.fchk".format(c),mw=mw,freq=freq)
        if bohr:
            currentCoordinates/=CST.BOHR_TO_ANGSTROM
            currentGradientA*=CST.BOHR_TO_ANGSTROM
            currentGradientB*=CST.BOHR_TO_ANGSTROM
            if freq:
                currentHessianA*=(CST.BOHR_TO_ANGSTROM**2)
                currentHessianB*=(CST.BOHR_TO_ANGSTROM**2)
        currentEnergyDifference=0.5*(currentEnergyB-currentEnergyA)
        currentEnergyAverage=0.5*(currentEnergyA+currentEnergyB)
        currentGradientDifference=0.5*(currentGradientB-currentGradientA)
        currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))
        currentGradientAverage=0.5*(currentGradientA+currentGradientB)
        if freq:
            currentHessianDifference=0.5*(currentHessianB-currentHessianA)
            currentHessianAverage=0.5*(currentHessianA+currentHessianB)

        pflog.close()
        pflog=open("ProgressionFile.log","a+")

        ###################################
        # COMPUTE BRANCHING SPACE VECTORS #
        # And associated projectors       #
        ###################################
        ## Previous modifiers influence:
        ## - mass-weighting or not the BSV
        currentHessianSquaredEnergyDifference=(
                2*currentEnergyDifference*currentHessianDifference
                +
                2*np.tensordot(currentGradientDifference,currentGradientDifference,axes=0)
                )
        # IDEALLY, Diagonalization of the previous matrix leads to two non-zero eigenvalues
        # for which the (orthogonal and unitary) eigenvectors span the branching space
        # IDEALLY: because this is valid close to degeneracy (say [1e-7;1e-4])
        # TODO: check away from degeneracy, there should be some non-negligible negative eigenvalues
        eigenvalues,eigenvectors=scipy.linalg.eigh(currentHessianSquaredEnergyDifference)
        currentBranchingSpaceVectorLengths=eigenvalues[-2:] # Index Zero is the lowest, One the highest
        currentBranchingSpaceVectors=eigenvectors.T[-2:] # Index Zero is the lowest, One the highest
        currentBranchingSpaceProjector=(
                np.tensordot(currentBranchingSpaceVectors[0],currentBranchingSpaceVectors[0],axes=0)
                +
                np.tensordot(currentBranchingSpaceVectors[1],currentBranchingSpaceVectors[1],axes=0)
                )
        currentIntersectionSpaceProjector=(
                np.eye(NCoords)
                -currentBranchingSpaceProjector
                )
        currentProjectedGradientAverage=np.dot(currentBranchingSpaceProjector,currentGradientAverage)
        currentProjectedOutGradientAverage=np.dot(currentIntersectionSpaceProjector,currentGradientAverage)
        ## Also works, same thing
        ## currentProjectedOutGradientAverage=(
                ## currentGradientAverage
                ## -np.dot(currentGradientAverage,currentBranchingSpaceVectors[0])*currentBranchingSpaceVectors[0]
                ## -np.dot(currentGradientAverage,currentBranchingSpaceVectors[1])*currentBranchingSpaceVectors[1]
                ## ) 
        #######################################
        # DEFINITION OF NEXT STEP TYPE        #
        #######################################
        ## Modifiers are: 
        ## - energyDifferenceSwitch 
        ##   (if 0, always a composed_gradient)
        ##   (if np.inf, always a composed_step)
        if currentEnergyDifference>=energyDifferenceSwitch:
            composed_gradient=True
            composed_step=False
        elif currentEnergyDifference<energyDifferenceSwitch:
            composed_gradient=False
            composed_step=True
        #######################################
        # COMPUTE OBJECTIVE FUNCTION GRADIENT #
        #######################################
        ## Modifiers are:
        ## - composed_gradient
        ## - [ ] TODO composed_step     ## mutually exclusive!
        ## - [ ] TODO Change Objective Function (upper, lower and average surfaces)
        if composed_gradient and not composed_step:
            pflog.write("Composed Gradient\n")
            pflog.write("Objective Gradient ~= Gradient Intersection Space + Gradient Difference\n")
            currentGradient=(
                    currentProjectedOutGradientAverage
                    +
                    2*currentEnergyDifference*currentGradientDifference/currentGradientDifferenceNorm
                    )
        elif composed_step and not composed_gradient:
            pflog.write("Composed Step\n")
            pflog.write("Objective Gradient ~= Gradient Intersection Space\n")
            currentGradient=np.copy(currentProjectedOutGradientAverage)
        #######################################
        # COMPUTE OBJECTIVE FUNCTION HESSIAN  #
        # Hessian Initialization and Update   #
        #######################################
        ##################
        # Initialization # 
        ##################
        ## Modifiers are:
        ## - mw_initial_hessian 
        ##   [ ] TODO get this compatible with mw=bool and trust_radius
        ##########
        # Update # 
        ##########
        ## Modifiers:
        ## - for now, only BFGS
        ## - [ ] TODO implement direct formula for BFGS inverse
        if c==0:
            currentHessian=np.eye(NCoords,NCoords)
            if mw_initial_hessian:
                for i in range(NCoords):
                    for j in range(NCoords):
                        currentHessian[i,j]=currentHessian[i,j]/(np.sqrt(atomicNumbers3[i])*np.sqrt(atomicNumbers3[j]))
                        ## currentHessian[i,j]=currentHessian[i,j]/(np.sqrt(atomicMasses3[i])*np.sqrt(atomicMasses3[j]))
                ####
                #### currentHessian/=np.max(currentHessian)
        else:
            currentHessian=MECISearch_MODULE.BFGSUpdate(previousCoordinatesDifference,previousGradient,previousHessian,currentGradient)

        ###########################################
        # COMPUTE DIRECTION OF INTERSECTION SPACE #
        ###########################################
        # Remark: minus or not minus here? not essential because we norm and follow best step...
        # directionIntersectionSpace=-np.dot(scipy.linalg.inv(currentHessian),currentGradient)
        directionIntersectionSpace=np.dot(scipy.linalg.inv(currentHessian),currentGradient)
        unitaryDirectionIntersectionSpace=directionIntersectionSpace/np.sqrt(np.sum(directionIntersectionSpace**2))

        ##################################################### 
        # COMPUTE STEP SIZE TO FOLLOW IN INTERSECTION SPACE # 
        # With a quadratic search along the NR direction    # 
        ##################################################### 
        step_direction=np.copy(directionIntersectionSpace)
        step_direction=step_direction/np.sqrt(np.sum(step_direction**2))
        def to_minimize(step_size):
            coordinatesDifference=step_size*step_direction
            estimate=currentEnergyAverage+np.dot(currentGradient,coordinatesDifference)+0.5*np.dot(coordinatesDifference,np.dot(currentHessian,coordinatesDifference))
            return estimate
        ## Defining constraints for a particular trust region #
        ## Modifiers:
        ## - trust_size
        ## - trust_RMS_and_MAX
        ## - [ ] TODO make both possible and mutually exclusive
        ## - RMK: works same way for both if everything is working...
        def constraint_size(step_size):
            return trust_radius-np.abs(step_size)
        cons={'type': 'ineq','fun': constraint_size}
        ## def constraint_RMS(step_size):
            ## coordinatesDifference=step_size*step_direction
            ## return trust_radius-np.sqrt(np.sum(coordinatesDifference**2)/NCoords)
        ## def constraint_MAX(step_size):
            ## coordinatesDifference=step_size*step_direction
            ## return trust_radius-np.max(np.abs(coordinatesDifference))
        ## cons={'type': 'ineq','fun': constraint_RMS,
              ## 'type': 'ineq','fun': constraint_MAX}
        x0=[-1.0]
        pflog.write("'Guess' step size for NR-step in Intersection Space = {}\n".format(x0))
        res=scipy.optimize.minimize(to_minimize,x0,constraints=cons)
        step_size=res.x
        pflog.write("Predicted step size for NR-step in Intersection Space = {}\n".format(step_size))
        currentCoordinatesDifference=step_size*step_direction
        pflog.write("Associated Max. = {}\n".format(np.max(np.abs(currentCoordinatesDifference))))
        pflog.write("Associated RMS  = {}\n".format(np.sqrt(np.sum(currentCoordinatesDifference**2)/NCoords)))
        estimateDifference=to_minimize(step_size)-to_minimize(0)
        pflog.write("Estimate Difference in Average Energy for NR-step in Intersection Space = {}\n".format(estimateDifference))

        pflog.close()
        pflog=open("ProgressionFile.log","a+")

        if composed_step and not composed_gradient:
            pflog.write("Adding Second Displacement (Composed Step) -(ΔE/NormGD)*(GD/NormGD)\n")
            gradientDifferenceDisplacement=-1.0*(currentEnergyDifference/currentGradientDifferenceNorm)*(currentGradientDifference/currentGradientDifferenceNorm)
            currentCoordinatesDifference+=gradientDifferenceDisplacement

        ####################################
        # COMPUTE COORDINATES OF NEXT STEP #
        # For electronic structure         #
        ####################################
        ## Modifiers:
        ## - mw
        if mw:
            nextCoordinates=currentCoordinates+currentCoordinatesDifference/np.sqrt(atomicMasses3)
        elif not mw:
            nextCoordinates=currentCoordinates+currentCoordinatesDifference
        # print("currentCoordinates",currentCoordinates)
        # print("nextCoordinates",nextCoordinates)
        if bohr:
            currentCoordinates*=CST.BOHR_TO_ANGSTROM
            nextCoordinates*=CST.BOHR_TO_ANGSTROM
            # print("nextCoordinates",nextCoordinates)
        nextCoordinates=nextCoordinates.reshape(NAtoms,3).astype(str)
        nextCoordinates=np.append(nextCoordinates,[[atomicNumber] for atomicNumber in atomicNumbers],axis=1)
        nextCoordinates=np.roll(nextCoordinates,1,axis=1)
        ###############################################
        # CONVERGENCE CHECK BEFORE ELEC. STRUCT. CALC #
        ###############################################
        energyDifferenceFlag=currentEnergyDifference<energyDifferenceThs
        gradientMax=np.max(np.sqrt(currentGradient**2))
        gradientRMS=np.sqrt((1/currentGradient.size)*np.sum(currentGradient**2))
        coordinatesDifferenceMax=np.max(np.sqrt(currentCoordinatesDifference**2))
        coordinatesDifferenceRMS=np.sqrt((1/currentCoordinatesDifference.size)*np.sum(currentCoordinatesDifference**2))
        convergenceTestValues=np.array([currentEnergyDifference,
                                gradientMax,
                                gradientRMS,
                                coordinatesDifferenceMax,
                                coordinatesDifferenceRMS,])

        gradientMaxFlag=gradientMax<gradientMaxThs
        gradientRMSFlag=gradientRMS<gradientRMSThs
        coordinatesDifferenceMaxFlag=coordinatesDifferenceMax<coordinatesDifferenceMaxThs
        coordinatesDifferenceRMSFlag=coordinatesDifferenceRMS<coordinatesDifferenceRMSThs
        convergenceTest=np.array([energyDifferenceFlag,
                                gradientMaxFlag,
                                gradientRMSFlag,
                                coordinatesDifferenceMaxFlag,
                                coordinatesDifferenceRMSFlag,])
        pflog.write("Total Energy SA {}\n".format(currentEnergyA))
        pflog.write("Total Energy SB {}\n".format(currentEnergyB))
        pflog.write("Energy Average  {}\n".format(currentEnergyAverage))
        pflog.write("Energy Mean     {}\n".format(currentEnergyAverage))
        to_print=pd.DataFrame({"Value":convergenceTestValues,"Threshold":convergenceTestThs,"Converged?":convergenceTest},index=["Energy  Difference","Maximum Force","RMS     Force","Maximum Displacement","RMS     Displacement"])
        pflog.write(to_print.to_string())
        pflog.write("\n")
        if c==0:
            pflog.write("Current Coordinates (Angstrom) at step {}\n".format(c))
            to_print=pd.DataFrame(initialCoordinates,index=atomicNumbers)
            to_print.columns=['x','y','z']
            pflog.write(to_print.to_string())
            pflog.write("\n")
        else:
            pflog.write("Current Coordinates (Angstrom) at step {}\n".format(c))
            to_print=pd.DataFrame(currentCoordinates.reshape(NAtoms,3),index=atomicNumbers)
            to_print.columns=['x','y','z']
            pflog.write(to_print.to_string())
            pflog.write("\n")
        if bohr:
            pflog.write("Current Gradient for Optimization (Eh/Bohr) at step {}\n".format(c))
        else:
            pflog.write("Current Gradient for Optimization (Eh/Angstrom) at step {}\n".format(c))
        to_print=pd.DataFrame(currentGradient.reshape(NAtoms,3),index=atomicNumbers)
        to_print.columns=['x','y','z']
        pflog.write(to_print.to_string())
        pflog.write("\n")
        if bohr:
            pflog.write("Gradient Energy Difference (Eh/Bohr) at step {}\n".format(c))
        else:
            pflog.write("Gradient Energy Difference (Eh/Angstrom) at step {}\n".format(c))
        to_print=pd.DataFrame(currentGradientDifference.reshape(NAtoms,3),index=atomicNumbers)
        to_print.columns=['x','y','z']
        pflog.write(to_print.to_string())
        pflog.write("\n")
        if bohr:
            hessianlog.write("Hessian Squared Energy Difference (updated) (Eh^2/Bohr^2) at step {}\n".format(c))
        else:
            hessianlog.write("Hessian Squared Energy Difference (updated) (Eh^2/Angstrom^2) at step {}\n".format(c))
        MECISearch_MODULE.writeTriangularMatrix(currentHessian,hessianlog)
        hessianlog.write("\n")
        hessianlog.close()
        if not np.all(convergenceTest):
            c+=1
            if not calcAllFreq:
                if (c%cfreq==0 or currentEnergyDifference<energyThsForFreqUpdate):
                    freq=True
                elif c%cfreq!=0:
                    freq=False
            if freq:
                derivative_calculation="freq=(savenm,hpmodes)"
            elif not freq:
                derivative_calculation="Force"

            comfilenames=["step_{}_A.com".format(c),"step_{}_B.com".format(c)]
            header_A=header_template
            header_A=header_A.replace("%STEP_NUMBER%",str(c))
            header_A=header_A.replace("%ROOT_TAG%","A")
            header_A=header_A.replace("%ROOT_NUMBER%",str(rootA))
            header_A=header_A.replace("%DERIVATIVE_CALCULATION%",derivative_calculation)
            header_A=header_A.replace("%TITLE%","step {} frequency calculation for state A".format(c))
            header_B=header_template
            header_B=header_B.replace("%STEP_NUMBER%",str(c))
            header_B=header_B.replace("%ROOT_TAG%","B")
            header_B=header_B.replace("%ROOT_NUMBER%",str(rootB))
            header_B=header_B.replace("%DERIVATIVE_CALCULATION%",derivative_calculation)
            header_B=header_B.replace("%TITLE%","step {} frequency calculation for state B".format(c))
            MECISearch_MODULE.writeComFile(comfilenames[0],header_A,nextCoordinates)
            MECISearch_MODULE.writeComFile(comfilenames[1],header_B,nextCoordinates)

            pflog.write("Computing step_{}".format(c)+"\n")
            pflog.write("com files: "+"\t".join(comfilenames)+"\n")
            os.system("subg16a03 {}".format(comfilenames[0]))
            os.system("subg16a03 {}".format(comfilenames[1]))
            finishedA=False
            finishedB=False
            logfilenames=[comfilenames[0].split(".")[0]+".log",comfilenames[1].split(".")[0]+".log"]
            pflog.write("log files: "+"\t".join(logfilenames)+"\n")

            pflog.close()
            pflog=open("ProgressionFile.log","a+")

            while not finishedA or not finishedB:
                time.sleep(sleep_buffer)
                if os.path.isfile(logfilenames[0]):
                    with open(logfilenames[0],"r") as f:
                        last_line=f.readlines()[-1].split()
                        if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                            finishedA=True
                if os.path.isfile(logfilenames[1]):
                    with open(logfilenames[1],"r") as f:
                        last_line=f.readlines()[-1].split()
                        if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                            finishedB=True
                time.sleep(sleep_buffer)
            chkfilenames=[comfilenames[0].split(".")[0]+".chk",comfilenames[1].split(".")[0]+".chk"]
            pflog.write("chk files: "+"\t".join(chkfilenames)+"\n")
            os.system("formchk16 {}".format(chkfilenames[0]))
            os.system("rm {}".format(chkfilenames[0]))
            os.system("formchk16 {}".format(chkfilenames[1]))
            os.system("rm {}".format(chkfilenames[1]))
            fchkfilenames=[comfilenames[0].split(".")[0]+".fchk",comfilenames[1].split(".")[0]+".fchk"]
        nextEnergyA=meci.fchk2derivatives(fchkfilenames[0],mw=mw,freq=freq)[0]
        nextEnergyB=meci.fchk2derivatives(fchkfilenames[1],mw=mw,freq=freq)[0]
        nextEnergyDifference=nextEnergyB-nextEnergyA
        nextEnergyAverage=0.5*(nextEnergyA+nextEnergyB)
        actualDifference=nextEnergyAverage-currentEnergyAverage
        pflog.write("Actual Difference in Average Energy after Composed-step in IS-BS = {}\n".format(actualDifference))
        # TODO: test this trust_region again
        if trust_region:
            if ratio>0 and ratio<0.25:
                pflog.write("Unsuccessful step, ratio < 0.25, reducing trust radius\n")
                pflog.write("Accept step but update trust radius\n")
                trust_radius=0.25*trust_radius
            if ratio>0.75 and ratio<1.15:
                pflog.write("Successful step, ratio > 0.75, increasing trust radius\n")
                pflog.write("Accept step and update trust radius \n")
                trust_radius=2*trust_radius
            if ratio>1.15:
                pflog.write("(Too?) Very Successful step, ratio > 1.15\n")
                pflog.write("Accept step\n")
                if np.isclose(np.sqrt(np.sum(currentCoordinatesDifference**2)),trust_radius,rtol=0.1*trust_radius):
                    pflog.write("Displacement close to trust radius, increasing trust radius\n")
                    trust_radius=2*trust_radius
            pflog.write("Updated trust_radius = {}\n".format(trust_radius))
        previousGradientDifference=np.copy(currentGradientDifference)
        previousCoordinatesDifference=np.copy(currentCoordinatesDifference)
        previousGradient=np.copy(currentGradient)
        previousHessian=np.copy(currentHessian)
        pflog.close()
