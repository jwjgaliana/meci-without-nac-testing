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
        Home implementation of SLM method from LJD
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
    parser.add_argument("--NBS_type",metavar="NBS_type",type=str,required=False,
                        choices=['gonon','maeda'],default='gonon',
                        help="Method used to numerically estimate the Branching Space and the associated projector; ultimately changes the type of convergence test for the gradient.")
    parser.add_argument("--trustRadius",metavar="trustRadius",type=float,required=False,default=1.0,
                        help="Trust Radius for Quasi-Newton Search Step")
    parser.add_argument("--rootA",metavar="rootA",type=int,required=False,default=1,
                        help="Index of excited state A")
    parser.add_argument("--rootB",metavar="rootB",type=int,required=False,default=2,
                        help="Index of excited state B")
    add_bool_arg(parser,'fromGeometry',default=False,doc="Switch on/off initialization from geometry only")
    add_bool_arg(parser,'calcAllFreq',default=False,doc="Switch on/off freq calculation at each step")
    add_bool_arg(parser,'noFreq',default=True,doc="Switch on/off ALL freq calculation")
    add_bool_arg(parser,'bohr',default=True,doc="Switch on/off bohr units for Hessian update")
    add_bool_arg(parser,'massWeighting',default=False,doc="Switch on/off mass weighting for Hessian update")
    add_bool_arg(parser,'trustRegion',default=False,doc="Switch on/off trust radius update")

    args=parser.parse_args()

    molecule=args.molecule
    energyDifferenceThs=args.deltaEThreshold
    energyDifferenceSwitch=args.deltaESwitch
    optCriteria=args.optCriteria
    NBS_type=args.NBS_type
    trust_radius=args.trustRadius
    rootA=args.rootA
    rootB=args.rootB
    roots=[rootA,rootB]

    fromGeometry=args.fromGeometry
    calcAllFreq=args.calcAllFreq
    noFreq=args.noFreq
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
        freq=False
        energyThsForFreqUpdate=1e-3
        cfreq=5
    if noFreq:
        freq=False

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
    hessianlog1=open("HessianFileSED.log","w")
    hessianlog2=open("HessianFileAE.log","w")
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

        if freq:
            derivative_calculation="freq=(savenm,hpmodes)"
        elif not freq:
            derivative_calculation="Force"
        header_A=header_template
        header_A=header_A.replace("%STEP_NUMBER%","0")
        header_A=header_A.replace("%ROOT_TAG%","A")
        header_A=header_A.replace("%ROOT_NUMBER%",str(rootA))
        header_A=header_A.replace("%DERIVATIVE_CALCULATION%",derivative_calculation)
        header_A=header_A.replace("%TITLE%","step 0 frequency calculation for state A")
        MECISearch_MODULE.writeComFileFromXYZ(comfilenames[0],header_A,"step_0.xyz")

        header_B=header_template
        header_B=header_B.replace("%STEP_NUMBER%","0")
        header_B=header_B.replace("%ROOT_TAG%","B")
        header_B=header_B.replace("%ROOT_NUMBER%",str(rootB))
        header_B=header_B.replace("%DERIVATIVE_CALCULATION%",derivative_calculation)
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
    hessianlog1.close()
    hessianlog2.close()
    ####################################################
    # OPTIMIZATION LOOP                                #
    ####################################################
    while not np.all(convergenceTest) and c<cmax:
        freq=False
        pflog=open("ProgressionFile.log","a+")
        hessianlog1=open("HessianFileSED.log","a+")
        hessianlog2=open("HessianFileAE.log","a+")
        pflog.write("step {}".format(c)+"\n")

        #################################################
        # GET SYSTEM INFORMATION AND ENERGY DERIVATIVES #
        #################################################
        ## Modifiers are:
        ## - freq
        ## - mw
        ## - bohr
        NAtoms,NCoords,atomicNumbers,currentCoordinates=meci.fchk2coordinates("step_{}_A.fchk".format(c))
        atomicNumbers3=np.array([[_]*3 for _ in atomicNumbers]).flatten()
        atomicMasses3=meci.fchk2derivatives("step_{}_A.fchk".format(c),freq=False)[2]
        atomicMasses3*=CST.AMU_TO_ME
        pflog.write("step {} was a Force-only Calculation".format(c)+"\n")
        currentEnergyA,currentGradientA=MECISearch_MODULE.getStateDerivatives("step_{}_A.fchk".format(c),mw=mw,freq=freq)
        currentEnergyB,currentGradientB=MECISearch_MODULE.getStateDerivatives("step_{}_B.fchk".format(c),mw=mw,freq=freq)
        if bohr:
            currentCoordinates/=CST.BOHR_TO_ANGSTROM
            currentGradientA*=CST.BOHR_TO_ANGSTROM
            currentGradientB*=CST.BOHR_TO_ANGSTROM
        currentEnergyDifference=0.5*(currentEnergyB-currentEnergyA)
        currentEnergyAverage=0.5*(currentEnergyA+currentEnergyB)
        currentGradientDifference=0.5*(currentGradientB-currentGradientA)
        currentGradientSquaredDifference=2*currentEnergyDifference*currentGradientDifference # k_n = 2 Omega_n d_n
        currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))
        currentGradientAverage=0.5*(currentGradientA+currentGradientB) # s_n 

        pflog.close()
        pflog=open("ProgressionFile.log","a+")

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
            currentHessianAverage=0.5*np.eye(NCoords,NCoords) # S_n
            # currentHessianSquaredDifference=0*np.eye(NCoords,NCoords) # K_n
            # currentHessianSquaredDifference=1*np.eye(NCoords,NCoords) # K_n
            currentInverseHessianSquaredDifference=10000*np.eye(NCoords,NCoords)
            currentHessianSquaredDifference=scipy.linalg.inv(currentInverseHessianSquaredDifference)
            currentLM=0.1 # lambda_n
        else:
            currentHessianAverage=MECISearch_MODULE.BFGSUpdate(previousCoordinatesDifference,previousGradientAverage,previousHessianAverage,currentGradientAverage)
            currentHessianSquaredDifference=MECISearch_MODULE.BFGSUpdate(previousCoordinatesDifference,previousGradientSquaredDifference,previousHessianSquaredDifference,currentGradientSquaredDifference)
            intermediary_hessian=(previousHessianAverage+previousLM*previousHessianSquaredDifference)
            currentLM=(
                    (previousEnergyDifference**2-np.dot(previousGradientSquaredDifference,np.dot(scipy.linalg.inv(intermediary_hessian),previousGradientAverage))) # Ω_n**2-k_n·(S_n+λ_nK_n)^-1·s_n
                    /
                    (np.dot(previousGradientSquaredDifference,np.dot(scipy.linalg.inv(intermediary_hessian),previousGradientSquaredDifference))) # k_n·(S_n+λ_nK_n)^-1·k_n
                    )


        ####################################
        # COMPUTE COORDINATES OF NEXT STEP #
        # For electronic structure         #
        ####################################
        ## Modifiers:
        ## - mw

        intermediary_hessian=(currentHessianAverage+currentLM*currentHessianSquaredDifference)
        intermediary_gradient=(currentGradientAverage+currentLM*currentGradientSquaredDifference)
        print("currentGradientAverage",currentGradientAverage)
        print("currentHessianAverage",currentHessianAverage)
        print("currentGradientSquaredDifference",currentGradientSquaredDifference)
        print("currentHessianSquaredDifference",currentHessianSquaredDifference)
        print("currentLM",currentLM)
        print("intermediary_hessian",intermediary_hessian)
        print("intermediary_gradient",intermediary_gradient)
        currentCoordinatesDifference=-np.dot(scipy.linalg.inv(intermediary_hessian),intermediary_gradient)
        maxCoordinates=np.max(np.abs(currentCoordinatesDifference))
        if maxCoordinates>=0.02:
            currentCoordinatesDifference=(0.02/maxCoordinates)*currentCoordinatesDifference

        nextCoordinates=currentCoordinates+currentCoordinatesDifference
                # print("currentCoordinates",currentCoordinates)
        print("nextCoordinates",nextCoordinates)
        if bohr:
            currentCoordinates*=CST.BOHR_TO_ANGSTROM
            nextCoordinates*=CST.BOHR_TO_ANGSTROM
        nextCoordinates=nextCoordinates.reshape(NAtoms,3).astype(str)
        nextCoordinates=np.append(nextCoordinates,[[atomicNumber] for atomicNumber in atomicNumbers],axis=1)
        nextCoordinates=np.roll(nextCoordinates,1,axis=1)
        ###############################################
        # CONVERGENCE CHECK BEFORE ELEC. STRUCT. CALC #
        ###############################################

        if c!=0:
            if NBS_type=="gonon":
                eigenvalues,eigenvectors=scipy.linalg.eigh(currentHessianSquaredDifference)
                currentBranchingSpaceVectorLengths=eigenvalues[-2:] # Index Zero is the lowest, One the highest
                currentBranchingSpaceVectors=eigenvectors.T[-2:] # Index Zero is the lowest, One the highest
                currentBranchingSpaceProjector=(
                        np.tensordot(currentBranchingSpaceVectors[0],currentBranchingSpaceVectors[0],axes=0)
                        +
                        np.tensordot(currentBranchingSpaceVectors[1],currentBranchingSpaceVectors[1],axes=0)
                        )
            elif NBS_type=="maeda":
                overlap=np.empty(4).reshape(2,2)
                overlap[0,0]=np.dot(previousGradientDifference,previousGradientDifference)
                overlap[0,1]=np.dot(currentGradientDifference,previousGradientDifference)
                overlap[1,0]=np.dot(previousGradientDifference,currentGradientDifference)
                overlap[1,1]=np.dot(currentGradientDifference,currentGradientDifference)
                inversed_overlap=scipy.linalg.inv(overlap)
                currentBranchingSpaceProjector=(
                        inversed_overlap[0,0]*np.tensordot(previousGradientDifference,previousGradientDifference,axes=0)
                        +
                        inversed_overlap[0,1]*np.tensordot(currentGradientDifference,previousGradientDifference,axes=0)
                        +
                        inversed_overlap[1,0]*np.tensordot(previousGradientDifference,currentGradientDifference,axes=0)
                        +
                        inversed_overlap[1,1]*np.tensordot(currentGradientDifference,currentGradientDifference,axes=0)
                        )

            currentIntersectionSpaceProjector=(
                    np.eye(NCoords)
                    -currentBranchingSpaceProjector
                    )
            currentProjectedGradientAverage=np.dot(currentBranchingSpaceProjector,currentGradientAverage)
            currentProjectedOutGradientAverage=np.dot(currentIntersectionSpaceProjector,currentGradientAverage)

        energyDifferenceFlag=currentEnergyDifference<energyDifferenceThs
        if c==0:
            currentGradient=np.copy(currentGradientAverage)
        else:
            currentGradient=np.copy(currentProjectedOutGradientAverage)
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
        pflog.write("Calculation method for BS projector (hence seam gradient) {}\n".format(NBS_type))
        pflog.write("Current Lagrange Multiplier {}\n".format(currentLM))
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
            hessianlog1.write("Hessian Squared Energy Difference (updated) (Eh^2/Bohr^2) at step {}\n".format(c))
            hessianlog2.write("Hessian Energy Average (updated) (Eh^2/Bohr^2) at step {}\n".format(c))
        else:
            hessianlog1.write("Hessian Squared Energy Difference (updated) (Eh^2/Angstrom^2) at step {}\n".format(c))
            hessianlog2.write("Hessian Energy Average (updated) (Eh^2/Angstrom^2) at step {}\n".format(c))
        MECISearch_MODULE.writeTriangularMatrix(currentHessianSquaredDifference,hessianlog1)
        MECISearch_MODULE.writeTriangularMatrix(currentHessianAverage,hessianlog2)
        hessianlog1.write("\n")
        hessianlog2.write("\n")
        hessianlog1.close()
        hessianlog2.close()
        if not np.all(convergenceTest):
            c+=1
            if not calcAllFreq:
                if (c%cfreq==0 or currentEnergyDifference<energyThsForFreqUpdate):
                    freq=True
                elif c%cfreq!=0:
                    freq=False
            if noFreq:
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
            os.system("subg16a03bg {}".format(comfilenames[0]))
            os.system("subg16a03bg {}".format(comfilenames[1]))
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
        previousEnergyDifference=currentEnergyDifference
        previousGradientDifference=np.copy(currentGradientDifference)
        previousGradientSquaredDifference=np.copy(currentGradientSquaredDifference)
        previousHessianSquaredDifference=np.copy(currentHessianSquaredDifference)
        previousGradientAverage=np.copy(currentGradientAverage)
        previousHessianAverage=np.copy(currentHessianAverage)
        previousLM=currentLM
        previousCoordinatesDifference=np.copy(currentCoordinatesDifference)
        previousGradient=np.copy(currentGradient)
        pflog.close()
