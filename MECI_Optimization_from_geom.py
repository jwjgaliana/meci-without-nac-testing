#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import sys,os
import shutil
import time
import MECI as meci

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        """
        Optimization of Conical Intersections.

        This program requires:
        - a initial geometry step_0.xyz
        - two initial fchk files from frequency calculations* (at this initial geometry)
            - step_0_{rootA}.fchk or step_0_A.fchk (depending on roottag)
            - step_0_{rootB}.fchk or step_0_B.fchk (depending on roottag)
        and computes steps n+1 starting from n=0 through a BFGS update of the Hessian of a Seam.

        At each step, the program updates the file "ProgressionFile" with the new geometry, gradient and inverse Hessian along with the convergence update.

        * The programs still requires frequency calculations to compute accurately the Numerical Branching Space.
        For improvements and real use of inverse Hessian updates, look into Ruiz-Barragan /et al./ Journal of Chemical Theory and Computation, 2013.

        Inspired from 
        - Gonon /et al./, The Journal of Chemical Physics, 2017 (Numerical Branching Space)
        - Harvey /et al./, Theoretical Chemistry Accounts, 1998 (Composed Gradient step)
        - Sicilia /et al./, Journal of Chemical Theory and Computation, 2008 (Composed Displacement step)

        TODO:
        [X] Copy the version of the program in the directory before launching
        [-] Implement the use external templates
        [-] Try DNR 
        [-] Try Hessian update for NBS (full update)
        [-] Implement more atoms and atomic masses (C, N, H for now)
        [X] Black-box initialization (before, required initial xyz and fchk files)
        [-] Use pandas table printing and Gaussian-like printing for gradients and hessians
        [-] Implement restart step
        """
    )

    parser.add_argument("--molecule",metavar="molecule",required=True,help="Name/tag of the molecule")
    parser.add_argument("--NAtoms",metavar="NAtoms",required=True,help="Number of atoms")
    parser.add_argument("--algo",metavar="algo",required=True,help="Type of calculations ({harvey} or {sicilia})")
    parser.add_argument("--rootA",metavar="rootA",required=True,help="Index of first excited state")
    parser.add_argument("--rootB",metavar="rootB",required=True,help="Index of second excited state")
    parser.add_argument("--nstate",metavar="nstate",required=True,help="Number of states in the calculation")
    parser.add_argument("--roottag",metavar="roottag",required=False,help="Suffix for the states ({filetag}_{1,2} or {filetag}_{A,B})",default="AB")
    parser.add_argument("--check_NBS",metavar="check_NBS",required=False,default="yes",help="If yes or y, saves the NBS vectors")
    parser.add_argument("--template",metavar="template",required=False,default="no",help="[WIP] Use of an external template for Gaussian calculations [Not implemented yet]")
    parser.add_argument("--step_limit",metavar="step_limit",required=False,default="500",help="Limit of steps")
    parser.add_argument("--background",metavar="background",required=False,default="no",help="If yes or y, sbatch calculations in background")
    parser.add_argument("--maxDeltaX",metavar="maxDeltaX",required=False,default=0.01,help="Limit for max change coordinates (default=0.01); 'inf' for no limit")
    args=vars(parser.parse_args())

    molecule=args["molecule"]
    NAtoms=int(args["NAtoms"])
    algo=args["algo"]
    rootA=args["rootA"]
    rootB=args["rootB"]
    nstate=int(args["nstate"])
    roottag=args["roottag"]
    check_NBS=args["check_NBS"]
    template=args["template"]
    step_limit=int(args["step_limit"])
    background=args["background"]
    maxDeltaX=args["maxDeltaX"]


    ## Copy version of the script used to the current directory
    if "cold" in os.path.dirname(__file__):
        copied_program_name=os.path.basename(__file__).split(".")[0]+'_'+time.strftime("%Y%m%d_%H%M")+".py"
        copied_library_name="MECI_"+time.strftime("%Y%m%d_%H%M")+".py"
        shutil.copy(__file__,copied_program_name)
        shutil.copy(os.path.dirname(__file__)+"/MECI.py",copied_library_name)

    ## Manual limitation (or not) for max change coordinates
    if maxDeltaX=="inf":
        maxDeltaX=np.inf
    else:
        maxDeltaX=float(maxDeltaX)

    ## Physical constants, conversion factors
    amu2me=1822.888486209 # amu to mass of the electron, me per amu (Dalton)
    bohr2angstrom=0.529177210903 # bohr to angstrom, angstrom per bohr

    ## System
    NCoords=3*NAtoms
    NModes=3*NAtoms-6

    ## Optimization criteria: default definitions, ~verytight
    maxGradientThreshold=7e-4 # Eh/Ang
    RMSGradientThreshold=5e-4 # Eh/Ang
    maxCoordinatesDifferenceThreshold=4e-4# Ang
    RMSCoordinatesDifferenceThreshold=2.5e-4 # Ang
    energyDifferenceThreshold=5e-5 # Eh
    energyDifferenceTargetValue=2e-6 # Eh 
    # Optimization criteria: redefinition by the user (uncomment and modify values)
    # maxGradientThreshold=7e-4 # Eh/Ang
    # RMSGradientThreshold=5e-4 # Eh/Ang
    # maxCoordinatesDifferenceThreshold=4e-4# Ang
    # RMSCoordinatesDifferenceThreshold=2.5e-4 # Ang
    # energyDifferenceThreshold=5e-5 # Eh
    # energyDifferenceTargetValue=2e-6 # Eh 
    if algo=="sicilia":
        ## Put to Eths to follow Sicilia Algorithm (hybrid CG-CS)
        siciliaEnergyDifferenceThreshold=energyDifferenceThreshold # Eh
    if algo=="harvey":
        ## Put to 0 to follow Harvey Algorithm (only CG)
        siciliaEnergyDifferenceThreshold=0 # Eh
    ## BFGS Loop
    # maxMaxChange=0.0100 ## Control of maximum displacement
    maxMaxChange=maxDeltaX
    maxRMSChange=NCoords*maxMaxChange
    ## Initialization
    maxGradientConverged=False
    RMSGradientConverged=False
    maxCoordinatesDifferenceConverged=False
    RMSCoordinatesDifferenceConverged=False
    energyDifferenceConverged=False
    first_step=True
    current_step=0
    check_current_NBS=True
    facPP=1 ## using xxxGradientDifferenceRMS as a factor...

    with open("ProgressionFile","w") as PF:
        PF.write("MECI Optimization, inspired from Sicilia, 2008, JCTC\n")
        PF.write("NAtoms "+str(NAtoms)+"\n")
        PF.write("NCoords "+str(NCoords)+"\n")
        PF.write("Convergence Criteria"+"\n")
        PF.write("Gradient Max "+str(maxGradientThreshold)+"\n")
        PF.write("Gradient RMS "+str(RMSGradientThreshold)+"\n")
        PF.write("Coordinates Change Max "+str(maxCoordinatesDifferenceThreshold)+"\n")
        PF.write("Coordinates Change RMS "+str(RMSCoordinatesDifferenceThreshold)+"\n")
        PF.write("Energy Difference "+str(energyDifferenceThreshold)+"\n")
        PF.write("Sicila Energy Difference "+str(siciliaEnergyDifferenceThreshold)+"\n")
    while not energyDifferenceConverged or not maxGradientConverged or not RMSGradientConverged or not maxCoordinatesDifferenceConverged or not RMSCoordinatesDifferenceConverged and current_step<step_limit:
        filetag="step_"+str(current_step)
        with open("step_"+str(current_step)+".xyz","r") as f:
            coordinates=f.readlines()[2:]
        currentCoordinatesText=np.array([line.split() for line in coordinates])
        currentCoordinates=currentCoordinatesText[:,1:].astype(float)
        if template=="no" or template=="n":
            ## First state
            with open("step_"+str(current_step)+"_A.com","w") as f:
                f.write("%chk=step_"+str(current_step)+"_A.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                f.write("# freq=(savenm,hpmodes) pop=full geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root="+str(rootA)+",nstate="+str(nstate)+") scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write(str(molecule)+" CoIn opt step"+str(current_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=currentCoordinates
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if currentCoordinatesText[p,0]=="C" or str(currentCoordinatesText[p,0])=="6":
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    elif currentCoordinatesText[p,0]=="N" or str(currentCoordinatesText[p,0])=="7":
                        f.write("N "+" ".join(to_print))
                        f.write('\n')
                    elif currentCoordinatesText[p,0]=="O" or str(currentCoordinatesText[p,0])=="8":
                        f.write("O "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
            ## Second state
            with open("step_"+str(current_step)+"_B.com","w") as f:
                f.write("%chk=step_"+str(current_step)+"_B.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                f.write("# freq=(savenm,hpmodes) pop=full geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root="+str(rootB)+",nstate="+str(nstate)+") scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write(str(molecule)+" CoIn opt step"+str(current_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=currentCoordinates
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if currentCoordinatesText[p,0]=="C" or str(currentCoordinatesText[p,0])=="6":
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    elif currentCoordinatesText[p,0]=="N" or str(currentCoordinatesText[p,0])=="7":
                        f.write("N "+" ".join(to_print))
                        f.write('\n')
                    elif currentCoordinatesText[p,0]=="O" or str(currentCoordinatesText[p,0])=="8":
                        f.write("O "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
        if background=="yes" or background=="y":
            os.system("subg16a03bg step_"+str(current_step)+"_A.com")
            os.system("subg16a03bg step_"+str(current_step)+"_B.com")
        else: 
            os.system("subg16a03 step_"+str(current_step)+"_A.com")
            os.system("subg16a03 step_"+str(current_step)+"_B.com")
        finished1=False
        finished2=False
        c=0
        while not finished1 or not finished2:
            c=c+1
            time.sleep(30)
            if os.path.isfile("step_"+str(current_step)+"_A.log"):
                with open("step_"+str(current_step)+"_A.log","r") as f:
                    last_line=f.readlines()[-1].split()
                    if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                        finished1=True
            if os.path.isfile("step_"+str(current_step)+"_B.log"):
                with open("step_"+str(current_step)+"_B.log","r") as f:
                    last_line=f.readlines()[-1].split()
                    if len(last_line)>2 and last_line[0]=="Normal" and last_line[1]=="termination":
                        finished2=True
        filetagSA="step_"+str(current_step)+"_A"
        os.system("formchk16 "+filetagSA+".chk")
        os.system("rm "+filetagSA+".chk")
        filetagSB="step_"+str(current_step)+"_B"
        os.system("formchk16 "+filetagSB+".chk")
        os.system("rm "+filetagSB+".chk")

        current_fosc=[]
        with open("step_"+str(current_step)+"_A.log","r") as f:
            lines=f.readlines()
        for l in range(len(lines)):
            line=lines[l].split()
            for n in range(1,nstate+1):
                if len(line)>=2 and line[0]=="Excited" and line[1]=="State" and line[2]==str(n)+":":
                    current_fosc.append(line[-2][2:])
        current_fosc=np.array(current_fosc,dtype=str)


        with open("step_"+str(current_step)+"_A.fchk","r") as f:
            lines=f.readlines()
            currentEnergyS1,currentGradientS1,currentHessianS1,atomicMasses=meci.fchk2derivatives(lines,mw=True,freq=True)[:4]
            currentGradientS1=currentGradientS1.flatten() ## Hartree / Bohr
            currentGradientS1=currentGradientS1/bohr2angstrom ## Hartree / (Angstrom · me^1/2)
            currentHessianS1=currentHessianS1/bohr2angstrom/bohr2angstrom ## Hartree / (Angstrom² · me)
        with open("step_"+str(current_step)+"_B.fchk","r") as f:
            lines=f.readlines()
            currentEnergyS2,currentGradientS2,currentHessianS2=meci.fchk2derivatives(lines,mw=True,freq=True)[:3]
            currentGradientS2=currentGradientS2.flatten() ## Hartree / Bohr
            currentGradientS2=currentGradientS2/bohr2angstrom ## Hartree / (Angstrom · me^1/2)
            currentHessianS2=currentHessianS2/bohr2angstrom/bohr2angstrom ## Hartree / (Angstrom² · me)
        with open("step_"+str(current_step)+".xyz","r") as f:
            coordinates=f.readlines()[2:]
        currentCoordinatesText=np.array([line.split() for line in coordinates])
        currentCoordinates=currentCoordinatesText[:,1:].astype(float)
        currentCoordinates=currentCoordinates.flatten() ## Angstrom

        currentEnergyDifference=currentEnergyS2-currentEnergyS1
        currentGradientDifference=currentGradientS2-currentGradientS1
        currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))

        currentHessianDifference=currentHessianS2-currentHessianS1
        currentGradientDifferenceProjector=np.tensordot(0.5*currentGradientDifference,0.5*currentGradientDifference,axes=0)
        currentSquaredHessianDifference=2*(0.5*currentEnergyDifference)*(0.5*currentHessianDifference)+2*currentGradientDifferenceProjector
        eigval,diagonalizer=linalg.eigh(currentSquaredHessianDifference)
        eigvec=diagonalizer.T
        currentBranchingSpaceVector1=eigvec[-1].flatten()
        currentBranchingSpaceVector2=eigvec[-2].flatten()

        if check_NBS=="y" or check_NBS=="yes":
            if not os.path.isdir("check_NBS"):
                os.system("mkdir check_NBS")
            current_coordinates=currentCoordinates.reshape(NAtoms,3)
            current_BranchingSpaceVector1=currentBranchingSpaceVector1.reshape(NAtoms,3) 
            current_BranchingSpaceVector2=currentBranchingSpaceVector2.reshape(NAtoms,3) 
    
            fig=plt.figure()
            ax=fig.add_subplot(111,projection="3d")
            ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
            ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector1[:,0],current_coordinates[:,1]+current_BranchingSpaceVector1[:,1],current_coordinates[:,2]+current_BranchingSpaceVector1[:,2])
            max_coordinates=np.max(np.abs(current_coordinates))
            ax.set_xlim(-max_coordinates,max_coordinates)
            ax.set_ylim(-max_coordinates,max_coordinates)
            ax.set_zlim(-max_coordinates,max_coordinates)
            plt.savefig("check_NBS/"+filetag+"_NBS1.png")
            fig=plt.figure()
            ax=fig.add_subplot(111,projection="3d")
            ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
            ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector2[:,0],current_coordinates[:,1]+current_BranchingSpaceVector2[:,1],current_coordinates[:,2]+current_BranchingSpaceVector2[:,2])
            ax.set_xlim(-max_coordinates,max_coordinates)
            ax.set_ylim(-max_coordinates,max_coordinates)
            ax.set_zlim(-max_coordinates,max_coordinates)
            plt.savefig("check_NBS/"+filetag+"_NBS2.png")

        with open("ProgressionFile","a+") as PF:
            PF.write("Current step "+str(current_step)+"\n")
            PF.write("Energy S1 "+str(currentEnergyS1)+"\n")
            PF.write("Energy S2 "+str(currentEnergyS2)+"\n")
            PF.write("Energy Difference "+str(currentEnergyDifference)+"\n")
            PF.write("Atomic Masses AMU, sqrt(M·me) \n")
            for atomicMass in atomicMasses:
                PF.write(str(atomicMass)+"\t"+str(np.sqrt(atomicMass*amu2me))+"\n")
            PF.write("Current Geometry \n")
            PF.write("Atom \t x \t y \t z \n")
            for atom in currentCoordinatesText.reshape(NAtoms,4):
                PF.write("\t".join(atom.astype(str))+"\n")
 
        currentGradientAverage=0.5*(currentGradientS1+currentGradientS2)
        currentProjectedGradient=(currentGradientAverage
                                  -np.dot(currentGradientAverage,currentBranchingSpaceVector1)*currentBranchingSpaceVector1
                                  -np.dot(currentGradientAverage,currentBranchingSpaceVector2)*currentBranchingSpaceVector2
                                  )

        if currentEnergyDifference>=siciliaEnergyDifferenceThreshold:
            ## Current Step is a Composed Gradient Step (CG)
            currentGradient=(currentProjectedGradient ## Projection of MG out of the GD
                            +2*facPP*currentEnergyDifference*currentGradientDifference/currentGradientDifferenceNorm ## f gradient normalized 
                            )
        if currentEnergyDifference<siciliaEnergyDifferenceThreshold:
            currentGradient=(currentProjectedGradient ## Projection of MG out of the GD
                            )
        if first_step:
            ## Initialization of the Hessian Update
            currentInverseHessian=0.7*np.eye(NCoords,NCoords) ## Was an initialization ok for not m.-w. Hessians... here it is in Angstrom²/Eh
            ## So it must be m.w.!
            for i in range(NCoords):
                for j in range(NCoords):
                    currentInverseHessian[i,j]=currentInverseHessian[i,j]*(np.sqrt(atomicMasses[i]*amu2me*atomicMasses[j]*amu2me))
                    ## Now in (Angstrom² · me)/Eh with correct ponderation for C and H...
        elif not first_step:
            ## Update of the Inverse Hessian, BFGS step
            totalGradientDifference=currentGradient-previousGradient ## Not the GD Gradient Difference, but the Difference in the Gradients of previous and current steps
            currentCoordinates=currentCoordinates.flatten()
            coordinatesDifference=currentCoordinates-previousCoordinates
            for i in range(NCoords):
                coordinatesDifference[i]=coordinatesDifference[i]*np.sqrt(atomicMasses[i]*amu2me)
            projectionGradientCoordinates=np.dot(totalGradientDifference,coordinatesDifference)
            currentInverseHessian=(previousInverseHessian ## Matrix
                                +(1+np.dot(totalGradientDifference,np.dot(previousInverseHessian,totalGradientDifference))/projectionGradientCoordinates) ## Number
                                *(np.tensordot(coordinatesDifference,coordinatesDifference,axes=0))/projectionGradientCoordinates ## Matrix, times the above number, still matrix
                                -np.dot(previousInverseHessian,np.tensordot(totalGradientDifference,coordinatesDifference,axes=0))/projectionGradientCoordinates ## Matrix
                                -np.dot(np.tensordot(coordinatesDifference,totalGradientDifference,axes=0),previousInverseHessian)/projectionGradientCoordinates ## Matrix 
                                )

        ## Computation of the displacement associated to the Composed Gradient (CG) step
        nextChangeCoordinates=-np.dot(currentInverseHessian,currentGradient)
        ## currentGradient is in (Eh)/(Angstrom · me^1/2) and currentInverseHessian is in (Angstrom² · me)/(Eh)
        ## so nextChangeCoordinates is here in (Angstrom · me^1/2)
        ## it must be set anew in the Angstrom Cartesian system
        for i in range(NCoords):
            nextChangeCoordinates[i]=nextChangeCoordinates[i]/(np.sqrt(atomicMasses[i]*amu2me)) ## from Angstrom · me^1/2 to Angstrom
        ## Control of the change in coordinates
        maxChangeCoordinates=np.max(np.abs(nextChangeCoordinates))
        RMSChangeCoordinates=np.sqrt(np.sum(nextChangeCoordinates**2)/NCoords)
        if RMSChangeCoordinates > maxRMSChange:
            nextChangeCoordinates=nextChangeCoordinates*(maxRMSChange/RMSChangeCoordinates)
        if maxChangeCoordinates > maxMaxChange:
            nextChangeCoordinates=nextChangeCoordinates*(maxMaxChange/maxChangeCoordinates)
        nextCoordinates=currentCoordinates+nextChangeCoordinates
        if currentEnergyDifference<siciliaEnergyDifferenceThreshold:
            nextChangeCoordinatesGradientDifference=((energyDifferenceTargetValue-currentEnergyDifference)/currentGradientDifferenceNorm)*(currentGradientDifference/currentGradientDifferenceNorm)
            ## (Eh / (Eh / (Angstrom · me^1/2)) = Angstrom · me^1/2
            ## it must be set anew in the Angstrom Cartesian system
            for i in range(NCoords):
                nextChangeCoordinatesGradientDifference[i]=nextChangeCoordinatesGradientDifference[i]/(np.sqrt(atomicMasses[i]*amu2me)) ## from Angstrom · me^1/2 to Angstrom
            nextCoordinates=nextCoordinates+nextChangeCoordinatesGradientDifference
        nextCoordinates=nextCoordinates.reshape(NAtoms,3)
        with open("ProgressionFile","a+") as PF:
            if currentEnergyDifference>=siciliaEnergyDifferenceThreshold:
                PF.write("Composed Gradient Step (CG) \n")
            if currentEnergyDifference<siciliaEnergyDifferenceThreshold:
                PF.write("Composed Step (CS) \n")
            PF.write("Gradient at step "+str(current_step)+"\n")
            PF.write("x \t y \t z \n")
            for atom in currentGradient.reshape(NAtoms,3):
                PF.write("\t".join(atom.astype(str))+"\n")
            PF.write("Inverse Hessian at step "+str(current_step)+"\n")
            for line in currentInverseHessian:
                PF.write(" ".join(line.astype(str))+"\n")

        ## Produce new next_step geometry and associated inputs for Gaussian
        next_step=current_step+1
        filetag="step_"+str(next_step)
        with open("step_"+str(next_step)+".xyz","w") as f:
            displaced_coordinates=nextCoordinates
            f.write(str(len(displaced_coordinates)))
            f.write('\n')
            f.write("step_"+str(next_step)+'\n')
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if currentCoordinatesText[p,0]=="C" or str(currentCoordinatesText[p,0])=="6":
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                elif currentCoordinatesText[p,0]=="N" or str(currentCoordinatesText[p,0])=="7":
                    f.write("N "+" ".join(to_print))
                    f.write('\n')
                elif currentCoordinatesText[p,0]=="O" or str(currentCoordinatesText[p,0])=="8":
                    f.write("O "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')

        ## Convergence test
        currentProjectedGradientNotMW=np.copy(currentProjectedGradient)
        currentGradientNotMW=np.copy(currentGradient)
        for i in range(NCoords):
            currentProjectedGradientNotMW[i]=currentProjectedGradientNotMW[i]*(np.sqrt(atomicMasses[i]*amu2me)) ## from Eh / (Angstrom · me^1/2) to Eh / Angstrom
            currentGradientNotMW[i]=currentGradientNotMW[i]*(np.sqrt(atomicMasses[i]*amu2me)) ## from Eh / (Angstrom · me^1/2) to Eh / Angstrom
        currentProjectedGradientRMS=np.sqrt(np.sum(currentProjectedGradientNotMW**2)/NCoords)
        currentProjectedGradientMax=np.max(np.abs(currentProjectedGradientNotMW))
        currentGradientRMS=np.sqrt(np.sum(currentGradientNotMW**2)/NCoords)
        currentGradientMax=np.max(np.abs(currentGradientNotMW))
        if not first_step:
            currentChangeCoordinatesRMS=np.sqrt(np.sum(currentChangeCoordinates**2)/NCoords)
            currentChangeCoordinatesMax=np.max(np.abs(currentChangeCoordinates))
        if currentGradientMax <= maxGradientThreshold:
            maxGradientConverged=True
        else:
            maxGradientConverged=False
        if currentGradientRMS <= RMSGradientThreshold:
            RMSGradientConverged=True
        else:
            RMSGradientConverged=False
        if not first_step:
            if currentChangeCoordinatesMax <= maxCoordinatesDifferenceThreshold:
                maxCoordinatesDifferenceConverged=True
            else:
                maxCoordinatesDifferenceConverged=False
            if currentChangeCoordinatesRMS <= RMSCoordinatesDifferenceThreshold:
                RMSCoordinatesDifferenceConverged=True
            else:
                RMSCoordinatesDifferenceConverged=False
        if currentEnergyDifference <= energyDifferenceThreshold:
            energyDifferenceConverged=True
        else:
            energyDifferenceConverged=False
        with open("ProgressionFile","a+") as PF:
            PF.write("Progression at step "+str(current_step)+"\n")
            PF.write("Geometry \n")
            PF.write("Atom \t x \t y \t z \n")
            for atom in currentCoordinatesText.reshape(NAtoms,4):
                PF.write("\t".join(atom.astype(str))+"\n")
            PF.write("Total Energy S1 "+str(currentEnergyS1)+"\n")
            PF.write("Total Energy S2 "+str(currentEnergyS2)+"\n")
            PF.write("Energy Difference "+str(currentEnergyDifference)+"\n")
            PF.write("Osc. Str.\n")
            PF.write(" ".join(current_fosc)+"\n")
     
        with open("ProgressionFile","a+") as PF:
            PF.write("Gradient Seam Max (not MW) "+str(currentProjectedGradientMax)+"\n")
            PF.write("Gradient Seam RMS (not MW) "+str(currentProjectedGradientRMS)+"\n")
            if currentEnergyDifference>=siciliaEnergyDifferenceThreshold:
                PF.write("Convergence Test at step "+str(current_step)+", Composed Gradient\n")
            if currentEnergyDifference<siciliaEnergyDifferenceThreshold:
                PF.write("Convergence Test at step "+str(current_step)+", Composed Step\n")
            PF.write("Total Gradient Max (not MW) "+str(currentGradientMax)+"\t"+str(maxGradientConverged)+"\n")
            PF.write("Total Gradient RMS (not MW) "+str(currentGradientRMS)+"\t"+str(RMSGradientConverged)+"\n")
            if first_step:
                PF.write("Coordinates Change Max "+" Not available (first step) "+"\t"+str(maxCoordinatesDifferenceConverged)+"\n")
                PF.write("Coordinates Change RMS "+" Not available (first step) "+"\t"+str(RMSCoordinatesDifferenceConverged)+"\n")
            elif not first_step:
                PF.write("Coordinates Change Max "+str(currentChangeCoordinatesMax)+"\t"+str(maxCoordinatesDifferenceConverged)+"\n")
                PF.write("Coordinates Change RMS "+str(currentChangeCoordinatesRMS)+"\t"+str(RMSCoordinatesDifferenceConverged)+"\n")
            PF.write("Energy Difference "+str(currentEnergyDifference)+"\t"+str(energyDifferenceConverged)+"\n")
        ## Passing data to the next loop
        current_step=next_step
        previousInverseHessian=currentInverseHessian
        previousGradient=currentGradient
        previousCoordinates=currentCoordinates
        previousCoordinatesText=currentCoordinatesText
        currentChangeCoordinates=nextChangeCoordinates
        if first_step:
            first_step=False
