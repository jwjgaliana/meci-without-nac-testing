#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg
import scipy
import sys,os
import time
import pandas as pd
import procrustes

import CONSTANTS as CST

def log2xyz(filetag,NAtoms,standard=False,write=False,write_log=True):
    """
    Takes as input the filetag associated to {filetag}.log, the output of a Gaussian calculation (optimization or no)
    Gets the last printed coordinates in angström.
    Default: takes the input orientation coordinates.
    Not default, write=True, prints the coordinates in a {filetag}_log2xyz.xyz files
    """
    with open(filetag+".log","r") as f:
        lines=f.readlines()
    coordinates=[]
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line) > 1 and line[0]=="NAtoms=":
            NAtoms=int(line[1])
            # NModes=int(3*NAtoms-6)
            # NCoords=int(3*NAtoms)
        if standard and len(line) > 1 and line[0]=="Standard" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
        if not standard and len(line) > 1 and line[0]=="Input" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
    if write:
        with open(filetag+"_log2xyz.xyz","w") as f:
            f.write(str(len(coordinates)))
            f.write('\n')
            f.write(filetag+" output geometry"+'\n')
            for p in range(len(coordinates)):
                to_print=coordinates[p,3:].astype(float)
                to_print=["%.16f" % _ for _ in to_print]
                if int(coordinates[p,1])==6:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                elif int(coordinates[p,1])==1:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    return(NAtoms,coordinates[:,3:].astype(float))

def getNormalModes(filetag):
    """
    Takes as input the filetag associated to {filetag}.log, the output of a Gaussian Freq. calculation.
    Outputs
    - normal modes, array of (3NAtoms-6)*(3NAtoms), not mass-weighted "cartesian" directions
    - frequencies, array of (3NAtoms-6) elements, in rcm
      rmk: "precise" freq. can return "*********" as a value for some normal modes at CoIns; this is here flagged as "-100000"
    - reduced masses, array of (3NAtoms-6) elements, in AMU
    """
    filename=filetag+".log"
    with open(filename,'r') as f:
        lines=f.readlines()
    precise=0
    normal_modes=[]
    reduced_masses=[]
    frequencies=[]
    symmetry_labels=[]
    CoIn_inf=False
    CoIn_sup=False
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line) > 1 and line[0]=="NAtoms=":
            NAtoms=int(line[1])
            NModes=int(3*NAtoms-6)
            NCoords=int(3*NAtoms)
        if len(line) > 1 and line[0]=="Frequencies" and precise==0:
            precise=1
            for p in range(NModes):
                ref_line=i-2+7*(1+p//5)
                current_reduced_mass=lines[ref_line-4+p//5*NCoords].split()[3+p%5]
                current_symmetry_label=lines[ref_line-6+p//5*NCoords].split()[0+p%5]
                # Collect frequency if there has been no information about unreadable frequency for superior state of a Conical Intersection
                if not CoIn_sup:
                    current_frequency=lines[ref_line-5+p//5*NCoords].split()[2+p%5]
                # Assign frequency=1000000 if there has been information about unreadable frequency for superior state of a Conical Intersection
                elif CoIn_sup:
                    current_frequency=str(100000)
                # Check if there is unreadable data in the current_frequency line
                if "*" in current_frequency:
                    # Check if the unreadable frequency is in the first subcolumn of the table or the last
                    # If in the first, then the unreadable frequency is an indication of an inferior state of a Conical Intersection
                    if current_frequency[0]=="*":
                        current_frequency=str(-100000)
                        CoIn_inf=True
                    # If not in the firt, than in the last and the unreadable frequency is an indication of superior state of a Conical Intersection
                    else:
                        # The readable data is recovered by removing unreadable data
                        current_frequency=current_frequency.translate({ord('*'): None})
                        CoIn_sup=True
                reduced_masses.append(current_reduced_mass)
                symmetry_labels.append(current_symmetry_label)
                frequencies.append(current_frequency)
                current_mode=[]
                for n in range(NCoords):
                    current_mode.append(lines[ref_line+p//5*NCoords+n].split()[3+p%5])
                normal_modes.append(current_mode)
    reduced_masses=np.array(reduced_masses).astype(float)
    frequencies=np.array(frequencies).astype(float)
    symmetry_labels=np.array(symmetry_labels).astype(str)
    normal_modes=np.array(normal_modes).astype(float) # cartesian coordinates not mass-rescaled
    return(normal_modes,frequencies,reduced_masses,symmetry_labels)

def rotate_vectors(vectors,angle):
    c=np.cos(angle)
    s=np.sin(angle)
    R_matrix=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return np.array([np.dot(R_matrix,vector) for vector in vectors])

#lines2file def fchk2derivatives(lines,mw=True,freq=True):
def fchk2derivatives(filename,mw=True,freq=True):
    # print(freq)
    """
    Takes as input the filename of a formatted chk Gaussian file, output.fchk.
    Returns
    - ETot energy of the root state, in Hartree
    - gradient of the root state, in Hartree/Bohr when mw=False
    - hessian in Hartree/Bohr**2 when mw=False
    - atomic_masses in AMU
    - NAtoms
    """
    with open(filename,'r') as f:
        lines=f.readlines()
    gradient=[]
    if freq:
        hessianElements=[]
    atomic_numbers=[]
    atomic_masses=[]
    # lines=lines_S1
    # Go through the lines of the fchk
    for i in range(len(lines)):
        line=lines[i].split()
        # Collect energy of the root state
        if len(line)>2 and line[0]=="Total" and line[1]=="Energy":
            ETot=float(line[-1])
        # Collect atomic numbers
        # Columns of 6 values
        if len(line)>2 and line[0]=="Atomic":
            NAtoms=int(line[-1])
            for p in range(NAtoms):
                atomic_numbers.append(lines[i+1+p//6].split()[p%6])
        # Collect atomic masses (in AMU)
        # Columns of 5 values
        if len(line)>2 and line[0]=="Real" and line[1]=="atomic":
            # NAtoms=int(line[-1])
            for p in range(NAtoms):
                atomic_masses.append(lines[i+1+p//5].split()[p%5])
        # Collect Cartesian Gradients (in atomic units, Hartree/bohr)
        # Columns of 5 values
        if len(line)>2 and line[0]=="Cartesian" and line[1]=="Gradient":
            NCoords=int(line[-1])
            for p in range(NCoords):
                gradient.append(lines[i+1+p//5].split()[p%5])
        # Collect Cartesian, non-mass-weighted, Hessian (in atomic units, Hartree/bohr²)
        # Hessian is given in an triangular matrix form, with NElements=NCoords*(NCoords+1)/2 elements
        # Columns of 5 values
        if freq and len(line)>2 and line[0]=="Cartesian" and line[1]=="Force":
            NElements=int(line[-1])
            for p in range(NElements):
                hessianElements.append(lines[i+1+p//5].split()[p%5])

    atomic_numbers=np.array(atomic_numbers).astype(str)
    # Either use the atomic masses from fchk or atomic numbers
    atomic_masses=np.array(atomic_masses).astype(float)
    atomic_masses=np.array([[atomic_mass]*3 for atomic_mass in atomic_masses]).flatten()
    gradient=np.array(gradient).astype(float)
    if freq:
        hessianElements=np.array(hessianElements).astype(float)

    if freq:
        # print("freq")
        # Build the upper triangular matrix
        # By filling line by line
        # First line and first column, first hessianElements...
        # Second line and two first columuns, second and third hessianElements...
        hessian=np.zeros((NCoords,NCoords))
        Start=0
        for p in range(NCoords):
            End=Start+p+1
            hessian[p,:p+1]=hessianElements[Start:End]
            Start=End
        # Symmetrize the Hessian (not necessary with linal.eigh)
        hessian=hessian+hessian.T-np.diag(np.diag(hessian))
    
        # Mass ponderation
        if mw:
            for i in range(NCoords):
                # in AMU
                # gradient[i]=gradient[i]/(np.sqrt(atomic_masses[i]))
                # in me
                gradient[i]=gradient[i]/(np.sqrt(atomic_masses[i]*CST.AMU_TO_ME))
            for i in range(NCoords):
                for j in range(NCoords):
                    # in AMU
                    # hessian[i,j]=hessian[i,j]/(np.sqrt(atomic_masses[i]*atomic_masses[j]))
                    # in me
                    hessian[i,j]=hessian[i,j]/(np.sqrt(atomic_masses[i]*CST.AMU_TO_ME*atomic_masses[j]*CST.AMU_TO_ME))
        return(ETot,gradient,hessian,atomic_masses,NAtoms)
    if not freq:
        if mw:
            for i in range(NCoords):
                # in AMU
                # gradient[i]=gradient[i]/(np.sqrt(atomic_masses[i]))
                # in me
                gradient[i]=gradient[i]/(np.sqrt(atomic_masses[i]*CST.AMU_TO_ME))
        return(ETot,gradient,atomic_masses,NAtoms)

def fchk2spectra(filename,normalized=False,stick_spectrum=False,fig=None,ax=None,units="rcm",secondary_axis=False,color=None,label=None,formatting=True,spectrum_max=None,stick_max=None,shift_intensity=0):
    """
    Takes as input:
    - the fchk of a FC(HT) calculation from Gaussian
    - normalized=True if spectra are to be normalized to max. intensity
      normalization is only "spectrum by spectrum" and not global to successive plots
    - stick_spectrum=True if stick spectrum is required
    - fig, ax not None if spectra are to be added to an existing fig, ax 
    - units={rcm(default),ev,nm}
    - secondary_axis=False if spectra are to be plotted in a secondary axis with respect 
      to the one given
    - color=None(black for stick spectrum) or color(with alpha=0.5 for stick spectrum)
    - label=None or str added to legend of ax(ax2 if secondary_axis)
    and returns:
    - fig, ax(ax2), figure and ax where the spectra are plotted (if secondary_axis)
    """
    with open(filename,"r") as f:
        lines=f.readlines()
    FCHTRAssign=[]
    FCHTSpectra=[]
    for i,line in enumerate(lines):
        line=line.split()
        if len(line)>2 and line[0]=="SCF" and line[1]=="Energy":
            SCF_Energy=float(line[-1])
        if len(line)>2 and line[0]=="FCHT" and line[1]=="RAssign":
            NElements=int(line[-1])
            for p in range(NElements):
                FCHTRAssign.append(lines[i+1+p//5].split()[p%5])
        if len(line)>2 and line[0]=="FCHT" and line[1]=="Spectra":
            NElements=int(line[-1])
            for p in range(NElements):
                FCHTSpectra.append(lines[i+1+p//5].split()[p%5])
    FCHTRAssign=np.array(FCHTRAssign,dtype=float)
    FCHTSpectra=np.array(FCHTSpectra,dtype=float)

    sticks=np.copy(FCHTRAssign)
    NSticks=len(sticks)//3
    sticks=sticks.reshape((NSticks,3))
    sticks_Energy=sticks[:,0]
    sticks_Intensity=sticks[:,1]
    sticks_DipStr=sticks[:,2]

    spectrum=np.copy(FCHTSpectra)
    NPoints=len(spectrum)//2
    spectrum_Energy=spectrum[:NPoints]
    spectrum_Intensity=spectrum[NPoints:]

    if units=="rcm":
        spectrum_Energy=np.copy(spectrum_Energy)
        sticks_Energy=np.copy(sticks_Energy)
    if units=="ev":
        spectrum_Energy=np.copy(spectrum_Energy)/CST.EV_TO_RCM
        sticks_Energy=np.copy(sticks_Energy)/CST.EV_TO_RCM
    if units=="nm":
        spectrum_Energy=(1/np.copy(spectrum_Energy))*CST.RNM_TO_RCM
        sticks_Energy=(1/np.copy(sticks_Energy))*CST.RNM_TO_RCM
    if normalized:
        # spectrum_Intensity/=np.max(spectrum_Intensity)
        # sticks_Intensity/=np.max(sticks_Intensity)
        if spectrum_max is None:
            spectrum_max=np.max(spectrum_Intensity)
        else:
            spectrum_max=max(spectrum_max,np.max(spectrum_Intensity))
        if stick_max is None:
            stick_max=np.max(sticks_Intensity)
        else:
            stick_max=max(stick_max,np.max(sticks_Intensity))
        spectrum_Intensity/=spectrum_max
        sticks_Intensity/=stick_max
        if shift_intensity!=0:
            sticks_Intensity+=shift_intensity
            spectrum_Intensity+=shift_intensity
    if fig is None:
        fig=plt.figure()
    if ax is None:
        ax=fig.add_subplot(111)
    if not secondary_axis:
        if color is None:
            ax.plot(spectrum_Energy,spectrum_Intensity,label=label)
        else:
            ax.plot(spectrum_Energy,spectrum_Intensity,color=color,label=label)
        if label is not None:
            ax.legend(loc="upper left")
        if stick_spectrum:
            for _ in range(NSticks):
                if color is None:
                    # ax.plot([sticks_Energy[_]]*2,[0,sticks_Intensity[_]],color="black")
                    ax.plot([sticks_Energy[_]]*2,[shift_intensity,sticks_Intensity[_]],color="black")
                else:
                    # ax.plot([sticks_Energy[_]]*2,[0,sticks_Intensity[_]],color=color,alpha=0.5)
                    ax.plot([sticks_Energy[_]]*2,[shift_intensity,sticks_Intensity[_]],color=color,alpha=0.5)
            ax.axhline(y=shift_intensity,color="black",linewidth=0.45)
    if secondary_axis:
        ax2=ax.twinx()
        if color is None:
            ax2.plot(spectrum_Energy,spectrum_Intensity,label=label)
        else:
            ax2.plot(spectrum_Energy,spectrum_Intensity,color=color,label=label)
        if label is not None:
            ax2.legend(loc="upper right")
        if stick_spectrum:
            for _ in range(NSticks):
                if color is None:
                    ax2.plot([sticks_Energy[_]]*2,[0,sticks_Intensity[_]],color="black")
                else:
                    ax2.plot([sticks_Energy[_]]*2,[0,sticks_Intensity[_]],color=color,alpha=0.5)
            ax2.axhline(y=shift_intensity,color="black",linewidth=0.45)

    if units=="rcm":
        if formatting:
            ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        else:
            ax.set_xlabel("Wavenumber (rcm)")
    elif units=="ev":
        ax.set_xlabel("Energy (eV)")
    elif units=="nm":
        ax.set_xlabel("Wavelength (nm)")

    if not secondary_axis:
        if normalized:
            ax.set_ylabel("Normalized intensity")
        else:
            ax.set_ylabel("Intensity")
        return fig,ax,spectrum_max,stick_max
    if secondary_axis:
        if normalized:
            ax2.set_ylabel("Normalized intensity")
        else:
            ax2.set_ylabel("Intensity")
        return fig,ax2,spectrum_max,stick_max

def num_BS(filetagS1,filetagS2,mw=True):
    """
    Takes is inputs the filetagSi corresponding to filetags for the two states of interest.
    Computes the fchk files from chk if not already there.
    Returns: the eigenvectors of the two biggest eigenvalues of the Hessian of the squared half energy difference.
    Not sure: Eigenvectors should be without dimensions, the dimensionality being in the eigenvalues
    """
    if not os.path.isfile(filetagS1+'.fchk'):
        if os.path.isfile(filetagS1+'.chk'):
            os.system('formchk16 '+filetagS1+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS1)
    # Read the fchk file (formatted with "formchk16 file.chk")
    file=filetagS1+'.fchk'
    # file="m22_preCoIn_freq_S1.fchk"
    #lines2file with open(file,"r") as f:
        #lines2file lines=np.array(f.readlines())
    ES1,gradient_S1,hessian_S1,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw)
    if not os.path.isfile(filetagS2+'.fchk'):
        if os.path.isfile(filetagS2+'.chk'):
            os.system('formchk16 '+filetagS2+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS2)
    # Read the fchk file (formatted with "formchk16 file.chk")
    file=filetagS2+'.fchk'
    #lines2file with open(file,"r") as f:
        #lines2file lines=np.array(f.readlines())
    ES2,gradient_S2,hessian_S2,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw)

    # Building the matrix of squared energy difference
    Delta=(ES2-ES1)/2
    hessianDelta=(hessian_S2-hessian_S1)/2
    gradientDelta=(gradient_S2-gradient_S1)/2
    gradientDeltaProduct=np.tensordot(gradientDelta,gradientDelta,axes=0)
    hessianSqDelta=2*Delta*hessianDelta+2*gradientDeltaProduct
    # Diagonalize the matrix
    eigval,diagonalizer=linalg.eigh(hessianSqDelta)
    eigvec=diagonalizer.T
    # Compute Branching Space vectors
    BS1=eigvec[-1]
    BS2=eigvec[-2]
    return (BS1.reshape((NAtoms,3)),
            BS2.reshape((NAtoms,3))
            )

def gradientDifference(filetagS1,filetagS2,mw=True,freq=True,half=True):
    # print(freq)
    """
    Takes is inputs the filetagSi corresponding to filetags for the two states of interest.
    Computes the gradient of the energy difference.
    Returns:
    - the energy difference (in Hartree)
    - the gradient of the energy difference (in Hartree/Bohr)
    """
    if not os.path.isfile(filetagS1+'.fchk'):
        if os.path.isfile(filetagS1+'.chk'):
            os.system('formchk16 '+filetagS1+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS1)
    # Read the fchk file (formatted with "formchk16 file.chk")
    file=filetagS1+'.fchk'
    # file="m22_preCoIn_freq_S1.fchk"
    #lines2file with open(file,"r") as f:
        #lines2file lines=np.array(f.readlines())
    if freq:
        ES1,gradient_S1,hessian_S1,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw,freq=freq)
    if not freq:
        ES1,gradient_S1,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw,freq=freq)
    if not os.path.isfile(filetagS2+'.fchk'):
        if os.path.isfile(filetagS2+'.chk'):
            os.system('formchk16 '+filetagS2+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS2)
    # Read the fchk file (formatted with "formchk16 file.chk")
    file=filetagS2+'.fchk'
    #lines2file with open(file,"r") as f:
        #lines2file lines=np.array(f.readlines())
    if freq:
        ES2,gradient_S2,hessian_S2,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw,freq=freq)
    if not freq:
        ES2,gradient_S2,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw,freq=freq)
    if not half:
        return (
            ES2-ES1,
            gradient_S2-gradient_S1
        )
    if half:
        return (
            0.5*(ES2-ES1),
            0.5*(gradient_S2-gradient_S1)
        )

def projectionOutBranchingSpace(filetagS1,filetagS2,state=2,mw=True,norm=True):
    """
    Takes as input the filetagSi associated to the states considered.
    Computes the gradient of the energy of the state=n (n=2 default), projected out of the Numerical Branching Space at this point, in Hartree/bohr
    Note that calculation is better to be run at a Conical Intersection, where the NBS as a sense and is valid.
    """
    # Get the numerical Branching Space
    BS1,BS2=num_BS(filetagS1,filetagS2,mw=mw)
    BS1=BS1.flatten()
    BS2=BS2.flatten()
    projectorBS1=np.tensordot(BS1,BS1,axes=0)
    projectorBS2=np.tensordot(BS2,BS2,axes=0)
    # Get the gradient of the i-th state
    if state==2:
        filetag=filetagS2
    elif state==1:
        filetag=filetagS1
    else:
        raise ValueError("Selected state not S1 or S2")
    if not os.path.isfile(filetag+'.fchk'):
        if os.path.isfile(filetag+'.chk'):
            os.system('formchk16 '+filetag+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetag)
    # Read the fchk file (formatted with "formchk16 file.chk")
    file=filetag+'.fchk'
    #lines2file with open(file,"r") as f:
        #lines2file lines=np.array(f.readlines())
    E,gradient,hessian,atomic_masses,NAtoms=fchk2derivatives(file,mw=mw)
    projectedOutGradient=(
        gradient
        -np.dot(projectorBS1,gradient)
        -np.dot(projectorBS2,gradient)
        )
    if norm:
        projectedOutGradient=projectedOutGradient/(np.sqrt(np.dot(projectedOutGradient,projectedOutGradient)))
    return projectedOutGradient

## STEP 0 ##
# with open("step_0.xyz","r") as f:
    # coordinates=f.readlines()[2:]
# coordinates_text=np.array([line.split() for line in coordinates])
# C_coordinates=coordinates_text[coordinates_text[:,0]=='C'][:,1:].astype(float)
# coordinates=coordinates_text[:,1:].astype(float)
# NAtoms=len(coordinates)
#
# check_3D=True
# check_3D=False
# if check_3D:
    # fig=plt.figure()
    # ax=fig.add_subplot(111,projection="3d")
    # ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
    # fraction=1
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.set_zlim(-0.2,0.2)
    # plt.show()

def makeDeltaE(old_step,new_step,energyDifferenceThs=2e-5,check_3D=False,produce_step=True,freq=False,half=True):
    """
    Takes old_step as geometry input and new_step as the name of the next input to produce.
    energyDifferenceThs is the threshold one wants for the energyDifference 2·ΔE=E2-E1 (in the notation of Gonon et al., and others, ΔE is the halvedEnergyDifference)
    optional arguments include:
    - check_3D, checking the form of the normed gradient difference vector on the input geometry;
    - produce_step, producing the next step;
    - freq, if True uses frequency calculations fchk files, if not uses single-point electronic calculations (these SPE must have "Force" keywords to print the gradients);
    - halv, in general has to be true to match the litterature definition of the gradient difference (which is the halved gradient difference).
    """
    with open("step_"+str(old_step)+".xyz","r") as f:
        coordinates=f.readlines()[2:]
    coordinates_text=np.array([line.split() for line in coordinates])
    C_coordinates=coordinates_text[coordinates_text[:,0]=='C'][:,1:].astype(float)
    coordinates=coordinates_text[:,1:].astype(float)
    NAtoms=len(coordinates)

    if freq:
        filetagS1="step_"+str(old_step)+"_freq_S1"
        filetagS2="step_"+str(old_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(old_step)+"_1"
        filetagS2="step_"+str(old_step)+"_2"
    halvedEnergyDifference,halvedGradientDifferenceVector=gradientDifference(filetagS1,filetagS2,mw=False,freq=freq,half=True)
    normHalvedGradientDifference=np.sqrt(np.dot(halvedGradientDifferenceVector,halvedGradientDifferenceVector))
    halvedGradientDifferenceVector=halvedGradientDifferenceVector/normHalvedGradientDifference

    halvedEnergyDifferenceThs=0.5*energyDifferenceThs
    energy_step=halvedEnergyDifferenceThs-halvedEnergyDifference
    space_step=energy_step/normHalvedGradientDifference # bohr
    space_step=space_step*CST.BOHR_TO_ANGSTROM # angstrom
    if check_3D:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
        halvedGradientDifferenceVector=halvedGradientDifferenceVector.reshape((NAtoms,3))
        fraction=1
        disp_coordinates=coordinates+fraction*halvedGradientDifferenceVector
        ax.scatter(disp_coordinates[:,0],disp_coordinates[:,1],disp_coordinates[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zlim(-0.2,0.2)
        plt.title("Gradient at old_step "+str(old_step)+" to new_step "+str(new_step))
        plt.show()

    displacement=(space_step*halvedGradientDifferenceVector).reshape((NAtoms,3))
    if produce_step:
        with open("step_"+str(new_step)+".xyz","w") as f:
            displaced_coordinates=coordinates+displacement
            f.write(str(len(displaced_coordinates)))
            f.write('\n')
            f.write("step_"+str(new_step)+'\n')
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    if produce_step:
        # First state
        with open("step_"+str(new_step)+"_1.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_1.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            f.write("# geom=NoCrowd sym=com Force cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        with open("step_"+str(new_step)+"_freq_S1.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_freq_S1.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            f.write("# geom=NoCrowd sym=com freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        # Second state
        with open("step_"+str(new_step)+"_2.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_2.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            f.write("# geom=NoCrowd sym=com Force cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        with open("step_"+str(new_step)+"_freq_S2.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_freq_S2.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            f.write("# geom=NoCrowd sym=com freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')


def makeNewStep(old_step,new_step,check_3D=False,produce_step=True,fraction_of_gradient=0.1,limit_z=True,limit_x=False,numerical_branching_space=True,fraction_fixed=False,space_step=0.05,freq=True,check_gradient=False):
    with open("step_"+str(old_step)+".xyz","r") as f:
        coordinates=f.readlines()[2:]
    coordinates_text=np.array([line.split() for line in coordinates])
    C_coordinates=coordinates_text[coordinates_text[:,0]=='C'][:,1:].astype(float)
    coordinates=coordinates_text[:,1:].astype(float)
    NAtoms=len(coordinates)

    if freq:
        filetagS1="step_"+str(old_step)+"_freq_S1"
        filetagS2="step_"+str(old_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(old_step)+"_1"
        filetagS2="step_"+str(old_step)+"_2"
    if numerical_branching_space:
        BS1,BS2=num_BS(filetagS1,filetagS2,mw=False)
        BS1=BS1.flatten()
        BS2=BS2.flatten()
        projectedOutGradient=projectionOutBranchingSpace(filetagS1,filetagS2,mw=False,norm=False)

        normProjectedOutGradient=np.sqrt(np.dot(projectedOutGradient,projectedOutGradient))
        projectedOutGradient=projectedOutGradient/normProjectedOutGradient
    if not numerical_branching_space:
        energyDifference,gradientDifferenceVector=gradientDifference(filetagS1,filetagS2,mw=False,freq=freq,half=True)
        normGradientDifference=np.sqrt(np.dot(gradientDifferenceVector,gradientDifferenceVector))
        gradientDifferenceVector=gradientDifferenceVector/normGradientDifference
        projectorGradientDifferenceVector=np.tensordot(gradientDifferenceVector,gradientDifferenceVector,axes=0)
        if not os.path.isfile(filetagS2+'.fchk'):
            if os.path.isfile(filetagS2+'.chk'):
                os.system('formchk16 '+filetagS2+'.chk')
            else:
                raise ValueError("Error, chk file not found for "+filetagS2)
        # Read the fchk file (formatted with "formchk16 file.chk")
        #lines2file with open(filetagS2+".fchk","r") as f:
            #lines2file lines=np.array(f.readlines())
        if freq:
            E,gradient,hessian,atomic_masses,NAtoms=fchk2derivatives(filetagS2+".fchk",mw=False,freq=freq)
        if not freq:
            E,gradient,atomic_masses,NAtoms=fchk2derivatives(filetagS2+".fchk",mw=False,freq=freq)

        projectedOutGradient=(
        gradient
        # -(np.dot(gradientDifferenceVector,gradient))*gradientDifferenceVector
        -np.dot(projectorGradientDifferenceVector,gradient)
        )

        normProjectedOutGradient=np.sqrt(np.dot(projectedOutGradient,projectedOutGradient)) # E_h/a_0 Hartree/Bohr
        projectedOutGradient=projectedOutGradient/normProjectedOutGradient

    if not fraction_fixed:
        space_step=-fraction_of_gradient*normProjectedOutGradient # 
    if fraction_fixed:
        space_step=-space_step
    if check_3D:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
        # BS1_disp=BS1.reshape((NAtoms,3))
        # BS2_disp=BS2.reshape((NAtoms,3))
        projectedOutGradient=projectedOutGradient.reshape((NAtoms,3))
        fraction=1
        # fraction=space_step
        # disp_coordinates=coordinates+fraction*BS1_disp
        # disp_coordinates=coordinates+fraction*BS2_disp
        disp_coordinates=coordinates+fraction*projectedOutGradient
        ax.scatter(disp_coordinates[:,0],disp_coordinates[:,1],disp_coordinates[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if limit_z:
            ax.set_zlim(-0.2,0.2)
        if limit_x:
            ax.set_xlim(-0.2,0.2)
        # ax.set_title("old_step "+str(old_step)+" new_step "+str(new_step))
        plt.title("old_step "+str(old_step)+" new_step "+str(new_step))
        plt.show()

    displacement=(space_step*projectedOutGradient).reshape((NAtoms,3))
    if produce_step:
        with open("step_"+str(new_step)+".xyz","w") as f:
            displaced_coordinates=coordinates+displacement
            f.write(str(len(displaced_coordinates)))
            f.write('\n')
            f.write("step_"+str(new_step)+'\n')
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    if produce_step:
        # First state
        with open("step_"+str(new_step)+"_1.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_1.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            # f.write("# geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write("# geom=NoCrowd sym=(com) Force cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        if freq:
            with open("step_"+str(new_step)+"_freq_S1.com","w") as f:
                f.write("%chk=step_"+str(new_step)+"_freq_S1.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                # f.write("# geom=NoCrowd nosym freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write("# geom=NoCrowd sym=(com) freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=coordinates+displacement
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if p<22:
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
        # Second state
        with open("step_"+str(new_step)+"_2.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_2.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            # f.write("# geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write("# geom=NoCrowd sym=(com) Force cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        if freq:
            with open("step_"+str(new_step)+"_freq_S2.com","w") as f:
                f.write("%chk=step_"+str(new_step)+"_freq_S2.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                # f.write("# geom=NoCrowd nosym freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write("# geom=NoCrowd sym=(com) freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=coordinates+displacement
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if p<22:
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
    if not check_gradient:
        return normProjectedOutGradient*projectedOutGradient
    if check_gradient:
        return (normProjectedOutGradient*projectedOutGradient,
                gradient,
                # (np.dot(gradientDifferenceVector,gradient))*gradientDifferenceVector)
                np.dot(projectorGradientDifferenceVector,gradient))

def makeComposite(old_step,new_step,check_3D=False,produce_step=True,fraction_of_gradient=0.1,limit_z=True,limit_x=False,numerical_branching_space=True,fraction_fixed=False,space_step=0.05,freq=True,check_gradient=False,energyDifferenceThs=0,half=True):
    with open("step_"+str(old_step)+".xyz","r") as f:
        coordinates=f.readlines()[2:]
    coordinates_text=np.array([line.split() for line in coordinates])
    C_coordinates=coordinates_text[coordinates_text[:,0]=='C'][:,1:].astype(float)
    coordinates=coordinates_text[:,1:].astype(float)
    NAtoms=len(coordinates)

    if freq:
        filetagS1="step_"+str(old_step)+"_freq_S1"
        filetagS2="step_"+str(old_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(old_step)+"_1"
        filetagS2="step_"+str(old_step)+"_2"
    if numerical_branching_space:
        BS1,BS2=num_BS(filetagS1,filetagS2,mw=False)
        BS1=BS1.flatten()
        BS2=BS2.flatten()
        projectedOutGradient=projectionOutBranchingSpace(filetagS1,filetagS2,mw=False,norm=False)

        normProjectedOutGradient=np.sqrt(np.dot(projectedOutGradient,projectedOutGradient))
        projectedOutGradient=projectedOutGradient/normProjectedOutGradient
    if not numerical_branching_space:
        energyDifference,gradientDifferenceVector=gradientDifference(filetagS1,filetagS2,mw=False,freq=freq,half=True)
        normGradientDifference=np.sqrt(np.dot(gradientDifferenceVector,gradientDifferenceVector))
        gradientDifferenceVector=gradientDifferenceVector/normGradientDifference
        projectorGradientDifferenceVector=np.tensordot(gradientDifferenceVector,gradientDifferenceVector,axes=0)
        if not os.path.isfile(filetagS2+'.fchk'):
            if os.path.isfile(filetagS2+'.chk'):
                os.system('formchk16 '+filetagS2+'.chk')
            else:
                raise ValueError("Error, chk file not found for "+filetagS2)
        # Read the fchk file (formatted with "formchk16 file.chk")
        #lines2file with open(filetagS2+".fchk","r") as f:
            #lines2file lines=np.array(f.readlines())
        if freq:
            E,gradient,hessian,atomic_masses,NAtoms=fchk2derivatives(filetagS2+".fchk",mw=False,freq=freq)
        if not freq:
            E,gradient,atomic_masses,NAtoms=fchk2derivatives(filetagS2+".fchk",mw=False,freq=freq)

        projectedOutGradient=(
        gradient
        # -(np.dot(gradientDifferenceVector,gradient))*gradientDifferenceVector
        -np.dot(projectorGradientDifferenceVector,gradient)
        )

        normProjectedOutGradient=np.sqrt(np.dot(projectedOutGradient,projectedOutGradient)) # E_h/a_0 Hartree/Bohr
        projectedOutGradient=projectedOutGradient/normProjectedOutGradient

    if not fraction_fixed:
        space_step=-fraction_of_gradient*normProjectedOutGradient # 
    if fraction_fixed:
        space_step=-space_step*CST.BOHR_TO_ANGSTROM
    if check_3D:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
        # BS1_disp=BS1.reshape((NAtoms,3))
        # BS2_disp=BS2.reshape((NAtoms,3))
        projectedOutGradient=projectedOutGradient.reshape((NAtoms,3))
        fraction=1
        # fraction=space_step
        # disp_coordinates=coordinates+fraction*BS1_disp
        # disp_coordinates=coordinates+fraction*BS2_disp
        disp_coordinates=coordinates+fraction*projectedOutGradient
        ax.scatter(disp_coordinates[:,0],disp_coordinates[:,1],disp_coordinates[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if limit_z:
            ax.set_zlim(-0.2,0.2)
        if limit_x:
            ax.set_xlim(-0.2,0.2)
        # ax.set_title("old_step "+str(old_step)+" new_step "+str(new_step))
        plt.title("old_step "+str(old_step)+" new_step "+str(new_step))
        plt.show()

    if freq:
        filetagS1="step_"+str(old_step)+"_freq_S1"
        filetagS2="step_"+str(old_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(old_step)+"_1"
        filetagS2="step_"+str(old_step)+"_2"
    halvedEnergyDifference,halvedGradientDifferenceVector=gradientDifference(filetagS1,filetagS2,mw=False,freq=freq,half=True)
    normHalvedGradientDifference=np.sqrt(np.dot(halvedGradientDifferenceVector,halvedGradientDifferenceVector))
    halvedGradientDifferenceVector=halvedGradientDifferenceVector/normHalvedGradientDifference

    halvedEnergyDifferenceThs=0.5*energyDifferenceThs
    energy_step=halvedEnergyDifferenceThs-halvedEnergyDifference
    # energy_step=-halvedEnergyDifference
    space_step_GD=energy_step/normHalvedGradientDifference # bohr
    space_step_GD=space_step_GD*CST.BOHR_TO_ANGSTROM # angstrom

    facGD=1
    # facGD=0.1
    facPGD=1
    displacement=(facPGD*space_step*projectedOutGradient+facGD*space_step_GD*halvedGradientDifferenceVector).reshape((NAtoms,3))
    if produce_step:
        with open("step_"+str(new_step)+".xyz","w") as f:
            displaced_coordinates=coordinates+displacement
            f.write(str(len(displaced_coordinates)))
            f.write('\n')
            f.write("step_"+str(new_step)+'\n')
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    if produce_step:
        # First state
        with open("step_"+str(new_step)+"_1.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_1.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            # f.write("# geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write("# geom=NoCrowd sym=(com) Force cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        if freq:
            with open("step_"+str(new_step)+"_freq_S1.com","w") as f:
                f.write("%chk=step_"+str(new_step)+"_freq_S1.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                # f.write("# geom=NoCrowd nosym freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write("# geom=NoCrowd sym=(com) freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=1,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=coordinates+displacement
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if p<22:
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
        # Second state
        with open("step_"+str(new_step)+"_2.com","w") as f:
            f.write("%chk=step_"+str(new_step)+"_2.chk"+'\n')
            f.write("%mem=16GB"+'\n')
            f.write("%nprocshared=16"+'\n')
            # f.write("# geom=NoCrowd nosym cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write("# geom=NoCrowd sym=(com) Force cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
            f.write('\n')
            f.write("m22 CoIn opt step"+str(new_step)+'\n')
            f.write('\n')
            f.write("0 1"+'\n')
            displaced_coordinates=coordinates+displacement
            for p in range(len(displaced_coordinates)):
                to_print=displaced_coordinates[p]
                to_print=["%.16f" % _ for _ in to_print]
                if p<22:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                else:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
            f.write('\n')
            f.write('\n')
        if freq:
            with open("step_"+str(new_step)+"_freq_S2.com","w") as f:
                f.write("%chk=step_"+str(new_step)+"_freq_S2.chk"+'\n')
                f.write("%mem=16GB"+'\n')
                f.write("%nprocshared=16"+'\n')
                # f.write("# geom=NoCrowd nosym freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write("# geom=NoCrowd sym=(com) freq=(savenm,hpmodes) cam-b3lyp/6-31+g(d) td(root=2,nstate=5) scf=(conver=10,novaracc) integral(grid=ultrafine)"+'\n')
                f.write('\n')
                f.write("m22 CoIn opt-freq step"+str(new_step)+'\n')
                f.write('\n')
                f.write("0 1"+'\n')
                displaced_coordinates=coordinates+displacement
                for p in range(len(displaced_coordinates)):
                    to_print=displaced_coordinates[p]
                    to_print=["%.16f" % _ for _ in to_print]
                    if p<22:
                        f.write("C "+" ".join(to_print))
                        f.write('\n')
                    else:
                        f.write("H "+" ".join(to_print))
                        f.write('\n')
                f.write('\n')
                f.write('\n')
    if not check_gradient:
        return normProjectedOutGradient*projectedOutGradient
    if check_gradient:
        return (normProjectedOutGradient*projectedOutGradient,
                gradient,
                # (np.dot(gradientDifferenceVector,gradient))*gradientDifferenceVector)
                np.dot(projectorGradientDifferenceVector,gradient))


def convergenceTest(old_step,new_step,standard=False,freq=True):
    ## Old step ##
    if freq:
        filetagS1="step_"+str(old_step)+"_freq_S1"
        filetagS2="step_"+str(old_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(old_step)+"_1"
        filetagS2="step_"+str(old_step)+"_2"
    if not os.path.isfile(filetagS1+'.fchk'):
        if os.path.isfile(filetagS1+'.chk'):
            os.system('formchk16 '+filetagS1+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS1)
    if not os.path.isfile(filetagS2+'.fchk'):
        if os.path.isfile(filetagS2+'.chk'):
            os.system('formchk16 '+filetagS2+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS2)
    # Get coordinates and write them if not filed
    with open(filetagS1+".log","r") as f:
        lines=f.readlines()
    coordinates=[]
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line) > 1 and line[0]=="NAtoms=":
            NAtoms=int(line[1])
    with open(filetagS1+".log","r") as f:
        lines=f.readlines()
    coordinates=[]
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line) > 1 and line[0]=="NAtoms=":
            NAtoms=int(line[1])
            # NModes=int(3*NAtoms-6)
            # NCoords=int(3*NAtoms)
        if standard and len(line) > 1 and line[0]=="Standard" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
        if not standard and len(line) > 1 and line[0]=="Input" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
    if not os.path.isfile("step_"+str(old_step)+'.xyz'):
        with open(filetag+"_log2xyz.xyz","w") as f:
            f.write(str(len(coordinates)))
            f.write('\n')
            f.write(filetag+" output geometry"+'\n')
            for p in range(len(coordinates)):
                to_print=coordinates[p,3:].astype(float)
                to_print=["%.16f" % _ for _ in to_print]
                if int(coordinates[p,1])==6:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                elif int(coordinates[p,1])==1:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    oldCoordinates=np.copy(coordinates)
    if freq:
        oldProjectedOutGradient=projectionOutBranchingSpace(filetagS1,filetagS2,mw=False,norm=False)
    oldEnergyDifference=gradientDifference(filetagS1,filetagS2,mw=False,freq=False)[0]
    # New step
    if freq:
        filetagS1="step_"+str(new_step)+"_freq_S1"
        filetagS2="step_"+str(new_step)+"_freq_S2"
    if not freq:
        filetagS1="step_"+str(new_step)+"_1"
        filetagS2="step_"+str(new_step)+"_2"
    if not os.path.isfile(filetagS1+'.fchk'):
        if os.path.isfile(filetagS1+'.chk'):
            os.system('formchk16 '+filetagS1+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS1)
    if not os.path.isfile(filetagS2+'.fchk'):
        if os.path.isfile(filetagS2+'.chk'):
            os.system('formchk16 '+filetagS2+'.chk')
        else:
            raise ValueError("Error, chk file not found for "+filetagS2)
    # Get coordinates and write them if not filed
    with open(filetagS1+".log","r") as f:
        lines=f.readlines()
    coordinates=[]
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line) > 1 and line[0]=="NAtoms=":
            NAtoms=int(line[1])
            # NModes=int(3*NAtoms-6)
            # NCoords=int(3*NAtoms)
        if standard and len(line) > 1 and line[0]=="Standard" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
        if not standard and len(line) > 1 and line[0]=="Input" and line[1]=="orientation:":
            current_coordinates=[]
            for p in range(NAtoms):
                current_coordinates.append(lines[i+5+p].split())
            coordinates=np.array(current_coordinates)
    if not os.path.isfile("step_"+str(old_step)+'.xyz'):
        with open(filetag+"_log2xyz.xyz","w") as f:
            f.write(str(len(coordinates)))
            f.write('\n')
            f.write(filetag+" output geometry"+'\n')
            for p in range(len(coordinates)):
                to_print=coordinates[p,3:].astype(float)
                to_print=["%.16f" % _ for _ in to_print]
                if int(coordinates[p,1])==6:
                    f.write("C "+" ".join(to_print))
                    f.write('\n')
                elif int(coordinates[p,1])==1:
                    f.write("H "+" ".join(to_print))
                    f.write('\n')
    newCoordinates=np.copy(coordinates)
    if freq:
        newProjectedOutGradient=projectionOutBranchingSpace(filetagS1,filetagS2,mw=False,norm=False)
    newEnergyDifference=gradientDifference(filetagS1,filetagS2,mw=False,freq=False)[0]
    if freq:
        return(
            oldCoordinates,
            newCoordinates,
            oldProjectedOutGradient,
            newProjectedOutGradient,
            oldEnergyDifference,
            newEnergyDifference
        )
    if not freq:
        return(
            oldCoordinates,
            newCoordinates,
            oldEnergyDifference,
            newEnergyDifference
        )

def MECI_extract(filetag,only_title=False):
    if only_title:
        print("file","ES1","ES2","MeanE","DeltaE","θ(G,GD)","θ(MG,PG)","θ(MG,GD)","RMSGradient","RMSGD")
    else:
        filetoopen=filetag+'_1.fchk'
        #lines2file with open(filetoopen,"r") as f:
            #lines2file lines=np.array(f.readlines())
        energyS1,gradientS1=fchk2derivatives(filetoopen,mw=False,freq=False)[0:2]
        filetoopen=filetag+'_2.fchk'
        #lines2file with open(filetoopen,"r") as f:
            #lines2file lines=np.array(f.readlines())
        energyS2,gradientS2=fchk2derivatives(filetoopen,mw=False,freq=False)[0:2]
        currentEnergyDifference=energyS2-energyS1
        meanGradient=0.5*(gradientS1+gradientS2)
        maxMG=np.max(np.abs(meanGradient))
        RMSMG=np.sqrt(np.sum(meanGradient**2))
        currentGradientDifference=gradientS2-gradientS1
        maxGD=np.max(np.abs(currentGradientDifference))
        RMSGD=np.sqrt(np.sum(currentGradientDifference**2))
        currentProjectedOutGradient=(meanGradient-np.dot(meanGradient,currentGradientDifference/RMSGD)*currentGradientDifference/RMSGD)
        currentProjectedGradient=np.dot(meanGradient,currentGradientDifference/RMSGD)*currentGradientDifference/RMSGD
        if currentEnergyDifference>5e-5:
            currentFactorGradientDifference=1
        else:
            currentFactorGradientDifference=0
        currentGradient=(currentProjectedOutGradient+currentFactorGradientDifference*2*(currentEnergyDifference)*currentGradientDifference/RMSGD)
        maxGradient=np.max(np.abs(currentGradient))
        RMSGradient=np.sqrt(np.sum((currentGradient)**2))
        thetaGGD=180/np.pi*np.arccos(np.dot(currentGradient.flatten()/RMSGradient,currentGradientDifference.flatten()/RMSGD)) # Angle entre le gradient total G et la différence des gradients GD
        thetaMGPG=180/np.pi*np.arccos(np.dot(meanGradient.flatten()/RMSMG,currentProjectedGradient.flatten()/(np.sqrt(np.sum(currentProjectedGradient**2))))) # Angle entre la moyenne des gradients et la projection de la moyenne des gradients sur la différence des gradients
        thetaMGGD=180/np.pi*np.arccos(np.dot(meanGradient.flatten()/RMSMG,currentGradientDifference.flatten()/RMSGD)) # Angle entre la moyenne des gradients et la différence des gradients
        print(filetag,energyS1,energyS2,0.5*(energyS1+energyS2),energyS2-energyS1,thetaGGD,thetaMGPG,thetaMGGD,RMSGradient,RMSGD)

def getNormalModesVibration(filetag,vibrationalBasis=False):
    #lines2file with open(filetag+".fchk","r") as f:
        #lines2file lines=f.readlines()
    currentEnergy,currentGradient,currentHessian,atomicMasses=fchk2derivatives(filetag+".fchk",mw=False,freq=True)[:4] # in E_h, E_h/(a_0), E_h/(a_0²)
    NCoords=len(atomicMasses)
    NAtoms=NCoords//3
    NModes=3*NAtoms-6
    lines=[line.split() for line in lines]
    coordinates=np.array([])
    for i in range(len(lines)):
        line=lines[i]
        if len(line)>3 and line[1]=="cartesian" and line[2]=="coordinates":
            c=1
            while len(coordinates)!=NCoords:
                coordinates=np.append(coordinates,lines[i+c])
                c+=1
    currentCoordinates=coordinates.astype(float) # in bohr, a_0
    
    atomicMasses=atomicMasses*CST.AMU_TO_ME
    for i in range(len(atomicMasses)):
        for j in range(len(atomicMasses)):
            currentHessian[i,j]=currentHessian[i,j]/(np.sqrt(atomicMasses[i]*atomicMasses[j]))

    # Generate coordinates in the rotating and translating frame
    D1=np.zeros(NCoords).reshape(NAtoms,3)
    D1[:,0]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,0]
    D1=D1.flatten()
    D2=np.zeros(NCoords).reshape(NAtoms,3)
    D2[:,1]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,1]
    D2=D2.flatten()
    D3=np.zeros(NCoords).reshape(NAtoms,3)
    D3[:,2]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,2]
    D3=D3.flatten()
    # in a_0·sqrt(m_e)
    D1=D1/np.sqrt(np.sum(D1**2))
    D2=D2/np.sqrt(np.sum(D2**2))
    D3=D3/np.sqrt(np.sum(D3**2))
    # normalized
    
    centerOfMass=np.sum(atomicMasses.reshape(NAtoms,3)*currentCoordinates.reshape(NAtoms,3),axis=0)/np.sum(atomicMasses.reshape(NAtoms,3))
    
    currentCoordinates=currentCoordinates.reshape(NAtoms,3) # Bohr, a_0
    masses=atomicMasses.reshape(NAtoms,3)[:,0]
    x=currentCoordinates[:,0]-centerOfMass[0]
    y=currentCoordinates[:,1]-centerOfMass[1]
    z=currentCoordinates[:,2]-centerOfMass[2]
    inertiaTensor=np.array(
            [
                [
                    np.sum(masses*(y*y+z*z)),
                    -np.sum(masses*x*y),
                    -np.sum(masses*x*z)
                    ],
                [
                    -np.sum(masses*y*x),
                    np.sum(masses*(x*x+z*z)),
                    -np.sum(masses*y*z)
                    ],
                [
                    -np.sum(masses*z*x),
                    -np.sum(masses*z*y),
                    np.sum(masses*(x*x+y*y))
                    ]
                ]
                ) # a_0^2·m_e
    # eigenvaluesInertiaTensor,diagonalizerInertiaTensor=scipy.linalg.eigh(inertiaTensor)
    eigenvaluesInertiaTensor,diagonalizerInertiaTensor=linalg.eigh(inertiaTensor)
    principalMoments=eigenvaluesInertiaTensor
    currentCoordinates=np.dot(currentCoordinates,diagonalizerInertiaTensor)
    
    atomicMasses=atomicMasses.reshape(NAtoms,3)
    D4=[]
    D5=[]
    D6=[]
    for i in range(NAtoms):
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[0,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[0,1])*np.sqrt(atomicMasses[i,0]))
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[1,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[1,1])*np.sqrt(atomicMasses[i,0]))
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[2,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[2,1])*np.sqrt(atomicMasses[i,0]))
    
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[0,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[0,2])*np.sqrt(atomicMasses[i,0]))
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[1,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[1,2])*np.sqrt(atomicMasses[i,0]))
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[2,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[2,2])*np.sqrt(atomicMasses[i,0]))
    
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[0,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[0,0])*np.sqrt(atomicMasses[i,0]))
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[1,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[1,0])*np.sqrt(atomicMasses[i,0]))
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[2,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[2,0])*np.sqrt(atomicMasses[i,0]))
    D4=np.array(D4)
    D5=np.array(D5)
    D6=np.array(D6)
    D4=D4/np.sqrt(np.sum(D4**2))
    D5=D5/np.sqrt(np.sum(D5**2))
    D6=D6/np.sqrt(np.sum(D6**2))
    D=np.array([D1,D2,D3,D4,D5,D6]).T
    # U,s,V=scipy.linalg.svd(D,full_matrices=True)
    U,s,V=linalg.svd(D,full_matrices=True)
    B=U[:,6:]
    
    currentHessianVib=np.dot(B.T,np.dot(currentHessian,B))
    
    # eigenvalues,diagonalizer=scipy.linalg.eigh(currentHessianVib)
    eigenvalues,diagonalizer=linalg.eigh(currentHessianVib)
    normalModes=np.dot(B,diagonalizer).T
    negativeFrequencies=-(CST.HARTREE_TO_EV/CST.EV_TO_NM*CST.RNM_TO_RCM)*np.sqrt(-eigenvalues[eigenvalues<0])
    positiveFrequencies=(CST.HARTREE_TO_EV/CST.EV_TO_NM*CST.RNM_TO_RCM)*np.sqrt(eigenvalues[eigenvalues>=0])
    frequencies=np.append(negativeFrequencies,positiveFrequencies)
    if not vibrationalBasis:
        return(frequencies,normalModes)
    if vibrationalBasis:
        return(frequencies,normalModes,B)

def reorderingNormalModes(initialNormalModes,finalNormalModes,check=True):
    """
    computes the overlap matrix between initial and final Normal Modes (3N vectors of 3N coordinates) 
    and returns 
        - the n-array ordering of finalNormalModes number overlapping the most with the initialNormalMode number 0...i...n-1;
        - the n²-array of finalNormalModes reordered using the n-array ordering.
    example: the ith element of ordering, ordering[i]=j, means that the vector i of initialNormalModes overlaps the most
    with vector j of finalNormalModes and conversely.
    """
    overlapMatrix=np.dot(initialNormalModes,finalNormalModes.T)**2
    orderingInitialInFinal=[]
    for n in range(len(initialNormalModes)):
        maxOverlapIndex=np.argmax(overlapMatrix[n,:])
        maxOverlap=np.max(overlapMatrix[n,:])
        if maxOverlapIndex in orderingInitialInFinal:
            overlapMatrixToDelete=np.copy(overlapMatrix)
            old_n=np.argmin((np.array(orderingInitialInFinal)-maxOverlapIndex)**2)
            alreadyOverlap=overlapMatrix[old_n,maxOverlapIndex]
            if alreadyOverlap>maxOverlap:
                overlapMatrixToDelete=np.delete(overlapMatrixToDelete,maxOverlapIndex,axis=1)
                oldOverlapIndex=maxOverlapIndex
                maxOverlapIndex=np.argmax(overlapMatrixToDelete[n,:])
                if maxOverlapIndex<oldOverlapIndex:
                    orderingInitialInFinal.append(maxOverlapIndex)
                else:
                    orderingInitialInFinal.append(maxOverlapIndex+1)
            elif alreadyOverlap<=maxOverlap:
                orderingInitialInFinal.append(maxOverlapIndex)
                overlapMatrixToDelete=np.delete(overlapMatrixToDelete,maxOverlapIndex,axis=1)
                oldOverlapIndex=maxOverlapIndex
                maxOverlapIndex=np.argmax(overlapMatrixToDelete[old_n,:])
                if maxOverlapIndex<oldOverlapIndex:
                    orderingInitialInFinal[old_n]=maxOverlapIndex
                else:
                    orderingInitialInFinal[old_n]=maxOverlapIndex+1
        else:
            orderingInitialInFinal.append(maxOverlapIndex)
    orderingInitialInFinal=np.array(orderingInitialInFinal)
    orderedFinalNormalModes=finalNormalModes[orderingInitialInFinal,:]
    if check:
        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ap1=ax1.imshow(np.dot(initialNormalModes,finalNormalModes.T)**2)
        divider = make_axes_locatable(ax1)
        colorbar_axes = divider.append_axes("right",
                                        size="5%",
                                        pad=0.1)
        # Using new axes for colorbar
        plt.colorbar(ap1, cax=colorbar_axes)
        ax1.set_xlabel("finalNormalModes")
        ax1.set_ylabel("initialNormalModes")
        ax2=fig.add_subplot(122)
        ap2=ax2.imshow(np.dot(initialNormalModes,orderedFinalNormalModes.T)**2)
        divider = make_axes_locatable(ax2)
        colorbar_axes = divider.append_axes("right",
                                        size="5%",
                                        pad=0.1)
        # Using new axes for colorbar
        plt.colorbar(ap2, cax=colorbar_axes)
        ax2.set_xlabel("orderedFinalNormalModes")
        ax2.set_ylabel("initialNormalModes")
        fig.tight_layout()
        plt.show()
    return(orderingInitialInFinal,orderedFinalNormalModes)

def orthogonalizeNormalModes(printedModesCart,reducedMasses,atomicMasses,check=True):
    normalModes=np.copy(printedModesCart)
    for i in range(len(reducedMasses)):
        for j in range(len(atomicMasses.flatten())):
            normalModes[i,j]=printedModesCart[i,j]*(np.sqrt(atomicMasses.flatten()[j])/np.sqrt(reducedMasses[i]*CST.AMU_TO_ME))
    if check:
        fig,ax1=plotGramMatrix(printedModesCart,nrows=2,ncols=1,index=1)
        fig,ax2=plotGramMatrix(normalModes,fig=fig,nrows=2,ncols=1,index=2)
        plt.show()
    return normalModes

def fchk2coordinates(filename):
    with open(filename,"r") as f:
        lines=f.readlines()
    coordinates=[]
    atomicNumbers=[]
    for i in range(len(lines)):
        line=lines[i].split()
        if len(line)>2 and line[0]=="Atomic":
            NAtoms=int(line[-1])
            NCoords=3*NAtoms
        if len(line)>3 and line[0]=="Atomic" and line[1]=="numbers":
            c=1
            while len(atomicNumbers)!=NAtoms:
                atomicNumbers=np.append(atomicNumbers,lines[i+c].split())
                c+=1
        if len(line)>3 and line[1]=="cartesian" and line[2]=="coordinates":
            c=1
            while len(coordinates)!=NCoords:
                coordinates=np.append(coordinates,lines[i+c].split())
                c+=1
    atomicNumbers=atomicNumbers.astype(int) # adimensional
    coordinates=coordinates.astype(float) # in bohr, a_0
    coordinates=coordinates*CST.BOHR_TO_ANGSTROM
    return(NAtoms,NCoords,atomicNumbers,coordinates)

def fchk2vibrationalAnalysis(filename,gaussianType=True,separate_TR=True):
    """
    Routine for the vibrational analysis of a Hessian single-point calculation.
    Takes as input:
    - filename, str for the fchk file for which the energy, 
      gradient, Hessian can be found
      along with NAtoms, NCoords, coordinates of the geometry
      and atomic numbers
    - the type of printedModes that are required
      (only Gaussian type for now)
    and returns
    - frequencies in reciprocal cm
    - reduced masses in a_0·sqrt(m_e)
    - printedModesCart
    - atomicNumbers
    - atomicMasses
    - coordinates in angstrom
    """
    # with open(filename,"r") as f:
        # lines=f.readlines()
    ETot,gradientInMWC,hessianInMWC,atomicMasses,NAtoms=fchk2derivatives(filename,mw=True,freq=True)
    NAtoms,NCoords,atomicNumbers,currentCoordinates=fchk2coordinates(filename)
    atomicMasses=atomicMasses*CST.AMU_TO_ME # amu->me

    # Generate coordinates in the translating and rotating frame
    # Generate translational modes
    D1=np.zeros(NCoords).reshape(NAtoms,3)
    D1[:,0]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,0]
    D1=D1.flatten()
    D2=np.zeros(NCoords).reshape(NAtoms,3)
    D2[:,1]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,1]
    D2=D2.flatten()
    D3=np.zeros(NCoords).reshape(NAtoms,3)
    D3[:,2]=np.sqrt(atomicMasses.reshape(NAtoms,3))[:,2]
    D3=D3.flatten()
    # in a_0·sqrt(m_e)
    # Normalize the translational modes
    D1=D1/np.sqrt(np.sum(D1**2))
    D2=D2/np.sqrt(np.sum(D2**2))
    D3=D3/np.sqrt(np.sum(D3**2))

    coordinates=np.copy(currentCoordinates)
    # Check for the center of mass of the molecule
    centerOfMass=np.sum(atomicMasses.reshape(NAtoms,3)*coordinates.reshape(NAtoms,3),axis=0)/np.sum(atomicMasses.reshape(NAtoms,3))
    
    currentCoordinates=coordinates.reshape(NAtoms,3)
    masses=atomicMasses.reshape(NAtoms,3)[:,0]
    x=currentCoordinates[:,0]-centerOfMass[0]
    y=currentCoordinates[:,1]-centerOfMass[1]
    z=currentCoordinates[:,2]-centerOfMass[2]
    inertiaTensor=np.array(
                [
                    [
                        np.sum(masses*(y*y+z*z)),
                        -np.sum(masses*x*y),
                        -np.sum(masses*x*z)
                        ],
                    [
                        -np.sum(masses*y*x),
                        np.sum(masses*(x*x+z*z)),
                        -np.sum(masses*y*z)
                        ],
                    [
                        -np.sum(masses*z*x),
                        -np.sum(masses*z*y),
                        np.sum(masses*(x*x+y*y))
                        ]
                    ]
                    ) # a_0^2·m_e
    eigenvaluesInertiaTensor,diagonalizerInertiaTensor=scipy.linalg.eigh(inertiaTensor)
    # print(eigenvaluesInertiaTensor)# à comparer avec constantes rotationnelles
    principalMoments=eigenvaluesInertiaTensor
    currentCoordinates=np.dot(currentCoordinates,diagonalizerInertiaTensor)
        
    atomicMasses=atomicMasses.reshape(NAtoms,3)
    D4=[]
    D5=[]
    D6=[]
    for i in range(NAtoms):
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[0,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[0,1])*np.sqrt(atomicMasses[i,0]))
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[1,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[1,1])*np.sqrt(atomicMasses[i,0]))
        D4.append((currentCoordinates[i,1]*diagonalizerInertiaTensor[2,2]-currentCoordinates[i,2]*diagonalizerInertiaTensor[2,1])*np.sqrt(atomicMasses[i,0]))
        
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[0,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[0,2])*np.sqrt(atomicMasses[i,0]))
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[1,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[1,2])*np.sqrt(atomicMasses[i,0]))
        D5.append((currentCoordinates[i,2]*diagonalizerInertiaTensor[2,0]-currentCoordinates[i,0]*diagonalizerInertiaTensor[2,2])*np.sqrt(atomicMasses[i,0]))
        
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[0,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[0,0])*np.sqrt(atomicMasses[i,0]))
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[1,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[1,0])*np.sqrt(atomicMasses[i,0]))
        D6.append((currentCoordinates[i,0]*diagonalizerInertiaTensor[2,1]-currentCoordinates[i,1]*diagonalizerInertiaTensor[2,0])*np.sqrt(atomicMasses[i,0]))
    D4=np.array(D4)
    D5=np.array(D5)
    D6=np.array(D6)
    # Normalize the rotational modes
    D4=D4/np.sqrt(np.sum(D4**2))
    D5=D5/np.sqrt(np.sum(D5**2))
    D6=D6/np.sqrt(np.sum(D6**2))
    # Vérifier les constantes rotationnelles 

    D=np.array([D1,D2,D3,D4,D5,D6]).T
    U,s,V=scipy.linalg.svd(D,full_matrices=True)
    B=U[:,6:]
    if separate_TR:
        M=np.diag(1/np.sqrt(atomicMasses.flatten()))

        DTR=np.copy(D)
        D=np.copy(B)
        intHessianTR=np.dot(DTR.T,np.dot(hessianInMWC,DTR))
        eigenvalues,diagonalizer=scipy.linalg.eigh(intHessianTR)
        # diagonalizer is lINT, NM in cols expressed in the internal coordinates.
        # thus normal modes have to be expressed in the MWC system again to be visualized.
        # normalModesVibration=np.dot(D,diagonalizer).T
        normalModesTR=np.dot(diagonalizer.T,DTR.T)
        negativeFrequenciesTR=-CST.HARTREE_TO_RCM*np.sqrt(-eigenvalues[eigenvalues<0])
        positiveFrequenciesTR=CST.HARTREE_TO_RCM*np.sqrt(eigenvalues[eigenvalues>=0])
        frequenciesTR=np.append(negativeFrequenciesTR,positiveFrequenciesTR)

        lCartTR=np.dot(M,np.dot(DTR,diagonalizer)) # diagonalizer is lINT, NM in cols expressed in the internal coordinates
        renormalizationTR=np.sqrt(1/np.sum(lCartTR**2,axis=0)) # sqrt(me)
        reducedMassesTR=1/np.sum(lCartTR**2,axis=0)/CST.AMU_TO_ME # me -> amu
        printedlCartTR=renormalizationTR*lCartTR # Modes in the columns # was in sqrt(me)^{-1}, now in 1
        printedModesCartTR=printedlCartTR.T # Modes in the rows
        ###################################################
        # Using D, one can transform the mass-weighted Hessian to the Hessian expressed in the internal coordinates
        intHessian=np.dot(D.T,np.dot(hessianInMWC,D))
        eigenvalues,diagonalizer=scipy.linalg.eigh(intHessian)
        # diagonalizer is lINT, NM in cols expressed in the internal coordinates.
        # thus normal modes have to be expressed in the MWC system again to be visualized.
        # normalModesVibration=np.dot(D,diagonalizer).T
        normalModesVibration=np.dot(diagonalizer.T,D.T)
        negativeFrequencies=-CST.HARTREE_TO_RCM*np.sqrt(-eigenvalues[eigenvalues<0])
        positiveFrequencies=CST.HARTREE_TO_RCM*np.sqrt(eigenvalues[eigenvalues>=0])
        frequencies=np.append(negativeFrequencies,positiveFrequencies)

        lCart=np.dot(M,np.dot(D,diagonalizer)) # diagonalizer is lINT, NM in cols expressed in the internal coordinates
        renormalization=np.sqrt(1/np.sum(lCart**2,axis=0)) # sqrt(me)
        reducedMasses=1/np.sum(lCart**2,axis=0)/CST.AMU_TO_ME # me -> amu
        printedlCart=renormalization*lCart # Modes in the columns # was in sqrt(me)^{-1}, now in 1
        printedModesCart=printedlCart.T # Modes in the rows

        frequencies=np.concatenate((frequenciesTR,frequencies))
        reducedMasses=np.concatenate((reducedMassesTR,reducedMasses))
        ## print(frequencies.shape)
        ## print(printedlCartTR.shape)
        ## print(printedlCart.shape)
        # printedlCart=np.stack((printedlCartTR.T,printedlCart.T),axis=1).T
        # printedModesCart=np.stack((printedModesCartTR.T,printedModesCart.T),axis=1).T
        # printedlCart=np.column_stack((printedlCartTR,printedlCart))
        # printedModesCart=np.column_stack((printedModesCartTR,printedModesCart))
        printedlCart=np.block([printedlCartTR,printedlCart])
        printedModesCart=np.block([[printedModesCartTR],[printedModesCart]])
        ## print(printedlCartTR.shape)
        ## print(printedlCart.shape)
    else:
        D=np.append(D,B,axis=1)
        # D is the total transformation matrix from mass-weighted coordinates to internal coordinates.
        # Note that the first 6 columns of D are the vectors relative to translations and rotations.

        # Using D, one can transform the mass-weighted Hessian to the Hessian expressed in the internal coordinates
        intHessian=np.dot(D.T,np.dot(hessianInMWC,D))
        eigenvalues,diagonalizer=scipy.linalg.eigh(intHessian)
        # diagonalizer is lINT, NM in cols expressed in the internal coordinates.
        # thus normal modes have to be expressed in the MWC system again to be visualized.
        # normalModesVibration=np.dot(D,diagonalizer).T
        normalModesVibration=np.dot(diagonalizer.T,D.T)
        negativeFrequencies=-CST.HARTREE_TO_RCM*np.sqrt(-eigenvalues[eigenvalues<0])
        positiveFrequencies=CST.HARTREE_TO_RCM*np.sqrt(eigenvalues[eigenvalues>=0])
        frequencies=np.append(negativeFrequencies,positiveFrequencies)

        M=np.diag(1/np.sqrt(atomicMasses.flatten()))

        lCart=np.dot(M,np.dot(D,diagonalizer)) # diagonalizer is lINT, NM in cols expressed in the internal coordinates
        renormalization=np.sqrt(1/np.sum(lCart**2,axis=0)) # sqrt(me)
        reducedMasses=1/np.sum(lCart**2,axis=0)/CST.AMU_TO_ME # me -> amu
        # print("Renormalization (a.u.):\n",renormalization)
        # print("reducedMasses (AMU):\n",reducedMasses)
        printedlCart=renormalization*lCart # Modes in the columns # was in sqrt(me)^{-1}, now in 1
        printedModesCart=printedlCart.T # Modes in the rows
    return(frequencies,reducedMasses,printedModesCart,atomicNumbers,atomicMasses,coordinates)
   
def orthogonalizeNormalModes(printedModesCart,reducedMasses,atomicMasses,check=True):
    normalModes=np.copy(printedModesCart)
    for i in range(len(reducedMasses)):
        for j in range(len(atomicMasses.flatten())):
            normalModes[i,j]=printedModesCart[i,j]*(np.sqrt(atomicMasses.flatten()[j])/np.sqrt(reducedMasses[i]*CST.AMU_TO_ME))
    if check:
        fig,ax1=plotGramMatrix(printedModesCart,nrows=2,ncols=1,index=1)
        fig,ax2=plotGramMatrix(normalModes,fig=fig,nrows=2,ncols=1,index=2)
        plt.show()
    return normalModes

def visualizeDisplacement(initialCoordinates,displacement,atomicNumbers=0):
    NCoords=len(initialCoordinates.flatten())
    NAtoms=int(NCoords/3)
    initialCoordinates=initialCoordinates.reshape(NAtoms,3)
    maxCoordinates=np.max(initialCoordinates)
    displacement=displacement.reshape(NAtoms,3)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    if type(atomicNumbers) is float:
        ax.scatter(initialCoordinates[:,0],initialCoordinates[:,1],initialCoordinates[:,2])
        ax.scatter(initialCoordinates[:,0]+displacement[:,0],initialCoordinates[:,1]+displacement[:,1],initialCoordinates[:,2]+displacement[:,2])
    else:
        atomicNumbers=atomicNumbers.astype(str)
        atomicNumbersColors={"1":"grey","6":"black"}
        atomicNumbersColorsDisp={"1":"blue","6":"red"}
        for atomicNumber in set(atomicNumbers):
            where=(atomicNumbers==atomicNumber)
            ax.scatter(initialCoordinates[where,0],initialCoordinates[where,1],initialCoordinates[where,2],color=atomicNumbersColors[atomicNumber])
            ax.scatter(initialCoordinates[where,0]+displacement[where,0],initialCoordinates[where,1]+displacement[where,1],initialCoordinates[where,2]+displacement[where,2],color=atomicNumbersColorsDisp[atomicNumber],alpha=0.3)
    ax.set_xlim(-maxCoordinates,maxCoordinates)
    ax.set_ylim(-maxCoordinates,maxCoordinates)
    ax.set_zlim(-maxCoordinates,maxCoordinates)
    # plt.show()
    return fig

def visualizeDisplacementNospec(initialCoordinates,displacement,atomicNumbers=0):
    NCoords=len(initialCoordinates.flatten())
    NAtoms=int(NCoords/3)
    initialCoordinates=initialCoordinates.reshape(NAtoms,3)
    maxCoordinates=np.max(initialCoordinates)
    displacement=displacement.reshape(NAtoms,3)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    if type(atomicNumbers) is float:
        ax.scatter(initialCoordinates[:,0],initialCoordinates[:,1],initialCoordinates[:,2])
        ax.scatter(initialCoordinates[:,0]+displacement[:,0],initialCoordinates[:,1]+displacement[:,1],initialCoordinates[:,2]+displacement[:,2])
    else:
        atomicNumbers=atomicNumbers.astype(str)
        atomicNumbersColors={"1":"grey","6":"black"}
        atomicNumbersColorsDisp={"1":"blue","6":"red"}
        for atomicNumber in set(atomicNumbers):
            where=(atomicNumbers==atomicNumber)
            ax.scatter(initialCoordinates[where,0],initialCoordinates[where,1],initialCoordinates[where,2],color=atomicNumbersColors[atomicNumber])
            ax.scatter(initialCoordinates[where,0]+displacement[where,0],initialCoordinates[where,1]+displacement[where,1],initialCoordinates[where,2]+displacement[where,2],color=atomicNumbersColorsDisp[atomicNumber],alpha=0.3)
    ax.set_xlim(-maxCoordinates,maxCoordinates)
    ax.set_ylim(-maxCoordinates,maxCoordinates)
    ax.set_zlim(-maxCoordinates,maxCoordinates)
    # plt.show()
    return fig,ax


def visualizeNormalMode(initialCoordinates,normalMode,atomicNumbers=None,scale=1,projection="3d"):
    NAtoms=len(initialCoordinates.flatten())//3
    atom_colors={}
    atom_colors["1"]="gray"
    atom_colors["H"]="gray"
    atom_colors["6"]="black"
    atom_colors["C"]="black"
    atom_colors["8"]="red"
    atom_colors["O"]="red"
    initialCoordinates=initialCoordinates.reshape(NAtoms,3)
    maxCoordinates=np.max(initialCoordinates)
    normalMode=normalMode.reshape(NAtoms,3)
    fig=plt.figure()
    if projection=="3d":
        ax=fig.add_subplot(111,projection=projection)
    else:
        ax=fig.add_subplot(111)
        xdisp=np.sum(initialCoordinates[:,0]**2)
        ydisp=np.sum(initialCoordinates[:,1]**2)
        zdisp=np.sum(initialCoordinates[:,2]**2)
        if xdisp < min(ydisp,zdisp):
            a=1 # xfig = ymol
            b=2 # yfig = zmol
        if ydisp < min(xdisp,zdisp):
            a=0 # xfig = xmol
            b=2 # yfig = zmol
        if zdisp < min(xdisp,ydisp):
            a=0 # xfig = xmol
            b=1 # yfig = ymol
    if atomicNumbers is None:
        if projection=="3d":
            ax.scatter(initialCoordinates[:,0],initialCoordinates[:,1],initialCoordinates[:,2])
        else:
            ax.scatter(initialCoordinates[:,a],initialCoordinates[:,b])
    else:
        for atomType in set(atomicNumbers):
            atomSelection=(atomicNumbers==atomType)
            if projection=="3d":
                ax.scatter(initialCoordinates[atomSelection,0],initialCoordinates[atomSelection,1],initialCoordinates[atomSelection,2],color=atom_colors[str(atomType)])
            else:
                ax.scatter(initialCoordinates[atomSelection,a],initialCoordinates[atomSelection,b],color=atom_colors[str(atomType)])
    
    normalModeVectors=normalMode/np.max(np.sqrt(np.sum(normalMode**2,axis=1)))
    normalModeVectors*=scale
    if projection=="3d":
        ax.quiver(initialCoordinates[:,0],initialCoordinates[:,1],initialCoordinates[:,2],
                  normalModeVectors[:,0],normalModeVectors[:,1],normalModeVectors[:,2],
                 )
    else:
        ax.quiver(initialCoordinates[:,a],initialCoordinates[:,b],
                  normalModeVectors[:,a],normalModeVectors[:,b],
                  color="tab:blue",scale=1,scale_units="xy"
                  )
    ax.set_xlim(-1.1*maxCoordinates,1.1*maxCoordinates)
    ax.set_ylim(-1.1*maxCoordinates,1.1*maxCoordinates)
    if projection=="3d":
        ax.set_zlim(-1.1*maxCoordinates,1.1*maxCoordinates)
    return fig,ax

def plotMatrix(matrix,fig=None,nrows=1,ncols=1,index=1,cmap="cividis"):
    if fig is None:
        fig=plt.figure()
    ax=fig.add_subplot(nrows,ncols,index)
    ap=ax.imshow(matrix,cmap="cividis")
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right",
                                        size="5%",
                                        pad=0.1)
    # Using new axes for colorbar
    plt.colorbar(ap, cax=colorbar_axes)
    return fig,ax

def plotGramMatrix(vectors,fig=None,nrows=1,ncols=1,index=1):
    if fig is None:
        fig=plt.figure()
    ax=fig.add_subplot(nrows,ncols,index)
    ap=ax.imshow(np.dot(vectors,vectors.T))
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right",
                                        size="5%",
                                        pad=0.1)
    # Using new axes for colorbar
    plt.colorbar(ap, cax=colorbar_axes)
    return fig,ax

def numericalBranchingSpace(filetag,rootA="A",rootB="B",roottag="AB",save="y",save_asNM="y",check=True,highest_negative=True):
    with open(filetag+"_NBS.dat","w") as ResultsFile:
        ResultsFile.write("Numerical Branching Space information for "+filetag+" calculations")
        ResultsFile.write("\n")
        ResultsFile.write("\n")

    if roottag=="num":
        filenameA=filetag+"_"+str(rootA)+".fchk"
        filenameB=filetag+"_"+str(rootB)+".fchk"
    elif roottag=="AB":
        filenameA=filetag+"_A.fchk"
        filenameB=filetag+"_B.fchk"
    currentEnergySA,currentGradientSA,currentHessianSA=fchk2derivatives(filenameA,mw=True,freq=True)[:3]
    currentEnergySB,currentGradientSB,currentHessianSB=fchk2derivatives(filenameB,mw=True,freq=True)[:3]
    currentGradientSA=currentGradientSA.flatten() # Hartree / (Bohr·me^1/2)
    currentGradientSB=currentGradientSB.flatten() # Hartree / (Bohr·me^1/2)

    frequencies,reducedMasses,printedModesCart,atomicNumbers,atomicMasses,coordinates=fchk2vibrationalAnalysis(filenameA)
    atomicMasses=fchk2derivatives(filenameA)[3]
    currentCoordinates=coordinates.flatten() # Angstrom
    NAtoms=coordinates.size//3
    NCoords=3*NAtoms
    NModes=3*NAtoms-6

    # currentEnergyDifference=currentEnergySB-currentEnergySA
    # currentGradientDifference=currentGradientSB-currentGradientSA
    # currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))

    # currentHessianDifference=currentHessianSB-currentHessianSA
    currentEnergyDifference=0.5*(currentEnergySB-currentEnergySA)
    currentGradientDifference=0.5*(currentGradientSB-currentGradientSA)
    currentGradientDifferenceNorm=np.sqrt(np.sum(currentGradientDifference**2))

    currentHessianDifference=0.5*(currentHessianSB-currentHessianSA)

    # currentGradientDifferenceProjector=np.tensordot(0.5*currentGradientDifference,0.5*currentGradientDifference,axes=0)
    # currentSquaredHessianDifference=2*(0.5*currentEnergyDifference)*(0.5*currentHessianDifference)+2*currentGradientDifferenceProjector
    currentGradientDifferenceProjector=np.tensordot(currentGradientDifference,currentGradientDifference,axes=0)
    currentSquaredHessianDifference=2*(currentEnergyDifference)*(currentHessianDifference)+2*currentGradientDifferenceProjector
    eigval,diagonalizer=linalg.eigh(currentSquaredHessianDifference)
    eigvec=diagonalizer.T
    BSVLengths,BSVVectors=eigval[-2:][::-1],eigvec[-2:][::-1]

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Eigenvalues of the Hessian of the squared energy difference:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        for i in range(len(eigval)):
            ResultsFile.write(str(i+1)+"\t"+str(eigval[i]))
            ResultsFile.write("\n")
        ResultsFile.write("\n")

        ResultsFile.write("NAtoms, NCoords: {}, {}\n".format(NAtoms,NCoords))
        ResultsFile.write("Branching-space vectors shape: \n".format(BSVVectors.shape))
        ResultsFile.write("Associated Lengths: \n".format(BSVLengths))
        ResultsFile.write("u1·u1 = {}\n".format(np.dot(BSVVectors[0],BSVVectors[0]))) # normalized
        ResultsFile.write("u2·u2 = {}\n".format(np.dot(BSVVectors[1],BSVVectors[1]))) # normalized
        ResultsFile.write("u1·u2 = {}\n".format(np.dot(BSVVectors[0],BSVVectors[1]))) # and orthogonal

    currentHighestNegative=eigvec[0].flatten()
    currentBranchingSpaceVector1=eigvec[-1].flatten()
    currentBranchingSpaceVector2=eigvec[-2].flatten()

    current_coordinates=currentCoordinates.reshape(NAtoms,3)
    current_HighestNegative=currentHighestNegative.reshape(NAtoms,3) 
    current_BranchingSpaceVector1=currentBranchingSpaceVector1.reshape(NAtoms,3) 
    current_BranchingSpaceVector2=currentBranchingSpaceVector2.reshape(NAtoms,3) 

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Length (E_h^2/(a_0^2·m_e)) and components of the Unitary BS vector (as NM) associated to the highest non-zero eigenvalue:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        ResultsFile.write("\t"+str(eigval[-1]))
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        # ResultsFile.write("Num. Atom.\t x \t y \t z \t (in Hartree/(Angstrom * sqrt(me))")
        ResultsFile.write("Num. Atom.\t x \t y \t z")
        ResultsFile.write("\n")
        for i in range(len(current_BranchingSpaceVector1)):
            ResultsFile.write(str(i+1)+"\t\t"+str(current_BranchingSpaceVector1[i][0])+"\t\t"+str(current_BranchingSpaceVector1[i][1])+"\t\t"+str(current_BranchingSpaceVector1[i][2])+"\n")
        ResultsFile.write("\n")
        ResultsFile.write("Length (E_h^2/(a_0^2·m_e)) and components of the Unitary BS vector (as NM) associated to the lowest non-zero eigenvalue:")
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        ResultsFile.write("\t"+str(eigval[-2]))
        ResultsFile.write("\n")
        ResultsFile.write("\n")
        # ResultsFile.write("Num. Atom.\t x \t y \t z \t (in Hartree/(Angstrom * sqrt(me))")
        ResultsFile.write("Num. Atom.\t x \t y \t z")
        ResultsFile.write("\n")
        for i in range(len(current_BranchingSpaceVector2)):
            ResultsFile.write(str(i+1)+"\t\t"+str(current_BranchingSpaceVector2[i][0])+"\t\t"+str(current_BranchingSpaceVector2[i][1])+"\t\t"+str(current_BranchingSpaceVector2[i][2])+"\n")


    inverseSquareRootMasses=np.diag(1/np.sqrt(atomicMasses.flatten()*CST.AMU_TO_ME)) 
    BSVVectorsCart=np.dot(inverseSquareRootMasses,BSVVectors.T) # vectors(mw coords) --> vectors(cart coords)
    normalizationFactors=np.sqrt(1/np.sum(BSVVectorsCart**2,axis=0)) # sqrt(m_e)
    BSVReducedMasses=1/np.sum(BSVVectorsCart**2,axis=0)/CST.AMU_TO_ME # m_e --> AMU
    BSVVectorsCart=normalizationFactors*BSVVectorsCart # not normalized --> normalized 
    BSVVectorsCart=BSVVectorsCart.T

    with open(filetag+"_NBS.dat","a+") as ResultsFile:
        ResultsFile.write("Normalization Factor (a.u.):\n {} \n".format(normalizationFactors))
        ResultsFile.write("reducedMasses (AMU):\n {} \n".format(BSVReducedMasses))
        ResultsFile.write("ucart1·ucart1 = {}\n".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[0]))) # normalized
        ResultsFile.write("ucart2·ucart2 = {}\n".format(np.dot(BSVVectorsCart[1],BSVVectorsCart[1]))) # normalized
        ResultsFile.write("ucart1·ucart2 = {}\n".format(np.dot(BSVVectorsCart[0],BSVVectorsCart[1]))) # but not rigorously orthogonal


    if check:
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
        ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector1[:,0],current_coordinates[:,1]+current_BranchingSpaceVector1[:,1],current_coordinates[:,2]+current_BranchingSpaceVector1[:,2])
        max_coordinates=np.max(np.abs(current_coordinates))
        # ax.set_title("NBS 1 with ||NBS₁||="+str(eigval[-1]))
        ax.set_xlim(-max_coordinates,max_coordinates)
        ax.set_ylim(-max_coordinates,max_coordinates)
        ax.set_zlim(-max_coordinates,max_coordinates)
        if save=="y" or save=="yes":
            plt.savefig(filetag+"_NBS1.png")
            plt.savefig(filetag+"_NBS1.pdf")
        plt.show()
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
        ax.scatter(current_coordinates[:,0]+current_BranchingSpaceVector2[:,0],current_coordinates[:,1]+current_BranchingSpaceVector2[:,1],current_coordinates[:,2]+current_BranchingSpaceVector2[:,2])
        # ax.set_title("NBS 2 with ||NBS₂||="+str(eigval[-2]))
        ax.set_xlim(-max_coordinates,max_coordinates)
        ax.set_ylim(-max_coordinates,max_coordinates)
        ax.set_zlim(-max_coordinates,max_coordinates)
        if save=="y" or save=="yes":
            plt.savefig(filetag+"_NBS2.png")
            plt.savefig(filetag+"_NBS2.pdf")
        plt.show()
    if highest_negative=="y" or highest_negative=="yes":
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        ax.scatter(current_coordinates[:,0],current_coordinates[:,1],current_coordinates[:,2])
        ax.scatter(current_coordinates[:,0]+current_HighestNegative[:,0],current_coordinates[:,1]+current_HighestNegative[:,1],current_coordinates[:,2]+current_HighestNegative[:,2])
        # ax.set_title("eigvec 0 with eigval="+str(eigval[0]))
        ax.set_xlim(-max_coordinates,max_coordinates)
        ax.set_ylim(-max_coordinates,max_coordinates)
        ax.set_zlim(-max_coordinates,max_coordinates)
        plt.show()

    # print("Save as NM:",save_asNM)
    if save_asNM=="y" or save_asNM=="yes":
        grid=False
        modeSelection=[0,1]
        for mode_index in modeSelection:
            fig,ax=visualizeNormalMode(coordinates.flatten(),BSVVectorsCart[mode_index],atomicNumbers=atomicNumbers,projection="2d")
            ax.set_aspect("equal")
            # fig.suptitle("{} mode {}".format(filetag,mode))
            if grid:
                plt.grid()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                plt.axis("off")
            plt.tight_layout()
            pdfname="{}_NBSCart_x{}.pdf".format(filetag,mode_index+1)
            plt.savefig(pdfname)
            plt.show()

    return BSVLengths,BSVVectors

def iterativeProcrustes(initialCoordinates,finalCoordinates,selectedFinalCoordinates,
                        check_initial=True,check_each_step=False,rotations=True,start_with="rotation"):
    """
    Uses functions of the Procrustes Library to find a sequence of rotation and permutation matrices
    to be applied to a set of initialCoordinates so that they align with a set of selectedFinalCoordinates
    Inputs:
    - initialCoordinates:          2D-array of shape (n,3)
                                   initial set of coordinates for the smaller molecular fragment geometry
    - finalCoordinates:            2D-array of shape (m,3), m>n
                                   set of coordinates for the total molecular geometry
    - selectedFinalCoordinates:    2D-array of shape (n,3)
                                   selected set of coordinates for the molecular fragment geometry inside
                                   the total molecular geometry
    - check_initial:               bool
                                   if True, plot initialCoordinates and selectedFinalCoordinates
    - check_each_step:             bool
                                   if True, at each step, plot rotatedInitialCoordinates(step_number) and
                                   selectedFinalCoordinates
    Returns:
    - rotation_matrices_right:     (NSteps)-list of rotation matrices (3,3)
    - permutation_matrices_right:  (NSteps)-list of permutation matrices (n,n)
    - translation_vectors[0]:      first translation vector (3) [identical at each step]
    """
    rmsd_before=np.sqrt(np.mean(np.sum((initialCoordinates-selectedFinalCoordinates)**2, axis=1)))
    if check_initial:
        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1, projection='3d')
        title="Original molecules: RMSD={:0.2f} (\AA)".format(rmsd_before)
        ax.scatter(xs=initialCoordinates[:, 0], ys=initialCoordinates[:, 1], zs=initialCoordinates[:, 2],
                   marker="o", color="blue", s=40, label="Initial Fragment")
        ax.scatter(xs=selectedFinalCoordinates[:, 0], ys=selectedFinalCoordinates[:, 1], zs=selectedFinalCoordinates[:, 2],
                   marker="o", color="red", s=40, label="Selected Fragment")
        ax.set_title(title)
        ax.set_xlim(-12,12)
        ax.set_ylim(-12,12)
        ax.set_zlim(-12,12)
        ax.set_xlabel("X (\AA)")
        ax.set_ylabel("Y (\AA)")
        ax.set_zlabel("Z (\AA)")
        ax.legend(loc="best")
        plt.show()

    c=0
    rotation_matrices_right=[]
    translation_vectors=[]
    permutation_matrices_right=[]
    converged=False
    while not converged:
        # Rotation
        if rotations and start_with=="rotation":
            result=procrustes.orthogonal(initialCoordinates,selectedFinalCoordinates,
                                         # pad=True,unpad_col=True,unpad_row=True,
                                         pad=True,unpad_col=False,unpad_row=False,
                                         translate=True)
            rotation_matrices_right.append(result.t)
            initialCoordinates=np.dot(result.new_a,result.t)
            # print(result.new_b[0])
            # print(result.new_b.shape)
            # print(selectedFinalCoordinates[0])
            # print(selectedFinalCoordinates.shape)
            translation_vector=result.new_b[0]-selectedFinalCoordinates[0]
            translation_vectors.append(translation_vector)
            selectedFinalCoordinates=result.new_b
            # Translation
            initialCoordinates-=translation_vector
            selectedFinalCoordinates-=translation_vector
            # Permutation
            result=procrustes.permutation(initialCoordinates.T,selectedFinalCoordinates.T,
                                          pad=True,unpad_col=True,unpad_row=True,
                                          translate=False)
            permutation_matrices_right.append(result.t)
            initialCoordinates=np.dot(result.new_a[:3],result.t).T
            selectedFinalCoordinates=result.new_b[:3].T
        elif rotations and start_with=="permutation":
            # Permutation
            result=procrustes.permutation(initialCoordinates.T,selectedFinalCoordinates.T,
                                          pad=True,unpad_col=True,unpad_row=True,
                                          translate=False)
            permutation_matrices_right.append(result.t)
            initialCoordinates=np.dot(result.new_a[:3],result.t).T
            selectedFinalCoordinates=result.new_b[:3].T

            result=procrustes.orthogonal(initialCoordinates,selectedFinalCoordinates,
                                         # pad=True,unpad_col=True,unpad_row=True,
                                         pad=True,unpad_col=False,unpad_row=False,
                                         translate=True)
            rotation_matrices_right.append(result.t)
            initialCoordinates=np.dot(result.new_a,result.t)
            # print(result.new_b[0])
            # print(result.new_b.shape)
            # print(selectedFinalCoordinates[0])
            # print(selectedFinalCoordinates.shape)
            translation_vector=result.new_b[0]-selectedFinalCoordinates[0]
            translation_vectors.append(translation_vector)
            selectedFinalCoordinates=result.new_b
            # Translation
            initialCoordinates-=translation_vector
            selectedFinalCoordinates-=translation_vector
        elif not rotations:
            # Permutation
            result=procrustes.permutation(initialCoordinates.T,selectedFinalCoordinates.T,
                                          pad=True,unpad_col=True,unpad_row=True,
                                          translate=False)
            permutation_matrices_right.append(result.t)
            initialCoordinates=np.dot(result.new_a[:3],result.t).T
            selectedFinalCoordinates=result.new_b[:3].T
            rotation_matrices_right.append(np.diag(np.ones(3)))
            translation_vectors.append(np.zeros(3))

        rmsd_after=np.sqrt(np.mean(np.sum((initialCoordinates-selectedFinalCoordinates)**2, axis=1)))
        print("RMSD(step={:0.2f}) = {}".format(c,rmsd_after))
        if check_each_step:
            fig=plt.figure()
            ax=fig.add_subplot(1, 1, 1, projection='3d')
            title="Rotated and Permuted (step={}) molecules: RMSD={:0.2f} (\AA)".format(c,rmsd_after)
            ax.scatter(xs=initialCoordinates[:, 0], ys=initialCoordinates[:, 1], zs=initialCoordinates[:, 2],
                       marker="o", color="blue", s=40, label="Rotated Fragment")
            ax.scatter(xs=selectedFinalCoordinates[:, 0], ys=selectedFinalCoordinates[:, 1], zs=selectedFinalCoordinates[:, 2],
                       marker="o", color="red", s=40, label="Selected Fragment")
            ax.set_title(title)
            ax.set_xlim(-12,12)
            ax.set_ylim(-12,12)
            ax.set_zlim(-12,12)
            ax.set_xlabel("X (\AA)")
            ax.set_ylabel("Y (\AA)")
            ax.set_zlabel("Z (\AA)")
            ax.legend(loc="best")
            plt.show()
        c+=1
        #print(translation_vector)
        if np.round(rmsd_after,2)==np.round(rmsd_before,2):
            converged=True
        rmsd_before=rmsd_after
    return rotation_matrices_right,permutation_matrices_right,translation_vectors[0]

def transformCoordinates(initialCoordinates,rotation_matrices_right,permutation_matrices_right,translation_vector):
    rotatedInitialCoordinates=np.copy(initialCoordinates)
    NSteps=len(rotation_matrices_right)
    for i in range(NSteps):
        rotatedInitialCoordinates=np.dot(rotatedInitialCoordinates,rotation_matrices_right[i])
        rotatedInitialCoordinates=np.dot(rotatedInitialCoordinates.T,permutation_matrices_right[i]).T
    rotatedInitialCoordinates-=translation_vector
    return rotatedInitialCoordinates

def transformNormalModes(initialNormalModes,rotation_matrices_right,permutation_matrices_right,translation_vector):
    rotatedInitialNormalModes=np.copy(initialNormalModes)
    NModes=len(rotatedInitialNormalModes)
    NCoords=rotatedInitialNormalModes[0].size
    rotatedInitialNormalModes=rotatedInitialNormalModes.reshape(NModes,NCoords//3,3)
    NSteps=len(rotation_matrices_right)
    for n in range(NModes):
        for i in range(NSteps):
            rotatedInitialNormalModes[n]=np.dot(rotatedInitialNormalModes[n],rotation_matrices_right[i])
            rotatedInitialNormalModes[n]=np.dot(rotatedInitialNormalModes[n].T,permutation_matrices_right[i]).T
        #rotatedInitialNormalModes[n]-=translation_vectors[0]
    return rotatedInitialNormalModes

def fchk2shift(filenameRef,filenameCurrent,normalModesRef,atomic_masses,auto_align=False,show_all=False):
    NAtoms,NCoords,atomicNumbersRef,coordinatesRef=fchk2coordinates(filenameRef)
    coordinatesRef=coordinatesRef.reshape(NAtoms,3)
    NAtoms,NCoords,atomicNumbersCurrent,coordinatesCurrent=fchk2coordinates(filenameCurrent)
    coordinatesCurrent=coordinatesCurrent.reshape(NAtoms,3)
    if auto_align:
        rotation_matrices_right,permutation_matrices_right,translation_vector=iterativeProcrustes(coordinatesCurrent,
                                                                          coordinatesRef,
                                                                          coordinatesRef,
                                                                          check_initial=show_all,
                                                                          check_each_step=show_all)
        rotatedCoordinatesCurrent=transformCoordinates(coordinatesCurrent,
                              rotation_matrices_right,
                              permutation_matrices_right,
                              translation_vector)
        # check if they were any atom permutations
        for _,matrix in enumerate(permutation_matrices_right):
            print("Permutation matrix of step {} is identity: {}".format(_,np.allclose(matrix,np.eye(len(matrix)))))
            if not np.allclose(matrix,np.eye(len(matrix))):
                print("WARNING, PERMUTATION HAPPENED")
        # check type of rotations
        for _,matrix in enumerate(rotation_matrices_right):
            print("Rotation matrix of step {} is\n {}".format(_,matrix))
        # if all True, we only rotated the "initial" fragment to superimpose it best to the "final" fragment

        # AFTER #
        if show_all:
            fig=plt.figure()
            ax=fig.add_subplot(1, 1, 1, projection='3d')
            rmsd_after=np.sqrt(np.mean(np.sum((rotatedCoordinatesCurrent-coordinatesRef)**2, axis=1)))
            title="Rotated and Permuted molecules: RMSD={:0.2f} (\AA)".format(rmsd_after)
            ax.scatter(xs=rotatedCoordinatesCurrent[:, 0], ys=rotatedCoordinatesCurrent[:, 1], zs=rotatedCoordinatesCurrent[:, 2],
                       marker="o", color="blue", s=40, label="Rotated Fragment")
            ax.scatter(xs=coordinatesRef[:, 0], ys=coordinatesRef[:, 1], zs=coordinatesRef[:, 2],
                       marker="o", color="red", s=40, label="Selected Fragment")
            ax.set_title(title)
            ax.set_xlim(-12,12)
            ax.set_ylim(-12,12)
            ax.set_zlim(-12,12)
            ax.set_xlabel("X (\AA)")
            ax.set_ylabel("Y (\AA)")
            ax.set_zlabel("Z (\AA)")
            ax.legend(loc="best")
            plt.show()

        initialCoordinates=np.copy(coordinatesRef)
        finalCoordinates=np.copy(rotatedCoordinatesCurrent)
    else:
        initialCoordinates=np.copy(coordinatesRef)
        finalCoordinates=np.copy(coordinatesCurrent)
    initialCoordinates=initialCoordinates.flatten()
    finalCoordinates=finalCoordinates.flatten()
    NAtoms=int(finalCoordinates.size//3)
    atomic_masses=atomic_masses.flatten()
    deltaCoordinates=finalCoordinates-initialCoordinates
    deltaCoordinates=deltaCoordinates.flatten()
    for i in range(len(atomic_masses)):
        deltaCoordinates[i]=deltaCoordinates[i]*np.sqrt(atomic_masses[i])/CST.BOHR_TO_ANGSTROM
    shifts=[]
    for i in range(len(normalModesRef)):
        shifts.append(np.dot(normalModesRef[i],deltaCoordinates))
    shifts=np.array(shifts)
    return shifts
    
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=
    """
    Initial toolbox for interfacing Gaussian outputs (.log and .fchk) with homemade (numerical) optimization/analysis programs of Conical Intersections
    """
    )
    args=vars(parser.parse_args())

