#!/usr/bin/env /usr/local/anaconda3/bin/python3
###!/usr/bin/env /usr/local/bin/python3
#
#Import modules
import sys
import json
import numpy as np
from scipy.optimize import brentq
from argparse import ArgumentParser

precision = np.float64(1.0e-8);
maxloop   = np.int64(1000);

def func1(rl,L,dxs,n):
    return np.float64(L)*(1.0-np.float64(rl))-np.float64(dxs)*(1.0-np.float64(rl)**n)

def func2(rl,L,dxe,n):
    return np.float64(L)*(np.float64(rl)**(n-1))*(1.0-np.float64(rl))-np.float64(dxe)*(1.0-np.float64(rl)**n)

def globalFromLocal(rl,n):
    return np.float64(np.power(rl,(n-1)));

def localFromGlobal(rg,n):
    return np.float64(np.power(rg,(1.0/np.float64(n-1))));

def getDeltaStart(L,rl,n):
    return np.float64(L*(1.0-rl)/(1.0 - np.power(rl,n)));

def getDeltaEnd(L,rl,n):
    deltaStart = getDeltaStart(L,rl,n);
    rg = globalFromLocal(rl,n);
    return np.float64(rg*deltaStart);

def findSeedPoints2(func,L,dx,n,guessSeed,verbose):
    if verbose: print(" . "*25)
    if verbose: print("Computing seed points for use with brent method.")
    notFound = True;
    xn = np.float64(guessSeed);
    fxn  = np.float64(func(xn,L,dx,n));
    if verbose: print("Starting with xn = ",xn,", f(x) = ",fxn);
    fx0  = fxn;
    xnp1 = xn + np.float64(0.001)*xn;
    xnm1 = xn - np.float64(0.001)*xn;
    fxnp1= np.float64(func(xnp1,L,dx,n));
    fxnm1= np.float64(func(xnm1,L,dx,n));
    df  = (fxnp1 - fxnm1 ) / ( xnp1 - xnm1); 
    counter = 0;
    if verbose: print("In xnp1 = ",xnp1,", f(x) = ",fxnp1);
    while ( notFound ):
        if (counter > maxloop ):
            raise RuntimeError('Max number of iterations reached. :: findSeedPoints2()');
        if (fx0*fxn < 0.0):
            notFound = False;
            if verbose: print("Valid pair seed point found");
            if verbose: print("Final second seed point = ", xn,", with f(x) = ",fxn)
            return xn;
        if (fx0*fxnp1 < 0.0):
            notFound = False;
            if verbose: print("Valid pair seed point found");
            if verbose: print("Final second seed point = ", xnp1,", with f(x) = ",fxnp1)
            return xnp1;
        elif (fx0*fxnm1 < 0.0):
            notFound = False;
            if verbose: print("Valid pair seed point found");
            if verbose: print("Final second seed point = ", xnm1,", with f(x) = ",fxnm1)
            return xnm1;
        else:
            if (fxn*df/np.abs(fxn*df) == -1.0):
                xn = xn + np.float64(0.01)*xn;
            elif (fxn*df/np.abs(fxn*df) == 1.0):
                xn = xn - np.float64(0.01)*xn;
            fxn = np.float64(func(xn,L,dx,n));
            xnp1 = xn + np.float64(0.001)*xn;
            fxnp1= np.float64(func(xnp1,L,dx,n));
            xnm1 = xn - np.float64(0.001)*xn;
            fxnm1= np.float64(func(xnm1,L,dx,n));
            df  = (fxnp1 - fxnm1 ) / ( xnp1 - xnm1); 
            if verbose: print("... xn = ",xn,", f(x) = ",fxn, ", df = ", df);

def findRatioDeltaEnd(L,dx,n,verbose):
    if verbose: print(" . "*25)
    if verbose: print("Computing rLocal using L = {0:.6f}; dxEnd={1:.6f}; n={2:d}".format(L,dx,n));
    testDelta = np.float64(L/n);
    steps = 0;
    if (testDelta > dx):
        seedA = 0.8;
    else:
        seedA = 1.2;
    seedB = findSeedPoints2(func2,L,dx,n,seedA,verbose);
    rLocal = np.NAN;
    globalRatio = np.NAN;
    while (steps < maxloop):
        steps+=1;
        if verbose:
            print("Checking consistency at findContractionRatio")
            print("f(a) = ",func2(seedA,L,dx,n));
            print("f(b) = ",func2(seedB,L,dx,n));
        rLocal=np.float64(brentq(func2,seedA,seedB,args=(L,dx,n)));  # type: ignore
        if (np.abs(rLocal-1.0)<1e-8):
            print("+-"*36);
            print("Error finding rLocal. Trying another pair of seed points.");
            print("+-"*36);
            if (seedA > 1.0):
                seedA*=1.10;
            else:
                seedA*=0.90;
            seedB = findSeedPoints2(func2,L,dx,n,seedA,verbose);
            continue;
        else:
            globalRatio = np.float64(rLocal**(n-1));
            break;
    if (steps > maxloop ):
            raise RuntimeError('Max number of iterations reached. :: findRatioDeltaEnd()');
    return rLocal, globalRatio;

def findRatioDeltaStart(L,dx,n,verbose):
    if verbose: print(" . "*25)
    if verbose: print("Computing rLocal using L = {0:.6f}; dxStart={1:.6f}; n={2:d}".format(L,dx,n));
    testDelta = np.float64(L/n);
    steps = 0; 
    if (testDelta > dx):
        seedA = 1.2;
    else:
        seedA = 0.8;
    seedB = findSeedPoints2(func1,L,dx,n,seedA,verbose);
    rLocal = np.NAN;
    globalRatio = np.NAN;
    while (steps < maxloop):
        steps+=1;
        if verbose:
            print("Checking consistency at findContractionRatio")
            print("f(a) = ",func1(seedA,L,dx,n));
            print("f(b) = ",func1(seedB,L,dx,n));
        rLocal=np.float64(brentq(func1,seedA,seedB,args=(L,dx,n)));  # type: ignore
        if (np.abs(rLocal-1.0)<1e-8):
            print("+-"*36);
            print("Error finding rLocal. Trying another pair of seed points.");
            print("+-"*36);
            if (seedA > 1.0):
                seedA*=1.10;
            else:
                seedA*=0.90;
            seedB = findSeedPoints2(func1,L,dx,n,seedA,verbose);
            continue;
        else:
            globalRatio = np.float64(rLocal**(n-1));
            break;
    if (steps > maxloop ):
            raise RuntimeError('Max number of iterations reached. :: findRatioDeltaStart()');
    return rLocal, globalRatio;

def main():
    parser = ArgumentParser(
            description='Produces grading factors for use with multiblock blockMesh.', 
            allow_abbrev=False)
    parser.add_argument("-f", "--config-file", dest='configFile', action='store', type=str, 
            help="File to define the parameters to be used for generating the grading factors", 
            metavar="CONFIG_FILE")
    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', 
            help="Get more info about internal numerical process")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args();
    configFile = open(args.configFile);
    configData = json.loads(configFile.read());
    configFile.close();

    deltaStartProvided = "deltaStart" in configData;
    if (deltaStartProvided): deltaStart = np.float64(configData["deltaStart"]);
    deltaEndProvided = "deltaEnd" in configData;
    if (deltaEndProvided): deltaEnd = np.float64(configData["deltaEnd"]);
    totalLProvided   = "totalL" in configData;
    if (totalLProvided): L = np.float64(configData["totalL"]);
    posMinProvided   = "posMin" in configData;
    if (posMinProvided): posMin = np.float64(configData["posMin"]);
    posMaxProvided   = "posMax" in configData;
    if (posMaxProvided): posMax = np.float64(configData["posMax"]);
    nCellsProvided   = "nCells" in configData;
    if (nCellsProvided): nCells = np.int64(configData["nCells"]);
    globlRatProvided = "globalRatio" in configData;
    if (globlRatProvided): globalRatio = np.float64(configData["globalRatio"]);

    if (globlRatProvided and nCellsProvided):
        localRatio = localFromGlobal(globalRatio,nCells);
        if (totalLProvided):
            print("\n");
            print("|","="*70,"|");
            print("|"," "*4,"Results for given length, global ratio, and number of cells."," "*4,"  |")
            print(" "*5,"Total length: \t\t{0:.8f} <<<".format(L));
            print(" "*5,"Global ratio: \t\t{0:.8f} <<<".format(globalRatio));
            print(" "*5,"Number of cells:\t\t{0:d} <<<".format(nCells));
            print(" "*5,"Local ratio:  \t\t{0:.8f} <<<".format(localRatio));
            print(" "*5,"Delta at maxPos:\t\t{0:.8f} <<<".format(getDeltaEnd(L,localRatio,nCells)));
            print(" "*5,"Delta at minPos:\t\t{0:.8f} <<<".format(getDeltaStart(L,localRatio,nCells)));
            print("|","="*70,"|");
            print("\n");
        else:
            if (posMinProvided and posMaxProvided):
                L = np.float64(posMax - posMin); #type: ignore
            else:
                raise RuntimeError('Neither total length or positions were provided.');
            print("\n");
            print("|","="*70,"|");
            print("|"," "*2,"Results for given length, global ratio, and number of cells."," "*4,"  |")
            print("|"," "*2,"Total length: \t\t{0:.8f} <<<".format(L));
            print("|"," "*2,"Global ratio: \t\t{0:.8f} <<<".format(globalRatio));
            print("|"," "*2,"Number of cells:\t\t{0:d} <<<".format(nCells));
            print("|"," "*2,"Local ratio:  \t\t{0:.8f} <<<".format(localRatio));
            print("|"," "*2,"Delta at maxPos:\t\t{0:.8f} <<<".format(getDeltaEnd(L,localRatio,nCells)));
            print("|"," "*2,"Delta at minPos:\t\t{0:.8f} <<<".format(getDeltaStart(L,localRatio,nCells)));
            print("|","="*70,"|");
            print("\n");

    if (deltaStartProvided): # type: ignore
        if (not totalLProvided):
            if (posMinProvided and posMaxProvided):
                L = np.float64(posMax - posMin); #type: ignore
            else:
                raise RuntimeError('Neither total length or positions were provided.');
            
        if (nCellsProvided):
            localRatio, globalRatio = findRatioDeltaStart(L,deltaStart,nCells,args.verbose);
            print("\n");
            print("|","="*70,"|");
            print("|"," "*2,"Results for given length, start-cell spacing and number of cells"," "*2,"|")
            print("|"," "*2,"Total length: \t\t{0:.8f} <<<".format(L));
            print("|"," "*2,"Delta at minPos:\t\t{0:.8f} <<<".format(deltaStart));
            print("|"," "*2,"Number of cells:\t\t{0:d} <<<".format(nCells));
            print("|"," "*2,"Global ratio: \t\t{0:.8f} <<<".format(globalRatio));
            print("|"," "*2,"Local ratio:  \t\t{0:.8f} <<<".format(localRatio));
            print("|"," "*2,"Delta at maxPos:\t\t{0:.8f} <<<".format(np.float64(globalRatio*deltaStart)));
            print("|","="*70,"|");
            print("\n");
        else:
            raise RuntimeError('Number of cells not provided.');

    if (deltaEndProvided): # type: ignore
        if (not totalLProvided):
            if (posMinProvided and posMaxProvided):
                L = np.float64(posMax - posMin); #type: ignore
            else:
                raise RuntimeError('Neither total length or positions were provided.');
            
        if (nCellsProvided):
            localRatio, globalRatio = findRatioDeltaEnd(L,deltaEnd,nCells,args.verbose);
            print("\n");
            print("|","="*70,"|");
            print("|"," "*2,"Results for given length, end-cell spacing and number of cells"," "*2,"  |")
            print("|"," "*2,"Total length: \t\t{0:.8f} <<<".format(L));
            print("|"," "*2,"Delta at maxPos:\t{0:.8f} <<<".format(deltaEnd));
            print("|"," "*2,"Number of cells:\t\t{0:d} <<<".format(nCells));
            print("|"," "*2,"Global ratio: \t\t{0:.8f} <<<".format(globalRatio));
            print("|"," "*2,"Local ratio:  \t\t{0:.8f} <<<".format(localRatio));
            print("|"," "*2,"Delta at minPos:\t{0:.8f} <<<".format(np.float64(deltaEnd/globalRatio)));
            print("|","="*70,"|");
            print("\n");
        else:
            raise RuntimeError('Number of cells not provided.');


if __name__ == "__main__":
    main()
