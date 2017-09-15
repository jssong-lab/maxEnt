from __future__ import division
import numpy as np
import os
import sys
import subprocess
import  matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.utils.np_utils import  to_categorical
import re


def listLayers(model):
    """
    model - keras sequential model
    """
    layerIdx = 0
    for layer in model.layers:
        print "{}\t{}".format(type(layer) , layerIdx)
        layerIdx+=1
    return None
    
def NTidx_to_1Hot(NTidxArr ):
    """
    Convert array describing nucleotide content with integer indices to array describing
    as a 1hot encoding.
    Inputs:
        NTidxArr -  numpy with entries in set {0,1,2,3} which index nuclotide type
    Returns:
        oneHotArr - of shape NTidxArr.shape + (4,) with array entries expanded into length 4 axis 
                    storing the one hot encoding of indexes  {0,1,2,3}. Ie
                    0 - > [1.0 , 0.0 , 0.0 ,0.0]
                    1 - > [0.0 , 1.0 , 0.0 ,0.0]
                    etc.
    """
    origShape = NTidxArr.shape
    oneHotArr = to_categorical(np.ravel( NTidxArr),  nb_classes=4 ).reshape((origShape ) + (4,) )
    return oneHotArr
    
def monoNTmarginalFromSamples(samples_intRep):
    """
    Inputs
        samples_intRep: (nbSample , seqLen) array with entries in  {0,1,2,3}
    Returns:
        (seqLen , 4 ) array of maringal distributions
    """
    ntIdxs = [0,1,2,3]
    counts = np.stack([ np.sum( (samples_intRep== ntIdx).astype(float) , axis =  0 ) \
                          for ntIdx  in ntIdxs ] ).T
    marginalDistrib = counts / np.sum(counts , axis =1)[:,None]

    return marginalDistrib    
    

    
    
def plotSingleNT_marginalDistrib(NTfrequencies,  xvals = None, ax = None ,
                                NTidxKey= { 0 : "G", 1 :"C" , 2: "A", 3 : "T" } ,
                                colorKey  ={  "G" : "orange" , "C" : "blue" , "A": "green" , "T" : "red" },
                                plotKwargs = {} , figsize = (12,4) , grid = True , 
                                xticks = None , xlabel = "" , ylabel="nucleotide frequency",
                                title= "", legendloc = 0 , frame_alpha= 0.7):
    """
    make plot of all single nucleotide frequencies
    Inputs
        NTfrequencies - (seqLen , 4) array ( should satisify np.sum(NTfrequencies ,ax  = -1) = ones )
    Returns:
        fig object if no axis object provided
        ax object if ax object provided 
    """
    ## preliminary
    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax= fig.add_subplot(1,1,1)
        returnFig =True
    else:
        returnFig = False
    if xvals is None:
        xvals = np.arange(0, NTfrequencies.shape[0])
    ## plot
    for NTidx in xrange(0,4,1):
        NTtype =  NTidxKey[NTidx]
        ax.plot(xvals ,NTfrequencies[:,NTidx] , label = NTtype, color = colorKey[NTtype] , **plotKwargs )
    ## cosmetics 
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(grid)
    ax.legend(loc = legendloc , framealpha =   frame_alpha)
    
    if returnFig:
        return fig
    else:
        return ax
        
def plotSeqLogo(countsArr, columnLabels = ['G' , 'C' , 'A' , 'T'] ,
                ax = None , logoFname = "" , transfacMatFname ="" ,
                logoFormat = "pdf",
                webLogoKwargs = { "stacksPerLine" : "100" , 
                                "aspectRatio"  : "5" },
                deleteUnamedLogo = True,
                debug = False):
    """
    countsArr - (seqLen,4 ) numpy array
    webLogoKwargs - a dictionary of keyword arguements to pass to the weblogo_wrapper script 
    
    """          
    ## steps:
    ##    save transfac mat
    ##    save seqLogo
    ##     add plot to ax if provided 
    
    def npArray2Transfac(countsArr, fo_name ,columnLabels = ['A' , 'C', 'G' ,'T'], 
                    motifName = "motif_name" , 
                    speciesName= "species_name"):
        """
        countsArr - (seqLen , 4) array of nucleotide counts for aligned sequences
        """
        ## get the number of base10 digits needed to represent the largest entry
        ## of the transfac matrix
        Fieldwidth =  int(
                        max(np.ceil(np.log10(countsArr.shape[0])) ,
                            np.ceil(np.log10(np.max(countsArr) ) )
                            )
                        )
        bodyFmtStr = '{:0>{width}d}\t{:<{width}d}\t{:<{width}d}\t{:<{width}d}\t{:<{width}d}'
        headerFmtStr = '{:<{width}s}\t{:<{width}s}\t{:<{width}s}\t{:<{width}s}\t{:<{width}s}'
        countsArr = countsArr.astype(int)
        
        f = open(fo_name , 'w')
        print >> f, "ID {}".format(motifName)
        print >> f , "BF {}".format(speciesName)
        print >> f, headerFmtStr.format("P0", *columnLabels, width =Fieldwidth)
        
        for idx ,  row in zip(xrange(1, countsArr.shape[0] +1 , 1 ), countsArr):
            print >> f, bodyFmtStr.format( idx , *row ,width = Fieldwidth  )
        f.close()
        return None
    def getDefaultFname(defaultFname_template, suffix=""):
        """
        give a template for default file name (e.g. ./defaultDir/defaultName)
        1. create defaultDir if does not exits
        2. find the smallest positive int k such that defaultName_k does not exist and return defaultName_k.suffix
        """
        
        if not os.path.isdir(os.path.dirname(defaultFname_template)):
            os.mkdir(os.path.dirname(defaultFname_template) )
        defaultFname_pattern = os.path.basename(defaultFname_template) + r'(\d+)'
        defaultFname_idx = 1 + int(max([ 0 if re.match(defaultFname_pattern  , x ) is None 
                                    else re.match(defaultFname_pattern , x ).group(1) for x in os.listdir(os.path.dirname(defaultFname_template) ) + [""]  ]) )
        defaultFname =  defaultFname_template +"{:d}".format( defaultFname_idx) +  suffix  
        return defaultFname
    def addLogoToAx(logoFname  , ax ):
        img = mpimg.imread(logoFname)
        ax.imshow(img , aspect= "auto")
        ax.axis("off")        
        return None
        
    defaultMotifMatFname = "./.motifMats/tfact_mat"
    defaultSeqLogoFname ="./.motifLogos/logo"
    webLogoScript= os.path.join( os.path.dirname(os.path.realpath(__file__ ) ), "webLogo_wrapper.R"  )

    ## set up default names for the transfac matrix and seqlogo files
    if transfacMatFname == "":
         transfacMatFname =  getDefaultFname(defaultMotifMatFname, suffix=".txt")
    if  logoFname == "":
        logoFname = getDefaultFname(defaultSeqLogoFname, suffix="."+ logoFormat)
    ## write counts matrix to file in transfac format
    npArray2Transfac(countsArr, fo_name = transfacMatFname ,
                    columnLabels = columnLabels, 
                    motifName = "motif_name" , 
                    speciesName= "species_name")
    try:
        ## create the motiflogo
        webLogoScriptArgs = ["--fi" , transfacMatFname , "--fo" , logoFname , "--outputFormat" ,  logoFormat ] + \
                            [ elem for key in webLogoKwargs.iterkeys() for elem in [ "--"+key , webLogoKwargs[key]] ]
        if debug:                    
            print "arguments to webLogoScript are:\n{}".format( webLogoScriptArgs )
            p = subprocess.Popen(["Rscript" , webLogoScript ] + webLogoScriptArgs , stdin = subprocess.PIPE , stdout = subprocess.PIPE , stderr = subprocess.PIPE ,
                                close_fds = True)
            print "webLogoScripts stdout:\n{}".format(p.stdout.read())
            print "webLogoScripts stderr:\n{}".format( p.stderr.read() )
        else:
            exitStatus = subprocess.call(["Rscript" , webLogoScript ] + webLogoScriptArgs)
            if exitStatus !=0:
                raise Exception(webLogoScript + " failed with status {} try calling this function with debug=True".format(exitStatus) )
    ## add plot to matplotlib axis object if provided
        if ax is None:
            print "motif logo saved to {}".format(logoFname)
        else:
            addLogoToAx(logoFname  , ax )
            if deleteUnamedLogo: 
                os.remove(logoFname)
    except:
        print "Error interfacing with Rweblogo. A transfac motif file has been saved at {} for manual plotting".format(transfacMatFname)
    return None
    
    
        
        
        
    