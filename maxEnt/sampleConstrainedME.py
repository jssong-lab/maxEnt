from __future__ import division
import numpy as np
import keras 
import keras.backend as K
from keras.models import Model
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class layerSampler(object):
    """
    Class for sampling constrained maxEnt distribution associated with a feed-forward 
    keras model. After initialization, perform interpretation with sample method.
    """
    def __init__(self , model, beta , mu ,layerIdx, outputUnitIdxs ,  
                debug = True, s_rng = None , seed = None):
        """
        Inputs: 
            model - keras sequential model
            beta - positive float
            mu - (float) natural logrithm of the ratio of G/C frequency genome wide to A/T frequency genome wide
            layerIdx - (int) index of the hidden layer whose activations constrain maxEnt distribution 
                            (should be the layer taken as input by the network's output layer)
            outputUnitIdxs- tuple  of indices of output untis . If length 1 then use weights of corresponding output unit
                            to scale distances  in space of penultimate acitvations . If length 2 then scale distances in 
                            space of penultimate acitvations by log( P(outputUnits[0] |x  ) / P(outputUnits[1] |x  ) 
                            where P( |x) denomts the class probability assigned by the network the interpreted input
        """
        
        self.inputT = model.input
        self.targetLayer = model.layers[layerIdx]
        if debug:
            print "sampling using similarity of represtation at layer {:d} of type {}".format(layerIdx, 
                                                                                          type(self.targetLayer) ) 
        self.targetLayerT = self.targetLayer.output
        self.scaleT = self.getPenultimeDistanceScaleTensor(outputLayer = model.layers[layerIdx +1],  
                                                        outputUnitIdxs=  outputUnitIdxs)
        self.beta = beta
        self.mu = mu
        self.Model = Model(input = self.inputT , output= self.targetLayerT )  ## defines an op taking input layer tensor as input and returning the activation tensor for target hidden layer
                
        if s_rng is None:
            s_rng =RandomStreams(seed = seed )
        self.s_rng = s_rng

    ### Helper methods #################################################
    def getPenultimeDistanceScaleTensor(self, outputLayer, outputUnitIdxs  ):
        """
        construct the 1d tensor used to scale penultimate activations when calculating distance
        Inputs:
            outputLayer - keras layer with .W attribute returning symbolic variable of weights
             outputUnitIdxs  - see __init__ 
        outputs:
            scaleT - 1d tensor scaling penutimate activations 
        """
        if len(outputUnitIdxs) ==1:
             scaleT = outputLayer.W[:,outputUnitIdxs[0]]
        else:
            assert len(outputUnitIdxs)==2 , "length of ouputUnitIdxs must be 1 or 2"
            scaleT = outputLayer.W[:, outputUnitIdxs[0]] - outputLayer.W[:, outputUnitIdxs[1]]
        return scaleT
    
    def reshapeSamplingOutput(self, samplingOutput , nbInterpInputs , chainsPerInput  ):
        """
        Reshape samplingOutput ordered by timestep, then chainIndex to array ordered by interpretedInput (ax0), then
        by chainIndex and timestep (with timestep dominating chainIdex in determining ordering) (ax1).
        Inputs:
            samplingOutput - (nbSamplesPerChain , nbChains ,  1 , seqLen ) array where samplingOutput[i,j,0,:]
                             returns ith sample collected for jth chain
            nbInterpInputs - (int) number of interpreted inputs ,
            chainsPerInput  - (int) number chain per interpreted input
            seqLen - lenght of the last axis of samplingOutput
        Outputs:
            samplesByInterpInput - (nbInterpInputs , chainsPerInput*samplesPerChain , seqLen )
                                    where for samples collected with x chains for input sampletime j for chain k
                                    associated with  inpterpreted input i is given by
                                    samplesByInterpInput[i , x*j + k , :]                           
        """
        samplingOutput = np.squeeze(samplingOutput)
        samplesPerChain , _  , seqLen = samplingOutput.shape 
        ## split axis 1 into two axes indexing network input and chain assocaited with the input
        samplingOutput = samplingOutput.reshape(( samplesPerChain ,         #  samplingOut[i,j,k ,:] stores
                                                        nbInterpInputs ,           # sample at ith timepoint of kth chain
                                                        chainsPerInput,            # associated with jth interpreted input
                                                        seqLen ) , order = 'C')
        ## collapse axes indexing chains sampling time and chains assocaited with given input into a single axes
        ## ordered by time and then by chain index
        samplesByInterpInput = (samplingOutput.swapaxes(0,1)).reshape((nbInterpInputs, samplesPerChain*chainsPerInput,
                                                                      seqLen) , order = 'C')
        return  samplesByInterpInput    
    
    #### Methods defining symbolic operations of MCMC step ##############################
    
    def getProposedStates(self, states, ntIdxs, s_rng , moves):
        """
        sample proposed states given the current states described by variables states and ntIdxs
        Inputs:
            states - symbolic variable  (nbInputs , 1, seqLen,4) 
            ntIdxs  - symbolic variable (nbInputs , 1,seqLen) 
                    indexing the type of nucleotide as each sequence position
            ntCounts - symbolic variable (nbInputs , 4)  storing the counts of each 
                        nucleotide type each MCMC chain being run
            s_rng- theano.tensor.shared_randomstreams object
            moves - (4,4) float tensor where moves[i] stores the 4 input unit
                        activations encoding nucleotide of type i
        Returns:
            states_proposed -  symbolic variable  (nbInputs , 1, seqLen,4) 
            ntIdxs_proposed -  symbolic variable (nbInputs , 1,seqLen) 
            ntCounts_incr - symbolic variable (nbInputs , 4) (elements are 1 , -1 , 0) indicated the increments to counts
                            of each type of nucleotide associated with the proposed mutation     
        """
        idxs2Mutate = s_rng.random_integers(size =(states.shape[0] , ) ,low =  0 ,  
                                         high= states.shape[-2] -1 )  ## the sequence positions at which mutations are proposed. 
        ntPermInts =  s_rng.random_integers(size =(states.shape[0] , ) ,low =  1 ,
                                           high = states.shape[-1] -1 ) ## sample integers that will be used to permute the nucleotide content at positions selected for mutation
        ## get the indices of the nucleotides before and after mutation
        mutatedNT_idxs_old =  ntIdxs[T.arange(states.shape[0]), T.arange(1, dtype="int64"), idxs2Mutate] 
        mutatedNT_idxs_new = T.mod( ntPermInts +  mutatedNT_idxs_old  , 4)
        ## uptade description of the states
        ntIdxs_proposed =  T.set_subtensor( ntIdxs[T.arange(states.shape[0], dtype="int64") , T.arange(1, dtype="int64"),  idxs2Mutate] ,
                                          mutatedNT_idxs_new ) 
        states_proposed =  T.set_subtensor(states[T.arange(states.shape[0],dtype="int64") ,  T.arange(1, dtype="int64") ,  idxs2Mutate,: ],
                                          moves[mutatedNT_idxs_new])
        ## update the counts of each nucleotide type for each input
        ## approach is to convert the indices of mutated nucleotides to 1 hot encoding then use this to update
        ## the ntCounts matrix
        ntCounts_incr = T.extra_ops.to_one_hot( mutatedNT_idxs_new  , nb_class = 4, dtype  = "int32" ) - \
                            T.extra_ops.to_one_hot( mutatedNT_idxs_old  , nb_class = 4, dtype  = "int32" )
        
        return states_proposed, ntIdxs_proposed , ntCounts_incr 
    
    def metropolisAccept_proposal(self, E_incr, N_incr,  states_proposed ,  ntIdxs_proposed,
                                  states_current,  ntIdxs_current, E_current,
                                  beta, mu, s_rng):
        """
        Accept proposed mutations according to metropolis criteria
        Inputs
            E_incr : (ntInputs, ) symbolic variable storing Energy_proposed - Energy_current
            N_incr : (nbInputs) symbolic variable storing N_proposed - N_current where N denotes counts of a collection of
                    user specified nucleotides
            states_proposed(_current) - symbolic variable  (nbInputs , 1, seqLen,4) describing proposed (current) state
            ntIdxs_proposed(_current)  - symbolic vaiable (nbInputs , 1,seqLen) 
                                       indexing the type of nucleotide as each sequence position of proposed (current) state
            E_current :  (ntInputs, ) sybolic variables storing energy of the current states
            beta - scalar
            mu - scalar
        Returns
        """
        accept = T.gt(T.exp( -1.0*beta * E_incr + mu * N_incr ), s_rng.uniform(size=E_incr.shape) )
        ## construct tensors used to update states_current , ntIdxs_current and expFactor_current
        ## accept.dimshuffle(...) broadcasts the boolean array over all axis after axis 0
        state_current_new = T.switch(accept.dimshuffle( 0, *(('x',) * (states_current.ndim - 1)) ),
                                     states_proposed  , states_current )
        ntIdxs_current_new = T.switch(accept.dimshuffle( 0, *(('x',) * (ntIdxs_current.ndim - 1)) ),
                                      ntIdxs_proposed , ntIdxs_current )
        E_current_new = accept*E_incr + E_current ## using accept (1d array of 1 and 0 as mask)
    
        return state_current_new, ntIdxs_current_new, E_current_new
    
    def calcEnergy(self, hiddenAct, targetT , scaleT,  ):
        """
        hiddenAct - (nbInputs , nbHiddenUnits ) symbolic float matrix storing  hidden layer 
                        representation of MCMC samples
        targetT  -  (nbInputs , nbHiddenUnits ) shared float matrix storing the hidden layer
                        representations of interpreted inputs
        scaleT - (nbHiddenUnits,) 1d tensor defining elements-wise scaling factors 
                for hidden unit activations used for calculating weighted euclidean dist
        """
        energy = T.sqrt( 
                         T.sum( (scaleT*(  hiddenAct - targetT))**2.0 , axis = -1 ) 
                             )
        return energy   
        
    def MCMC_step(self, states_proposed, ntIdxs_proposed,
                   states_current, ntIdxs_current, 
                  E_current, ntCounts_incr ,
                  moves , targetT, constrained_ntIdxs  ):
        """
        given current and proposed states and current exponential factors.
            1. compute the exponential factors of the proposed state
            2. apply metropolis accept criteria to exponential factors of proposed and current
                states and get updated value of current states as well as updated values of
                exponential factors
            3. get updated proposed states 
            
        Inputs: 
            ntCounts_incr : symbolic variable (nbInputs , 4) (elements are 1 , -1 , 0) indicated the 
                        increments to counts of each type of nucleotide 
            constrained_ntIdxs : ivector (ie  a 1d array indexing nucleotides whose average sequence content is
                                            constrained via the parameter mu)
        """
        ## compute the exponential factors of the proposed state
        hiddenAct = self.Model(states_proposed)
        E_incr = self.calcEnergy(hiddenAct, targetT , self.scaleT,  )  - E_current
        N_incr = T.sum(ntCounts_incr[:,constrained_ntIdxs] ,axis = -1 )
        ## apply metropolis accept criteria
        state_current_new, ntIdxs_current_new, E_current_new= self.metropolisAccept_proposal(E_incr, 
                                                                                        N_incr, 
                                                                                        states_proposed ,  
                                                                                        ntIdxs_proposed, 
                                                                                        states_current, 
                                                                                        ntIdxs_current, 
                                                                                        E_current,
                                                                                        self.beta, 
                                                                                        self.mu, 
                                                                                        self.s_rng)
        ## get updated proposed states 
        states_proposed_new, ntIdxs_proposed_new, ntCounts_incr_new = self.getProposedStates(state_current_new,
                                                                                      ntIdxs_current_new,
                                                                                         self.s_rng ,
                                                                                         moves)
        return  states_proposed_new, ntIdxs_proposed_new, state_current_new, ntIdxs_current_new,  E_current_new, ntCounts_incr_new 

    def get_samplingFunc(self ):
        """
        construct the theano function used to do sampling
        Note: current code assumes that nucleotide indices can be obtained from
            input layer activations by applying argmax to last axis 
        """
        ## set-up symbolic variables for inputs to theano function
        interpInputs = T.tensor4(name = "interpInputs")
        moves = T.matrix(name = "moves")
        constrained_ntIdxs  = T.ivector(name ="constrained_ntIdxs" )
        nbSteps = T.iscalar(name = 'nbSteps')
        samplePeriod = T.iscalar(name = 'samplePeriod')
    
        ## intialize the symbolic variables passed to scan
        ntIdxs_initial = T.argmax(interpInputs , axis = -1)
        targetT = self.Model(interpInputs) 
        E_initial = self.calcEnergy(targetT , targetT , self.scaleT ) ## should be 0's
        ntCounts_incr_initial = T.zeros(shape = (interpInputs.shape[0] , 4) , dtype="int32" )
        
        results , updates = theano.scan(fn = self.MCMC_step ,
                                         outputs_info = [interpInputs , ntIdxs_initial , 
                                                        interpInputs , ntIdxs_initial ,
                                                        E_initial,  ntCounts_incr_initial ],
                                       non_sequences = [moves , targetT  , constrained_ntIdxs ],
                                       n_steps = nbSteps)
        samples = results[3][::samplePeriod, :,:]  ## get ntIdxs_current_new entry     
        samplingFunc = theano.function(inputs = [interpInputs , nbSteps, samplePeriod, constrained_ntIdxs , moves ,  K.learning_phase() ],
                                        outputs = samples , updates= updates , on_unused_input = 'ignore')
        return samplingFunc
          
    def sample(self, interpInputs, constrained_ntIdxs = np.array([0,1] , dtype = np.int32) 
               ,moves = np.eye(4, dtype= theano.config.floatX),
                nbSteps = 500 , samplePeriod =100,  chainsPerInput=1 ):
        """
        collect samples for a collection of inputs in interpInputs. 

        interpInputs - (nbInputs ,1, seqlen, 4)
        """
        ## inputs to sampling function are for the form [seq0_chain0, ... , seq0_chain_nbChains,
        ##  seq1_chain0_initital, ... , seq1_chain_nbChains_initial , ...]
        ## where for each k seqi_chaink is a copy of the sequence interpInputs[i]
        nbInterpInputs , _ , seqLen, _ = interpInputs.shape
        samplingInputs = np.repeat( interpInputs, repeats=chainsPerInput, axis = 0 ).astype(theano.config.floatX)
        samplingFunc = self.get_samplingFunc()
        samplingOutput = samplingFunc( samplingInputs, nbSteps ,                 ## samplingOutput[i,j,0,:]
                                      samplePeriod ,constrained_ntIdxs, moves, 0 )  ##  returns ith sample collected
                                                                                 ## for jth axis 0 entry of samplingInputs
        samplesByInterpInput = self.reshapeSamplingOutput(samplingOutput , nbInterpInputs , chainsPerInput )
    
        return  samplesByInterpInput 
