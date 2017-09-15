

CTCFbound_oneHot.npy contains an array representation of 10, 101 bp sequences correctly identified as CTCF bound by CTCF_network.hd5. The array is of shape (10, 101,4) where axis 0 indexes sequences, axis 1 indexes sequence position and axis 2 is a oneHot sequence encoding. The order of the one hot sequence encoding is G,C,A,T.  The sequences are a subset of those used to test the network and come from a data set constructed by [1].  The full data set is available at
http://cnn.csail.mit.edu/motif_discovery/wgEncodeAwgTfbsHaibSknshraCtcfV0416102UniPk/test.data
 and
http://cnn.csail.mit.edu/motif_discovery/wgEncodeAwgTfbsHaibSknshraCtcfV0416102UniPk/train.data

[1]  Zeng H, Edwards MD, Liu G, Gifford DK. Convolutional neural network architectures for predicting DNA-protein binding. Bioinformatics. 2016;32(12):i121-i7. doi: 10.1093/bioinformatics/btw255. PubMed PMID: 27307608; PubMed Central PMCID: PMCPMC4908339 
