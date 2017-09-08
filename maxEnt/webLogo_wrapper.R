#!/usr/bin/Rscript
##TODO: add code to install RWebLogo if library not found

suppressPackageStartupMessages(library("argparse"))
library("RWebLogo")

parser <- ArgumentParser()

parser$add_argument("--fi" , help="path to transfac format motif matrix" )
parser$add_argument("--fo" , help="name for output sequence logo file", default ="" )
parser$add_argument("--seq" , help="name of sequence file used to annotate logo", default ="" )
parser$add_argument("--seqType" , help="either 'dna' for 'rna'", default="dna" )
parser$add_argument("--outputFormat" , help="one of 'pdf' (default) 'eps', 'png', 'jpeg', 'svg'", default= "pdf" )
parser$add_argument("--firstIndex" , help="Index of first position in sequence data (default: 1)", default = 1, type = "integer" )
parser$add_argument("--lower" , help="lower bound index of sequence to display",  default = 1 , type = "integer" )
parser$add_argument("--upper" , help="Upper bound index of sequence to display",  default = -1 , type = "integer" )
parser$add_argument("--stacksPerLine" , help="",  default = 40 , type = "integer" )
parser$add_argument("--aspectRatio" , help="Ratio of stack height to width (default: 5)" , default=5, type="integer")

args<-parser$parse_args()


## if the sequence for annotation is provided load it as vector
if (args$seq != '') 
	{
	line <- readLines(args$seq , n = 1)
#	unlink(args$seq )
	seqVec <- unlist(strsplit(line, split ="") )  ## convert to a vector if sigle character elements
	annotation <- paste(1:length(seqVec), seqVec )
	annotateLogo<- TRUE
} else {
	annotateLogo<- FALSE
}


if (annotateLogo)
	{
	if (args$upper > 0) 
	{
	weblogo(file.in = args$fi , open=FALSE , datatype="transfac" , file.out=args$fo , format=args$outputFormat , 
		sequence.type=args$seqType, size = "medium" ,first.index=args$firstIndex ,lower=args$lower ,upper=args$upper,
		stacks.per.line=args$stacksPerLine, annotate=annotation ,errorbars=FALSE ,color.scheme="classic" , 
		aspect.ratio=args$aspectRatio , rotate.numbers= TRUE )
	} else {
	weblogo(file.in = args$fi , open=FALSE , datatype="transfac" , file.out=args$fo , format=args$outputFormat ,
                sequence.type=args$seqType , size = "medium",first.index=args$firstIndex ,lower=args$lower,
                stacks.per.line=args$stacksPerLine, annotate=annotation ,errorbars=FALSE ,color.scheme="classic" ,
                aspect.ratio=args$aspectRatio , rotate.numbers= TRUE )
	}

} else {
        {
        if (args$upper > 0) 
        {
        weblogo(file.in = args$fi , open=FALSE , datatype="transfac" , file.out=args$fo , format=args$outputFormat ,
                sequence.type=args$seqType ,size = "medium" ,first.index=args$firstIndex ,lower=args$lower ,upper=args$upper,
                stacks.per.line=args$stacksPerLine ,errorbars=FALSE ,color.scheme="classic" ,
                aspect.ratio=args$aspectRatio , rotate.numbers= TRUE )
        } else {
        weblogo(file.in = args$fi , open=FALSE , datatype="transfac" , file.out=args$fo , format=args$outputFormat ,
                sequence.type=args$seqType ,size = "medium" ,   first.index=args$firstIndex ,lower=args$lower,
                stacks.per.line=args$stacksPerLine, errorbars=FALSE ,color.scheme="classic" ,
                aspect.ratio=args$aspectRatio , rotate.numbers= TRUE )
        }
	}
}
