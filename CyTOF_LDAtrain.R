CyTOF_LDAtrain <- function (TrainingSamplesExt,TrainingLabelsExt,mode,
                                 RelevantMarkers,LabelIndex = FALSE,Transformation){
  
  # CyTOF_LDAtrain function can be used to train a Linear Discriminant
  # Analysis (LDA) classifier, on the labeled CyTOF samples. The trained
  # classifier model can be then used to automatically annotate new CyTOF
  # samples.
  #
  # Input description
  #
  # TrainingSamplesExt: extension of the folder containing the training samples,
  #                     can be either in FCS or csv format.
  #
  # TrainingLabelsExt: extention of the folder containing the labels (cell types)
  #                    for the training samples, must be in csv format. Labels
  #                    files must be in the same order as the samples, to match
  #                    each lables file to the corresponding sample.
  #
  # mode:         either 'FCS' or 'CSV', defining the samples format.
  #
  # RelevantMarkers: list of integers, enumerating the markers to be used for
  #                  classification.
  #
  # LabelIndex: Integer value indicating the column containing the labels of each cell, 
  #             in case it exists in the same files with the samples. In such case, 'TrainingLabelsExt'
  #             is not used.
  #             FALSE (default) = labels are stored in separate csv files found in 'TrainingLabelsExt' 
  #
  # Transformation: 'arcsinh' = apply arcsinh transformation with cofactor of 5 prior to classifier training
  #                 'log' = apply logarithmic transformation proir to classifier training
  #                 FALSE = no transformation applied.
  #
  # Example
  # Model <- CyTOF_LDAtrain('HMIS-2/Samples/','HMIS-2/Labels/','CSV',c(1:28),'arcsinh')
  #
  # For citation and further information please refer to this publication:
  # "Predicting cell populations in single cell mass cytometry data"
  
  Training.data = data.frame()
  Training.labels <- data.frame()
  if (mode == 'FCS'){
    files = list.files(path = TrainingSamplesExt, pattern = '.fcs',full.names = TRUE)
    
    library(flowCore)
    for (i in files){
      Temp <- read.FCS(i,transformation = FALSE, truncate_max_range = FALSE)
      colnames(Temp@exprs) <- Temp@parameters@data$name
      Training.data = rbind(Training.data,as.data.frame(Temp@exprs)[,RelevantMarkers])
      if(LabelIndex){
        Training.labels = rbind(Training.labels,as.data.frame(Temp@exprs)[,LabelIndex,drop = FALSE])
      }
    }
  }
  else if (mode == 'CSV'){
    files = list.files(path = TrainingSamplesExt, pattern = '.csv',full.names = TRUE)
    
    for (i in files){
      Temp <- read.csv(i,header = FALSE)
      Training.data = rbind(Training.data,as.data.frame(Temp)[,RelevantMarkers])
      if(LabelIndex){
        Training.labels = rbind(Training.labels,as.data.frame(Temp)[,LabelIndex,drop = FALSE])
      }
    }
  }
  else{
    stop('Invalid file format mode, choose FCS or CSV')
  }
  
  if(Transformation != FALSE){
    if(Transformation == 'arcsinh'){
      Training.data = asinh(Training.data/5)
    }
    else if (Transformation == 'log'){
      Training.data = log(Training.data)
      Training.data[sapply(Training.data,is.infinite)] = 0
    }
  }
  
  if(!LabelIndex){
    Labels.files <- list.files(path = TrainingLabelsExt, pattern = '.csv', full.names = TRUE)
    
    for (i in Labels.files){
      Temp <- read.csv(i,header = FALSE, colClasses = "character")
      Training.labels = rbind(Training.labels,as.data.frame(Temp))
    }
    
    if(dim(Training.data)[1]!=dim(Training.labels)[1]){
      stop('Length of training data and labels does not match')}
  }
  
    
  LDAclassifier <- MASS::lda(Training.data,as.factor(Training.labels[,1]))
  
  Model <- list(LDAclassifier = LDAclassifier,Transformation = Transformation, markers = RelevantMarkers)
  return(Model)
 
}
