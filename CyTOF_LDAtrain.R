CyTOF_LDAtrain <- function (TrainingSamplesExt,TrainingLabelsExt,mode,
                                 RelevantMarkers,arcsinhTrans){
  
  # CyTOF_LDAtrain function can be used to train a Linear Discriminant
  # Analysis (LDA) classifier, on the labeled CyTOF samples. The trained
  # classifier model can be then used to automatically annotate new CyTOF
  # samples.
  #
  # Input description
  #
  # TrainingSamplesExt: extension of the folder containing the training samples,
  #                can be either in FCS or csv format.
  #
  # TrainingLabelsExt: extention of the folder containing the labels (cell types)
  #               for the training samples, must be in csv format. Labels
  #               files must be in the same order as the samples, to match
  #               each lables file to the corresponding sample.
  #
  # mode:         either 'FCS' or 'CSV', defining the samples format.
  #
  # RelevantMarkers: list of integers, enumerating the markers to be used for
  #                  classification.
  #
  # arcsinhTrans: True = apply arcsinh transformation with cofactor of 5
  #                    prior to train the classifier.
  #                    False = no transformation applied.
  #
  # Example
  # Model <- CyTOF_LDAtrain('HMIS-2/Samples/','HMIS-2/Labels/','CSV',c(1:28),True)
  #
  # For citation and further information please refer to this publication:
  # "Predicting cell types in single cell mass cytometry data"
  
  
  if (mode == 'FCS'){
    files = list.files(path = TrainingSamplesExt, pattern = '.fcs',full.names = TRUE)
    Training.data = data.frame()
    
    library(flowCore)
    for (i in files){
      Temp <- read.FCS(i,transformation = FALSE, truncate_max_range = FALSE)
      colnames(Temp@exprs) <- Temp@parameters@data$desc
      Training.data = rbind(Training.data,as.data.frame(Temp@exprs)[,RelevantMarkers])
    }
  }
  else if (mode == 'CSV'){
    files = list.files(path = TrainingSamplesExt, pattern = '.csv',full.names = TRUE)
    Training.data = data.frame()
    
    for (i in files){
      Temp <- read.csv(i,header = FALSE)
      Training.data = rbind(Training.data,as.data.frame(Temp)[,RelevantMarkers])
    }
  }
  else{
    stop('Invalid file format mode, choose FCS or CSV')
  }
  
  if(arcsinhTrans){
    Training.data = asinh(Training.data/5)
  }
  
  Labels.files <- list.files(path = TrainingLabelsExt, pattern = '.csv', full.names = TRUE)
  Training.labels <- data.frame()
  for (i in Labels.files){
    Temp <- read.csv(i,header = FALSE, colClasses = "character")
    Training.labels = rbind(Training.labels,as.data.frame(Temp))
  }
  
  if(dim(Training.data)[1]!=dim(Training.labels)[1]){
    stop('Length of training data and labels does not match')}
    
  LDAclassifier <- MASS::lda(Training.data,as.factor(Training.labels[,1]))
  
  Model <- list(LDAclassifier = LDAclassifier,arcsinh = arcsinhTrans, markers = RelevantMarkers)
  return(Model)
 
}