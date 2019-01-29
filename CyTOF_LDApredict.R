CyTOF_LDApredict <- function (Model,TestingSamplesExt,mode,
                                 RejectionThreshold){
  # CyTOF_LDApredict function can be used to produce automatic cell type
  # annotations for new samples, based on the trained LDA classifier using
  # CyTOF_LDAtrain function.
  #
  # Input description
  #
  # Model: Model produced by CyTOF_LDAtrain, which includes the
  #               trained LDA classifier, the relevant markers used to train
  #               the model and whether or not an arcsinh transformation is
  #               involved.
  #
  # TestingSamplesExt: extension of the folder containing the test samples, can be
  #             either in FCS or CSV format.
  #
  # mode:       either 'FCS' or 'CSV', defining the samples format.
  #
  # RejectionThreshold: Posterior probability lower threshold, below which
  #                     the prediction will be 'unknown', this presents how
  #                     confident the classifier is, before assigning a
  #                     specific cell type to a cell.
  #                     Value between 0 and 1, '0' means no rejection.
  #
  # Example:
  # Celltypes <- CyTOF_LDApredict(Model,'HMIS-2/Samples/','CSV',0.7)
  #
  # For citation and further information please refer to this publication:
  # "Predicting cell types in single cell mass cytometry data"
  
  if (mode == 'FCS'){
    files = list.files(path = TestingSamplesExt, pattern = '.fcs',full.names = TRUE)
    
    Cell.types = list()
    library(flowCore)
    for (i in files){
      Testing.data = data.frame()
      Temp <- read.FCS(i,transformation = FALSE, truncate_max_range = FALSE)
      colnames(Temp@exprs) <- Temp@parameters@data$name
      Testing.data = as.data.frame(Temp@exprs)[,Model$markers]
      
      if(Model$Transformation != FALSE){
        if(Model$Transformation == 'arcsinh'){
          Testing.data = asinh(Testing.data/5)
        }
        else if (Model$Transformation == 'log'){
          Testing.data = log(Testing.data)
          Testing.data[sapply(Testing.data,is.infinite)] = 0
        }
      }
      Predictions <- predict(Model$LDAclassifier,Testing.data)
      Post.max <-apply(Predictions$posterior,1,max)
      Predictions$class <- factor(Predictions$class,levels = c(levels(Predictions$class),"unknown"))
      Predictions$class[Post.max < RejectionThreshold] <- "unknown"
      Predictions <- list(as.character(Predictions$class))
      Cell.types <- c(Cell.types,Predictions)
    }
  }
  else if (mode == 'CSV'){
    files = list.files(path = TestingSamplesExt, pattern = '.csv',full.names = TRUE)
    
    Cell.types = list()
    for (i in files){
      Testing.data = data.frame()
      Temp <- read.csv(i,header = FALSE)
      Testing.data = as.data.frame(Temp)[,Model$markers]
      
      if(Model$Transformation != FALSE){
        if(Model$Transformation == 'arcsinh'){
          Testing.data = asinh(Testing.data/5)
        }
        else if (Model$Transformation == 'log'){
          Testing.data = log(Testing.data)
          Testing.data[sapply(Testing.data,is.infinite)] = 0
        }
      }
      Predictions <- predict(Model$LDAclassifier,Testing.data)
      Post.max <-apply(Predictions$posterior,1,max)
      Predictions$class <- factor(Predictions$class,levels = c(levels(Predictions$class),"unknown"))
      Predictions$class[Post.max < RejectionThreshold] <- "unknown"
      Predictions <- list(as.character(Predictions$class))
      Cell.types <- c(Cell.types,Predictions)
    }
  }
  else{
    stop('Invalid file format mode, choose FCS or CSV')
  }
  return(Cell.types)
}
