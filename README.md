# CyTOF-Linear-Classifier
## LDA on mass cytometry data

This is a Matlab implementation of "Predicting cell types in single cell mass cytometry data", an automated method to annotate the CyTOF dataset.

**Implementation description**

1. function Model = CyTOF_LDAtrain(SamplesFolder,mode,LabelsFolder,RelevantMarkers,arcsinh_transform)

CyTOF_LDAtrain function can be used to train a Linear Discriminant Analysis (LDA) classifier, on the labeled CyTOF samples. The trained classifier model can be then used to automatically annotate new CyTOF samples.

For full description, check CyTOF_LDAtrain.m

2. function Predictions = CyTOF_LDApredict(TrainedModel,DataFolder,mode,RejectionThreshold)

CyTOF_LDApredict function can be used to produce automatic cell type annotations for new samples, based on the trained LDA classifier using CyTOF_LDAtrain function.

For full description, check CyTOF_LDApredict.m

**Tutorials description**

In the following six folders, AML, BMMC, PANORAMA, MultiCenter, HMIS-1 and HMIS-2, we provide the scripts to reproduce the results shown in the publication, including the LDA classifier performance and comparisons with ACDC and DeepCyTOF. Also, we provide a simple documentation in pdf format.

The DeepCyTOF_on_HMIS folder contains python scripts needed to apply DeepCyTOF on our HMIS-1 and HMIS-2 datasets.

All datasets can be downloaded from (https://www.dropbox.com/sh/cf3rinp4oan8ugm/AAD_-G93O72R88-_2McfL2D3a?dl=0)
