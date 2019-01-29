# CyTOF-Linear-Classifier
## LDA on mass cytometry data

This is an implementation of the LDA classifier proposed in "Predicting cell populations in single cell mass cytometry data", an automated method to annotate the CyTOF dataset.

**Implementation description**

1. function Model = CyTOF_LDAtrain(SamplesFolder,mode,LabelsFolder,RelevantMarkers,arcsinh_transform)

CyTOF_LDAtrain function can be used to train a Linear Discriminant Analysis (LDA) classifier, on the labeled CyTOF samples. The trained classifier model can be then used to automatically annotate new CyTOF samples.

For full description, check CyTOF_LDAtrain

2. function Predictions = CyTOF_LDApredict(TrainedModel,DataFolder,mode,RejectionThreshold)

CyTOF_LDApredict function can be used to produce automatic cell type annotations for new samples, based on the trained LDA classifier using CyTOF_LDAtrain function.

For full description, check CyTOF_LDApredict

Implementation is available in R and Matlab

The 'Examples' folder contains R notebooks showing how to use the implementation, using CyTOF and Flow Cytometry datasets.

**Experiments code description**

In the following six folders, AML, BMMC, PANORAMA, MultiCenter, HMIS-1 and HMIS-2, we provide Matlab scripts to reproduce the results shown in the pre-print, including the LDA and the Nearest Median classifiers performance, and comparisons with ACDC and DeepCyTOF. Also, we provide a full documentation in pdf format.

Further, the k-NN classifier implementation is available in the HMIS-2 folder, including the editing and the feature selection functions.

The DeepCyTOF_on_HMIS folder contains python scripts needed to apply DeepCyTOF on our HMIS-1 and HMIS-2 datasets.

All datasets can be downloaded from Flow Repository (http://flowrepository.org/id/FR-FCM-ZYTT)
