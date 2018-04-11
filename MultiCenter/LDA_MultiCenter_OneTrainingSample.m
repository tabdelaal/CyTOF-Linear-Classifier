%% Read the Data and Preprocess

VarNames = {'CCR6','CD20','CD45','CD14','CD16','CD8','CD3','CD4'};
SamplesData=struct('Data',[],'Labels',{});

H=dir(fullfile('Samples\', '*.csv'));
SamplesFiles = cellstr(char(H(1:end).name));

H=dir(fullfile('Labels\', '*.csv'));
LabelsFiles = cellstr(char(H(1:end).name));
clear H

for i=1:length(SamplesFiles)
    SamplesData(i).Data = csvread(['Samples\' SamplesFiles{i}]);
    SamplesData(i).Labels = csvread(['Labels\' LabelsFiles{i}]);
end
clear i SamplesFiles LabelsFiles

Labels = [];
for i=1:length(SamplesData)
    % Apply arcsinh5 transformation
    SamplesData(i).Data = asinh((SamplesData(i).Data-1)/5);
    Labels = [Labels; SamplesData(i).Labels];
end
clear i
%% run LDA Classifier
tic
CellTypes = unique(Labels);
classificationLDA = fitcdiscr(...
    SamplesData(2).Data, ...
    SamplesData(2).Labels);
Accuracy = zeros(length(SamplesData),1);
WeightedFmeasure = zeros(length(SamplesData),1);
for i = 1:length(SamplesData)
    
    Predictor = predict(classificationLDA,SamplesData(i).Data);
    Accuracy(i) = nnz((Predictor(SamplesData(i).Labels~=0)==SamplesData(i).Labels(SamplesData(i).Labels~=0)))/size(SamplesData(i).Labels(SamplesData(i).Labels~=0),1);
    ConfusionMat = confusionmat(SamplesData(i).Labels,Predictor,'order',CellTypes);
    % F1 measure
    Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
    Recall = diag(ConfusionMat)./sum(ConfusionMat,2);
    Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
    % MedianFmeasure = median(Fmeasure);
    Subset_size = sum(ConfusionMat,2);
    WeightedFmeasure(i) = (Subset_size(2:end)./sum(Subset_size(2:end)))'*Fmeasure(2:end);
end
Total_time = toc;           %in seconds
WeightedFmeasure(2)=[];
Accuracy(2)=[];

MeanWeightedFmeasure=mean(WeightedFmeasure);
MeanAccuracy=mean(Accuracy);
StdAccuracy=std(Accuracy);

clear i Predictor classificationLDA ConfusionMat Precision Recall Fmeasure 