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

Data=[];
Labels = [];
for i=1:length(SamplesData)
    % Apply arcsinh5 transformation
    SamplesData(i).Data = asinh((SamplesData(i).Data-1)/5);
    Data = [Data; SamplesData(i).Data];
    Labels = [Labels; SamplesData(i).Labels];
end
clear i
%% run LDA Classifier

CVO = cvpartition(Labels,'k',5);
Accuracy = zeros(CVO.NumTestSets,1);
training_time = zeros(CVO.NumTestSets,1);
testing_time = zeros(CVO.NumTestSets,1);
CellTypes = unique(Labels);
ConfusionMat = zeros(length(CellTypes));
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    tic
    classificationLDA = fitcdiscr(...
        Data(trIdx,:), ...
        Labels(trIdx));
    training_time(i)=toc;          %in seconds
    
    tic
    Predictor = predict(classificationLDA,Data(teIdx,:));
    LabelsTest = Labels(teIdx);
    Accuracy(i) = nnz((Predictor(LabelsTest~=0)==LabelsTest(LabelsTest~=0)))/size(LabelsTest(LabelsTest~=0),1);
    ConfusionMat = ConfusionMat + confusionmat(Labels(teIdx),Predictor,'order',CellTypes);
    testing_time(i)=toc;           %in seconds
end
Total_time = sum(training_time)+sum(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
training_time = mean(training_time);
testing_time = mean(testing_time);
clear i Predictor classificationLDA trIdx teIdx CVO Accuracy DataTrain LabelsTrain
clear DataTest LabelsTest
%% Performance evaluation

% F1 measure
Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
Recall = diag(ConfusionMat)./sum(ConfusionMat,2);
Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
MedianFmeasure = median(Fmeasure);
Subset_size = sum(ConfusionMat,2);
WeightedFmeasure = (Subset_size(2:end)./sum(Subset_size(2:end)))'*Fmeasure(2:end);

%% Population Frequency

True_Freq = sum(ConfusionMat,2)./sum(sum(ConfusionMat));
Predicted_Freq = sum(ConfusionMat,1)'./sum(sum(ConfusionMat));

figure,bar([True_Freq(2:end) Predicted_Freq(2:end)])
xticklabels({'B Cells','CD4+ T Cells','CD8+ T Cells','Monocytes'})
set(gca,'FontSize',20)
legend({'True','Predicted'},'FontSize',15)
legend show
ylabel('Frequency'),title('Multi Centre CyTOF Data')