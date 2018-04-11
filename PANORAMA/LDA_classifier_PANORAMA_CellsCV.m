%% Read the Data and Preprocess

VarNames = {'Ter119';'CD45.2';'Ly6G';'IgD';'CD11c';'F480';'CD3';'NKp46';'CD23';...
    'CD34';'CD115';'CD19';'120g8';'CD8';'Ly6C';'CD4';'CD11b';'CD27';'CD16_32';...
    'SiglecF';'Foxp3';'B220';'CD5';'FceR1a';'TCRgd';'CCR7';'Sca1';'CD49b';'cKit';...
    'CD150';'CD25';'TCRb';'CD43';'CD64';'CD138';'CD103';'IgM';'CD44';'MHCII'};

SamplesData=struct('Data',[],'Labels',{});
H=dir(fullfile('Samples\', '*.csv'));
SamplesFiles = cellstr(char(H(1:end).name));

H=dir(fullfile('Labels\', '*.xlsx'));
LabelsFiles = cellstr(char(H(1:end).name));
clear H

for i=1:length(SamplesFiles)
    SamplesData(i).Data = csvread(['Samples\' SamplesFiles{i}]);
    [~,txt]=xlsread(['Labels\' LabelsFiles{i}]);
    SamplesData(i).Labels = txt;
end
clear i SamplesFiles LabelsFiles txt

Data=[];
Labels = [];
for i=1:length(SamplesData)
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
    Accuracy(i) = nnz(strcmp(Predictor,Labels(teIdx)))/size(Labels(teIdx),1);
    ConfusionMat = ConfusionMat + confusionmat(Labels(teIdx),Predictor,'order',CellTypes);
    testing_time(i)=toc;           %in seconds
end
Total_time = sum(training_time)+sum(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
training_time = mean(training_time);
testing_time = mean(testing_time);
clear i Predictor classificationLDA trIdx teIdx CVO Accuracy
%% Performance evaluation

% F1 measure
Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
Recall = diag(ConfusionMat)./sum(ConfusionMat,2);
Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
MedianFmeasure = median(Fmeasure);
Subset_size = sum(ConfusionMat,2);
WeightedFmeasure = (Subset_size./size(Data,1))'*Fmeasure;

%% Population Frequency

True_Freq = sum(ConfusionMat,2)./sum(sum(ConfusionMat));
Predicted_Freq = sum(ConfusionMat,1)'./sum(sum(ConfusionMat));
figure,bar([True_Freq Predicted_Freq])
% ticklabels=CellTypes;
% ticklabels = cellfun(@(x) strrep(x,' ','\newline'), ticklabels,'UniformOutput',false);
xticks(1:22)
xticklabels(CellTypes)
xtickangle(90)
set(gca,'FontSize',10)
set(gca,'XLim',[0 23])
legend({'True','Predicted'},'FontSize',10)