%% Read the Data and Preprocess

VarNames = {'CCR6','CD19','C-KIT','CD11b','CD4','CD8a','CD7','CD25','CD123','TCRgd','CD45',...
    'CRTH2','CD122','CCR7','CD14','CD11c','CD161','CD127','CD8b','CD27','IL-15Ra','CD45RA',...
    'CD3','CD28','CD38','NKp46','PD-1','CD56'};

Samples_Tag = [cellstr(repmat('CeD',4,1)); cellstr(repmat('Ctrl',7,1)); cellstr(repmat('CeD',9,1));...
    cellstr(repmat('Ctrl',7,1)); cellstr(repmat('RCDII',6,1)); cellstr(repmat('CD',14,1))];

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

Labels = [];
for i=1:length(SamplesData)
    Labels = [Labels; SamplesData(i).Labels];
end
CellTypes = unique(Labels);
CellTypes(strcmp('Discard',CellTypes)) = [];
clear i Labels

%% run LDA Classifier

CVO = cvpartition(1:1:length(SamplesData),'k',3);
Accuracy = zeros(length(SamplesData),1);
training_time = zeros(CVO.NumTestSets,1);
testing_time = zeros(CVO.NumTestSets,1);
ConfusionMat = zeros(length(CellTypes));
Prediction = [];
Test = [];
for i = 1:CVO.NumTestSets
    trIdx = find(CVO.training(i));
    teIdx = find(CVO.test(i));
    
    DataTrain=[];
    LabelsTrain=[];
    for j=1:length(trIdx)
        DataTrain = [DataTrain; SamplesData(trIdx(j)).Data];
        LabelsTrain = [LabelsTrain; SamplesData(trIdx(j)).Labels];
    end
    clear j
    DataTrain(strcmp('Discard',LabelsTrain),:) = [];
    LabelsTrain(strcmp('Discard',LabelsTrain)) = [];
    
    tic
    classificationLDA = fitcdiscr(...
        DataTrain, ...
        LabelsTrain);
    training_time(i)=toc;          %in seconds
    
    tic
    
    for j=1:length(teIdx)
        DataTest = SamplesData(teIdx(j)).Data;
        LabelsTest = SamplesData(teIdx(j)).Labels;
        
        DataTest(strcmp('Discard',LabelsTest),:) = [];
        LabelsTest(strcmp('Discard',LabelsTest)) = [];
        
        Predictor = predict(classificationLDA,DataTest);
        Accuracy(teIdx(j)) = nnz(strcmp(Predictor,LabelsTest))/size(LabelsTest,1);
        ConfusionMat = ConfusionMat + confusionmat(LabelsTest,Predictor,'order',CellTypes);
        
        Prediction = [Prediction; Predictor];
        Test = [Test; LabelsTest];
    end
    clear j
    testing_time(i)=toc;           %in seconds
end

Total_time = sum(training_time)+sum(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
training_time = mean(training_time);
testing_time = mean(testing_time);
clear i Predictor classificationLDA trIdx teIdx CVO DataTrain LabelsTrain
clear DataTest LabelsTest  
%% Performance evaluation

% F1 measure
% AccuracyPerCluster = diag(ConfusionMat)./sum(ConfusionMat,2);
Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
Recall = diag(ConfusionMat)./sum(ConfusionMat,2);
Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
MedianFmeasure = median(Fmeasure);
Subset_size = sum(ConfusionMat,2);
WeightedFmeasure = (Subset_size./sum(Subset_size))'*Fmeasure;

%% Population Frequency

True_Freq = sum(ConfusionMat,2)./sum(sum(ConfusionMat));
Predicted_Freq = sum(ConfusionMat,1)'./sum(sum(ConfusionMat));
Max_Freq_diff = max(abs(True_Freq-Predicted_Freq))*100;

figure,bar([True_Freq Predicted_Freq])
xticks(1:57)
xticklabels(CellTypes)
xtickangle(90)
set(gca,'FontSize',10)
set(gca,'XLim',[0 58])
legend({'True','Predicted'},'FontSize',10),title('Human Mucosal Immune Dataset')