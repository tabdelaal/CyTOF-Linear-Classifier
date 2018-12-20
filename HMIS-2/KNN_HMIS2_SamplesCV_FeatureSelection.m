%% Read the Data and Preprocess

VarNames = {'CCR6','CD19','C-KIT','CD11b','CD4','CD8a','CD7','CD25','CD123','TCRgd','CD45',...
    'CRTH2','CD122','CCR7','CD14','CD11c','CD161','CD127','CD8b','CD27','IL-15Ra','CD45RA',...
    'CD3','CD28','CD38','NKp46','PD-1','CD56'};

Samples_Tag = [cellstr(repmat('CeD',4,1)); cellstr(repmat('Ctrl',7,1)); cellstr(repmat('CeD',9,1));...
    cellstr(repmat('Ctrl',7,1)); cellstr(repmat('RCDII',6,1)); cellstr(repmat('CD',14,1))];

SamplesData=struct('Data',[],'Labels',{});
H=dir(fullfile('Samples\', '*.csv'));
SamplesFiles = cellstr(char(H(1:end).name));

H=dir(fullfile('Labels\', '*.csv'));
LabelsFiles = cellstr(char(H(1:end).name));
clear H

for i=1:length(SamplesFiles)
    SamplesData(i).Data = csvread(['Samples\' SamplesFiles{i}]);
    SamplesData(i).Labels = table2cell(readtable(['Labels\' LabelsFiles{i}],'ReadVariableNames',0,'Delimiter',','));
end
clear i SamplesFiles LabelsFiles txt

Labels = [];
for i=1:length(SamplesData)
    Labels = [Labels; SamplesData(i).Labels];
end
CellTypes = unique(Labels);

% remove cells annonated as 'Discard' 
%(very small cell types < 0.1% of the total number of cells)
CellTypes(strcmp('Discard',CellTypes)) = [];
clear i Labels
% Data is already arcsinh(5) transformed
%% run KNN Classifier with 3-fold cross-validation on samples

CVO = cvpartition(1:1:length(SamplesData),'k',3);
Accuracy = zeros(length(SamplesData),1);
training_time = zeros(CVO.NumTestSets,1);
FeatureSelection_time = zeros(CVO.NumTestSets,1);
testing_time = zeros(CVO.NumTestSets,1);
FeatureSelectionMarkers=struct('name',[]);
ConfusionMat = zeros(length(CellTypes));
K=50;
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
    [DataTr,LabelsTr]=KNN_Edit(DataTrain,LabelsTrain);
    DataTr_Size(i)=size(DataTr,1);
    training_time(i)=toc/60;          %in minutes
    
    tic
    [FeatureSelectionMarkers(i).name,Indexes] = MarkersProfile_KNN_FF(DataTr,LabelsTr,VarNames);
    FeatureSelection_time(i)=toc/60;          %in minutes
    
    classificationKNN = fitcknn(...
        DataTr(:,Indexes), ...
        LabelsTr, ...
        'Distance', 'Euclidean', ...
        'NumNeighbors', K,...
        'DistanceWeight', 'Equal', ...
        'Standardize', true);
   tic
    
    for j=1:length(teIdx)
        DataTest = SamplesData(teIdx(j)).Data;
        LabelsTest = SamplesData(teIdx(j)).Labels;
        Predictor = predict(classificationKNN,DataTest(:,Indexes));
        Predictor(strcmp('Discard',LabelsTest)) = [];
        LabelsTest(strcmp('Discard',LabelsTest)) = [];
        Accuracy(teIdx(j)) = nnz(strcmp(Predictor,LabelsTest))/size(LabelsTest,1);
        ConfusionMat = ConfusionMat + confusionmat(LabelsTest,Predictor,'order',CellTypes);
    end
    clear j
    testing_time(i)=toc/60;           %in minutes
end
Total_time = sum(training_time)+sum(FeatureSelection_time)+sum(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
training_time = mean(training_time);
FeatureSelection_time = mean(FeatureSelection_time);
testing_time = mean(testing_time);
disp(['KNN Accuracy = ' num2str(cvAcc) ' ' char(177) ' ' num2str(cvSTD) ' %'])
clear i Predictor classificationKNN trIdx teIdx CVO DataTrain LabelsTrain DataTr LabelsTr
clear DataTest LabelsTest
%% Editing Function
function [Data_train_sample,Labels_train_sample]=KNN_Edit(Data,Labels)
K=50;
X=randperm(size(Data,1),50000);

Data_train_sample=Data(X,:);
Labels_train_sample=Labels(X);
Data(X,:)=[];
Labels(X)=[];
classificationKNN = fitcknn(...
    Data_train_sample, ...
    Labels_train_sample, ...
    'Distance', 'Euclidean', ...
    'NumNeighbors', K,...
    'DistanceWeight', 'Equal', ...
    'Standardize', true);

while (exist('Data','var'))
    if(size(Data,1)>=50000)
        X=randperm(size(Data,1),50000);
        Data_test_sample=Data(X,:);
        Labels_test_sample=Labels(X);
        Data(X,:)=[];
        Labels(X)=[];
    else
        Data_test_sample=Data;
        Labels_test_sample=Labels;
        clear Data Labels
    end
    
    L=predict(classificationKNN,Data_test_sample);
    Data_train_ext=Data_test_sample(~strcmp(L,Labels_test_sample),:);
    Labels_train_ext=Labels_test_sample(~strcmp(L,Labels_test_sample));
    Data_train_sample=[Data_train_sample; Data_train_ext];
    Labels_train_sample=[Labels_train_sample; Labels_train_ext];
    classificationKNN = fitcknn(...
        Data_train_sample, ...
        Labels_train_sample, ...
        'Distance', 'Euclidean', ...
        'NumNeighbors', K,...
        'DistanceWeight', 'Equal', ...
        'Standardize', true);
end
clear L Data_test_sample Data_train_ext X
end
%% Feature selection function
function [Names,Indexes,Plot] = MarkersProfile_KNN_FF(input,Labels,VarNames)
K=50;
Marker_Accuracy=zeros(length(VarNames),1);
CVO = cvpartition(Labels,'k',5);
for i = 1:length(VarNames)
    Accuracy = zeros(CVO.NumTestSets,1);
    for j = 1:CVO.NumTestSets
        trIdx = CVO.training(j);
        teIdx = CVO.test(j);
        
        classificationKNN = fitcknn(...
            input(trIdx,i), ...
            Labels(trIdx), ...
            'Distance', 'Euclidean', ...
            'NumNeighbors', K,...
            'DistanceWeight', 'Equal', ...
            'Standardize', true);
        
        Predictor = predict(classificationKNN,input(teIdx,i));
        Accuracy(j) = nnz(strcmp(Predictor,Labels(teIdx)))/size(Labels(teIdx),1);
    end
    Marker_Accuracy(i) = sum(Accuracy)/CVO.NumTestSets;
end

[Sorted_Accuracy,ranking]=sort(Marker_Accuracy,'descend');

Iteration_Accuracy=zeros(length(VarNames),1);
Iteration_Accuracy(1)=Sorted_Accuracy(1);
CVO = cvpartition(Labels,'k',5);
for i = 2:length(Sorted_Accuracy)
    Accuracy = zeros(CVO.NumTestSets,1);
    for j = 1:CVO.NumTestSets
        trIdx = CVO.training(j);
        teIdx = CVO.test(j);
        
        classificationKNN = fitcknn(...
            input(trIdx,ranking(1:i)), ...
            Labels(trIdx), ...
            'Distance', 'Euclidean', ...
            'NumNeighbors', K,...
            'DistanceWeight', 'Equal', ...
            'Standardize', true);
        
        Predictor = predict(classificationKNN,input(teIdx,ranking(1:i)));
        Accuracy(j) = nnz(strcmp(Predictor,Labels(teIdx)))/size(Labels(teIdx),1);
    end
    Iteration_Accuracy(i) = sum(Accuracy)/CVO.NumTestSets;
end

% figure,plot(1:length(VarNames),Iteration_Accuracy);
Plot=Iteration_Accuracy';
[MaxAcc,I]=max(Iteration_Accuracy);
Indexes=ranking(1:I);
Names=VarNames(Indexes);
disp(['Acheived Accuracy = ' num2str(MaxAcc)]);

end
