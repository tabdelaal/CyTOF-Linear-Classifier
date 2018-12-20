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
clear i SamplesFiles LabelsFiles

Data=[];
Labels = [];
for i=1:length(SamplesData)
    Data = [Data; SamplesData(i).Data];
    Labels = [Labels; SamplesData(i).Labels];
end
CellTypes = unique(Labels);

% remove cells annonated as 'Discard' 
%(very small cell types < 0.1% of the total number of cells)
CellTypes(strcmp('Discard',CellTypes)) = [];
Data(strcmp('Discard',Labels),:) = [];
Labels(strcmp('Discard',Labels)) = [];
clear i

% Data is already arcsinh(5) transformed
%% run LDA Classifier with 5-fold cross-validation

CVO = cvpartition(Labels,'k',5);
Accuracy = zeros(CVO.NumTestSets,1);
training_time = zeros(CVO.NumTestSets,1);
testing_time = zeros(CVO.NumTestSets,1);
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
    testing_time(i)=toc;           %in seconds
    Accuracy(i) = nnz(strcmp(Predictor,Labels(teIdx)))/size(Labels(teIdx),1);
    ConfusionMat = ConfusionMat + confusionmat(Labels(teIdx),Predictor,'order',CellTypes);
    
end
Total_time = sum(training_time)+sum(testing_time);
training_time = mean(training_time);
testing_time = mean(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
disp(['LDA Accuracy = ' num2str(cvAcc) ' ' char(177) ' ' num2str(cvSTD) ' %'])
clear i Predictor classificationLDA trIdx teIdx CVO DataTrain LabelsTrain
%% Performance evaluation

% F1 measure
Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
Recall = diag(ConfusionMat)./sum(ConfusionMat,2);
Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
MedianFmeasure = median(Fmeasure);
Subset_size = sum(ConfusionMat,2);
WeightedFmeasure = (Subset_size./sum(Subset_size))'*Fmeasure;

disp(['Median F1-score = ' num2str(MedianFmeasure)])
figure,scatter(log10(Subset_size),Fmeasure,100,'filled'),title('HMIS-1')
xlabel('Log10(population size)'),ylabel('F1-score'),box on, grid on
%% Population Frequency

True_Freq = sum(ConfusionMat,2)./sum(sum(ConfusionMat));
Predicted_Freq = sum(ConfusionMat,1)'./sum(sum(ConfusionMat));
Max_Freq_diff = max(abs(True_Freq-Predicted_Freq))*100;

disp(['delta_f = ' num2str(Max_Freq_diff)])
figure,bar([True_Freq*100 Predicted_Freq*100])
xticks(1:6)
xticklabels(CellTypes)
xtickangle(90)
set(gca,'FontSize',10)
set(gca,'XLim',[0 7])
legend({'True','Predicted'},'FontSize',10)
legend show
ylabel('Freq. %'),title('HMIS-1')
%% Population Frequency scatter plot

X=log(True_Freq*100);
Y=log(Predicted_Freq*100);
figure,scatter(X,Y,50,'filled')
box on, grid on
xlabel('Log(True frequency %)'),ylabel('Log(Predicted frequency %)')
title('HMIS-1')
for k=1:length(CellTypes)
    text(X(k),Y(k),CellTypes{k})
end
lsline
text(0,0,['R = ' num2str(corr(X,Y))])
