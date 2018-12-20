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
%% run LDA Classifier with 5-fold cross-validation

CVO = cvpartition(Labels,'k',5);
Accuracy = zeros(CVO.NumTestSets,1);
training_time = zeros(CVO.NumTestSets,1);
testing_time = zeros(CVO.NumTestSets,1);
CellTypes = unique(Labels);
ConfusionMat = zeros(length(CellTypes));
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    DataTrain = Data(trIdx,:);
    LabelsTrain = Labels(trIdx);
    tic
    classificationLDA = fitcdiscr(...
        DataTrain(LabelsTrain~=0,:), ...
        LabelsTrain(LabelsTrain~=0));
    training_time(i)=toc;          %in seconds
    
    tic
    [Predictor,scores] = predict(classificationLDA,Data(teIdx,:));
    Current_Scores = max(scores,[],2);
    Predictor(Current_Scores < 0.4)=0;
    testing_time(i)=toc;           %in seconds
    LabelsTest = Labels(teIdx);
    Accuracy(i) = nnz((Predictor(LabelsTest~=0)==LabelsTest(LabelsTest~=0)))/size(LabelsTest(LabelsTest~=0),1);
    ConfusionMat = ConfusionMat + confusionmat(LabelsTest,Predictor,'order',CellTypes);
end
Total_time = sum(training_time)+sum(testing_time);
training_time = mean(training_time);
testing_time = mean(testing_time);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
disp(['LDA Accuracy = ' num2str(cvAcc) ' ' char(177) ' ' num2str(cvSTD) ' %'])
clear i Predictor classificationLDA trIdx teIdx CVO Accuracy DataTrain LabelsTrain
clear DataTest LabelsTest
%% Performance evaluation
col1 = ConfusionMat(2:end,1);
ConfusionMat = ConfusionMat(2:end,2:end);
% F1 measure
Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
Recall = diag(ConfusionMat)./(sum(ConfusionMat,2)+col1);
Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
MedianFmeasure = median(Fmeasure);
Subset_size = sum(ConfusionMat,2)+col1;
WeightedFmeasure = (Subset_size./sum(Subset_size))'*Fmeasure;

disp(['Weighted F1-score = ' num2str(WeightedFmeasure)])
figure,scatter(log10(Subset_size),Fmeasure,100,'filled'),title('Multi-Center')
xlabel('Log10(population size)'),ylabel('F1-score'),box on, grid on
%% Population Frequency

True_Freq = (sum(ConfusionMat,2)+col1)./(sum(sum(ConfusionMat))+sum(col1));
Predicted_Freq = sum(ConfusionMat,1)'./(sum(sum(ConfusionMat))+sum(col1));
Max_Freq_diff = max(abs(True_Freq-Predicted_Freq))*100;

disp(['delta_f = ' num2str(Max_Freq_diff)])
figure,bar([True_Freq*100 Predicted_Freq*100])
xticklabels({'B Cells','CD4+ T Cells','CD8+ T Cells','Monocytes'})
set(gca,'FontSize',20)
legend({'True','Predicted'},'FontSize',15)
legend show
ylabel('Freq. %'),title('Multi-Center')
%% Population Frequency scatter plot

CellTypes = {'B cells','CD4+ T cells','CD8+ T cells','Monocytes'};
X=log(True_Freq*100);
Y=log(Predicted_Freq*100);
figure,scatter(X,Y,50,'filled')
box on, grid on
xlabel('Log(True frequency %)'),ylabel('Log(Predicted frequency %)')
title('Multi-Center')
for k=1:length(CellTypes)
    text(X(k),Y(k),CellTypes{k})
end
lsline
text(3,3,['R = ' num2str(corr(X,Y))])
