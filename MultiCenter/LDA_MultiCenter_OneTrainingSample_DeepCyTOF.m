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
%% run LDA Classifier with Sample no.2 as training as per DeepCyTOF

CellTypes = unique(Labels);
tic
% training is excluding class 0 'unlabeled'
classificationLDA = fitcdiscr(...
    SamplesData(2).Data(SamplesData(2).Labels~=0,:), ...
    SamplesData(2).Labels(SamplesData(2).Labels~=0));
training_time=toc;          %in seconds
Accuracy = zeros(length(SamplesData),1);
WeightedFmeasure = zeros(length(SamplesData),1);
testing_time = zeros(length(SamplesData),1);
for i = 1:length(SamplesData)
    tic
    [Predictor,scores] = predict(classificationLDA,SamplesData(i).Data);
    Current_Scores = max(scores,[],2);
    Predictor(Current_Scores < 0.4)=0;   % prob < 0.4 = class 0 'unlabeled'
    testing_time(i)=toc;
    Accuracy(i) = nnz((Predictor(SamplesData(i).Labels~=0)==SamplesData(i).Labels(SamplesData(i).Labels~=0)))/size(SamplesData(i).Labels(SamplesData(i).Labels~=0),1);
    ConfusionMat = confusionmat(SamplesData(i).Labels,Predictor,'order',CellTypes);
    col1 = ConfusionMat(2:end,1);
    ConfusionMat = ConfusionMat(2:end,2:end);
    % F1 measure
    Precision = diag(ConfusionMat)./sum(ConfusionMat,1)';
    Recall = diag(ConfusionMat)./(sum(ConfusionMat,2)+col1);
    Fmeasure = 2 * (Precision.*Recall)./(Precision+Recall);
    % MedianFmeasure = median(Fmeasure);
    Subset_size = sum(ConfusionMat,2)+col1;
    WeightedFmeasure(i) = (Subset_size./sum(Subset_size))'*Fmeasure;
end
clear i Predictor classificationLDA ConfusionMat Precision Recall Fmeasure
WeightedFmeasure(2)=[];
Accuracy(2)=[];

MeanWeightedFmeasure=mean(WeightedFmeasure);
cvAcc = mean(Accuracy)*100;
cvSTD = std(Accuracy)*100;
disp(['LDA Accuracy = ' num2str(cvAcc) ' ' char(177) ' ' num2str(cvSTD) ' %'])
disp(['Weighted F1-score = ' num2str(MeanWeightedFmeasure)])

% Supplementary Fig. S2
figure,bar(WeightedFmeasure)
xticklabels([1 3:16])
set(gca,'YLim',[0.7 1])
xlabel('Samples')
ylabel('Weighted F1-score'),title('Multi-Center LDA Performance')
