%% Read the Data and Preprocess

DataTable = readtable('AML_benchmark.csv');

% remove unneeded columns
DataTable.Time=[];
DataTable.Cell_length=[];
DataTable.DNA1=[];
DataTable.DNA2=[];
DataTable.Viability=[];
DataTable.file_number=[];
DataTable.event_number=[];
DataTable.subject=[];

% Separate Data points and Labels
Labels=DataTable.cell_type;
DataTable.cell_type=[];
Data = table2array(DataTable);
clear DataTable

% clear NotDebrisSinglets
Data(strcmp('NotDebrisSinglets',Labels),:)=[];
Labels(strcmp('NotDebrisSinglets',Labels))=[];

% Apply arcsinh5 transformation
Data=asinh((Data-1)/5);
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
Max_Freq_diff = max(abs(True_Freq-Predicted_Freq))*100;
figure,bar([True_Freq Predicted_Freq])
% ticklabels=CellTypes;
% ticklabels = cellfun(@(x) strrep(x,' ','\newline'), ticklabels,'UniformOutput',false);
xticklabels(CellTypes)
xtickangle(90)
set(gca,'FontSize',15)
legend({'True','Predicted'},'FontSize',15)
legend show
ylabel('Frequency'),title('AML')