%% Cluster subejcts from the PG into two groups using k-means and Squared Euclidean distance metric. 
% Calculates data of different behavioural and hormonal markers of each
% group, and plots results

% Fran?oise Diaz-Rojas, 2021

%% Clustering parameters and network selection
close all

nSess = 3;

contrast_name = 'SgtC';
roi2use = 'dmPFC_L';
voxelDataSaveDir = 'voxelData'; % Directory to save results
voxelDataSaveFile = sprintf('voxelData_%s_%s_PG_perSession.mat',contrast_name,roi2use);
load(fullfile(voxelDataSaveDir,voxelDataSaveFile));

behavDataSaveDir = 'voxelData'; % Directory to save results
behavDataSaveFile = 'behavData_PG_perSession.mat';
load(fullfile(behavDataSaveDir,behavDataSaveFile));


net1_sess = [1,2];
net2_sess = [2,3];

useBin = 0; % Flag for whether to use binarized variables or not.
useColorBkgnd = 1; % Set to 1 to plot the background of the scatter plot 
plotBackground = 0;
plotEdgeCases = 0; % If set to 1, marks "edge cases" (subjects who may belong to one group or another) with a black circle.
remEdgeCases = 0;

barOrViolin = 0; % If set to 1, plot single-session data with bar graphs; 0 uses violin plot

addDiag = 0;
norm_data = 0;
dist2use = 'sqeuclidean'; 
barOrBox = 1;
sortBy = 'BottomRight';

num_of_k = 2;

refSess = 3;

nFolds = 100;
testSize = 4;

grCol = hsv(num_of_k);
rng(1)
rng('default')

xAxisTitle = [];
yAxisTitle = [];

% Calculate change, defined as (Sess X - Sess Y) / |Sess Y|
net_sess = net1_sess;
a = vD_all(:,net_sess(1));
b = vD_all(:,net_sess(2));            
net_voxel = (b-a)./abs(a);
xAxisTitle = sprintf('Change from Session %i to %i - (%s)',...
                     net_sess(1),net_sess(2),roiLabel);
net1_voxel = net_voxel;

net_sess = net2_sess;
a = vD_all(:,net_sess(1));
b = vD_all(:,net_sess(2));     
net_voxel = (b-a)./abs(a);
yAxisTitle = sprintf('Change from Session %i to %i - (%s)',...
                     net_sess(1),net_sess(2),roiLabel);
net2_voxel = net_voxel;

% Absolute value of the change
net1_voxel = abs(net1_voxel);
net2_voxel = abs(net2_voxel);
xAxisTitle = ['Absolute ' xAxisTitle];
yAxisTitle = ['Absolute ' yAxisTitle];

% Logarithm of the previous values
net1_voxel = log10(net1_voxel);
net2_voxel = log10(net2_voxel);     
xAxisTitle = ['Log ' xAxisTitle];
yAxisTitle = ['Log ' yAxisTitle];

% Clip outliers
net1_voxel = filloutliers(net1_voxel,'clip','median');
net2_voxel = filloutliers(net2_voxel,'clip','median');

V = [net1_voxel,net2_voxel];

% Remove subjects that are lacking one session
rmIdx = any(isnan(V),2);
V(rmIdx,:) = [];
vD = vD_all(~rmIdx,:);
Behav = Behav_all(~rmIdx,:,:);

testInd_all = nan(size(V,1),nFolds);

nDots = 100;

x1 = linspace(floor(min(V(:,1))*10)/10,ceil(max(V(:,1))*10)/10,nDots);
x2 = linspace(floor(min(V(:,2))*10)/10,ceil(max(V(:,2))*10)/10,nDots);
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
A = zeros([size(XGrid,1),3]);
C_all = [];

msym = {'o','d','^','s','*'};

trainIdx = zeros(nFolds,testSize);
for iFold=1:nFolds
    trainIdx(iFold,:) = randperm(size(V,1),testSize);
end

% Main routine - repeat clustering process nFolds times to average out
% variations due to initial conditions
for iFold=1:nFolds
    fprintf('Processing permutation %i...\n',iFold)
    targetIdx = trainIdx(iFold,:);
    trainV = V;
    trainV(targetIdx,:) = [];    
    testV = V(targetIdx,:);
    
    [Pgr,C] = kmeans(trainV,num_of_k,'distance',dist2use,'Replicates',100);        
    
    % Sort cluster centres as to always have the same order across folds
    % Sort based on proximty to bottom right corner
    c = zeros(num_of_k,1);
    for i=1:num_of_k
        c(i) = pdist([C(i,:);x1(end),x2(1)]);
    end
        
    [~,sortIdx]= sort(c);
    C = C(sortIdx,:);        
    C_all(iFold,:,:) = C;    
    
    % Find out the cluster that is closest to the test data
    c = zeros(size(testV,1),num_of_k);    
    for j=1:size(testV,1)
        for i=1:num_of_k
            c(j,i) = pdist([C(i,:);testV(j,:)],'squaredeuclidean');
        end
        [~,testInd_all(targetIdx(j),iFold)] = min(c(j,:));   
    end
end

% Plot each participant's data as well as clusters centroids
hf = figure('Color','w');
hold on
xlim([x1(1) x1(end)]), ylim([x2(1) x2(end)])  
for i=1:num_of_k
    scatter(squeeze(nanmean(C_all(:,i,1),1)),squeeze(nanmean(C_all(:,i,2),1)),300,'MarkerFaceColor',grCol(i,:),'MarkerEdgeColor','k','Marker','s','LineWidth',2)        
end
xlabel(xAxisTitle,'Interpreter','none','FontSize',14)
ylabel(yAxisTitle,'Interpreter','none','FontSize',14)

subjGrIdx = kmeans(V,num_of_k,'distance',dist2use,'MaxIter',1,'Start',squeeze(nanmean(C_all,1))); 
for i=1:size(V,1)
    plot(V(i,1),V(i,2),'marker',msym{subjGrIdx(i)},'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',grCol(subjGrIdx(i,1),:))
end
set(gca,'FontName','Calibri','FontSize',14,'MinorGridColor',[0 0 0],'XColor',[0 0 0],'YColor',[0 0 0])

%% Plot brain and other covariates

testInd = subjGrIdx;

grList = [];
N = zeros(num_of_k,1);
for i=1:num_of_k
    grList(i).list = i;
    N(i) = length(find(any(testInd==grList(i).list,2))); % Number of subjects per group
end

var1_data_mean = [];
var1_data_std = [];

figure
hold on
a = [];
b = [];  
b1 = [];
b2 = [];
dummy1 = zeros(size(testInd));    
pptitle = [];
for i=1:length(grList)   
    m = squeeze(nanmean(vD(any(testInd==grList(i).list,2),:),1));
    s = squeeze(nanstd(vD(any(testInd==grList(i).list,2),:),[],1))/sqrt(N(i));
    var0_data_mean(1,(i-1)*nSess+1:(i-1)*nSess+nSess) = m;
    var0_data_std(1,(i-1)*nSess+1:(i-1)*nSess+nSess) = s;
    a(i) = length(find(any(testInd==grList(i).list,2)));
    plot(m,'Color',grCol(i,:)) 
    xlim([0.8 nSess+0.2])
    xticks([1:nSess])
    xticklabels(num2cell(1:nSess))
    errorbar([1:nSess],m,s,'Color',grCol(i,:),'LineStyle','none')  

    vD1 = vD(any(testInd==grList(i).list,2),:);
    sess = repmat([1 2 3],N(i),1);
    [pps(i,:),tbl,stats] = anovan(vD1(:),{sess(:)},'varnames',{'Sess'},'model','full','display','off');
    tbl_all0(1,7+i) = {sprintf('%0.2f (p=%0.4f)',tbl{2,6},tbl{2,7})};   
end
for j=1:6
    tbl_all0(1,j+1) = {sprintf('%0.2f \x00B1 %0.3f',var0_data_mean(1,j),var0_data_std(1,j))};
end
tbl_all0(1,1) = {roiLabel};   

b = vD(:);
c = repmat(testInd,[nSess 1]);
d = reshape(repmat([1:nSess],[size(vD,1) 1]),[],1);
b(c==0) = [];
d(c==0) = [];
c(c==0) = [];
[pp,tbl,stats] = anovan(b,{c,d},'varnames',{'HighLow','Sess'},'model','full','display','off');
title(sprintf('%s\nGr=%.3f - Sess=%.3f - Int=%.3f\nGroup1 - Session: p=%.3f\nGroup2 - Session: p=%.3f',roiLabel,pp(1),pp(2),pp(3),pps(1),pps(2)),'Interpreter','none')
tbl_all0(1,10) = {sprintf('%0.2f (p=%0.4f)',tbl{2,6},tbl{2,7})};
tbl_all0(1,11) = {sprintf('%0.2f (p=%0.4f)',tbl{3,6},tbl{3,7})};
tbl_all0(1,12) = {sprintf('%0.2f (p=%0.4f)',tbl{4,6},tbl{4,7})};


testInd = subjGrIdx;

% Plot line graphs for variables that can change across sessions      
var2use = {'Pmean','Nmean','Corrected DAS','FetalAttach',...
           'Intimacy','TraitAnxiety','Testosterone (no nuis)','Oxytocin (no nuis)'};
  


D = Behav(:,contains(behavVarNames,var2use),:);
varNames2use = behavVarNames(contains(behavVarNames,var2use));

var1_data_mean = [];
var1_data_std = [];


figure
nCol = min([4,length(varNames2use)]);
nRow = ceil(length(varNames2use)/nCol);
pp1 = zeros(length(varNames2use),1);
pp2 = [];
pp3 = [];
varLabels = {[]};
for iVar=1:length(varNames2use)            
    subplot(nRow,nCol,iVar)
    hold on
    b = [];  
    b1 = [];
    b2 = [];
    dummy1 = zeros(size(testInd));    
    pptitle = [];
    for i=1:length(grList) 
        m = squeeze(nanmean(D(any(testInd==grList(i).list,2),iVar,:),1))';
        s = squeeze(nanstd(D(any(testInd==grList(i).list,2),iVar,:),[],1))'/sqrt(N(i));
        var1_data_mean(iVar,(i-1)*nSess+1:(i-1)*nSess+nSess) = m;
        var1_data_std(iVar,(i-1)*nSess+1:(i-1)*nSess+nSess) = s;
        
        plot(m,'Color',grCol(i,:),'LineWidth',2)  
        xlim([0.8 nSess+0.2])
        xticks([1:nSess])
        xticklabels(num2cell([1:nSess]))        
        errorbar([1:nSess],m,s,'Color',grCol(i,:),'LineStyle','none')  

    end
    
    b = squeeze(D(:,iVar,:));
    b = b(:);
    c = repmat(testInd,[nSess 1]);
    d = reshape(repmat([1:nSess],[size(D,1) 1]),[],1);
    b(c==0) = [];
    d(c==0) = [];
    c(c==0) = [];
    [pp,tbl,stats] = anovan(b,{c,d},'varnames',{'Group','Sess'},'model','full','display','off');    
    for j=1:6
        tbl_all(iVar,j+1) = {sprintf('%0.2f \x00B1 %0.3f',var1_data_mean(iVar,j),var1_data_std(iVar,j))};
    end
    tbl_all(iVar,j+2) = {sprintf('%0.2f (p=%0.4f)',tbl{2,6},tbl{2,7})};
    tbl_all(iVar,j+3) = {sprintf('%0.2f (p=%0.4f)',tbl{3,6},tbl{3,7})};
    tbl_all(iVar,j+4) = {sprintf('%0.2f (p=%0.4f)',tbl{4,6},tbl{4,7})};
    varLabels(iVar) = varNames2use(iVar);
    tbl_all(iVar,1) = varLabels(iVar);
    title(sprintf('%s',varLabels{iVar}),'Interpreter','none')    
    set(gca,'LineWidth',2,'FontSize',14,'XColor','k','YColor','k','FontName','Calibri')
end   


% Plot bar graphs for variables that don't change across sessions
var2use = {'Age','Educ','WeeklyWorktime','HouseIncome','Parental Involvement','Time together','PapaSchool - No. Times'};


D = Behav(:,contains(behavVarNames,var2use),3);
varNames2use = behavVarNames(contains(behavVarNames,var2use));
var2_data_mean = [];
var2_data_std = [];

figure
nCol = min([4,length(varNames2use)]);
nRow = ceil(length(varNames2use)/nCol);
nRow = 2;
pp2 = [];
varLabels = {[]};
for iVar=1:length(varNames2use)            
    subplot(nRow,nCol,iVar)
    hold on
    a = [];
    b = [];
    dummy1 = [];
    pptitle = [];
    violins = violinplot(squeeze(D(:,iVar,:)),testInd);
    for i=1:length(grList)
        m = squeeze(nanmean(D(any(testInd==grList(i).list,2),iVar),1));
        s = squeeze(nanstd(D(any(testInd==grList(i).list,2),iVar),[],1))/sqrt(N(i));
        var2_data_mean(iVar,i) = m;
        var2_data_std(iVar,i) = s;  
        violins(i).ViolinColor = grCol(i,:);   
        violins(i).ScatterPlot.MarkerEdgeColor = 'k'; 
    end
    
    b = squeeze(D(:,iVar));
    b = b(:);
    c = testInd;   
    b(c==0) = [];
    c(c==0) = [];
    [pp,tbl,stats] = anovan(b,c,'varnames',{'HighLow'},'display','off');
    varLabels(iVar) = varNames2use(iVar);    
    tbl_all1(iVar,1) = varLabels(iVar); 
    tbl_all1(iVar,2) = {sprintf('%0.2f \x00B1 %0.3f',var2_data_mean(iVar,1),var2_data_std(iVar,1))};
    tbl_all1(iVar,3) = {sprintf('%0.2f \x00B1 %0.3f',var2_data_mean(iVar,2),var2_data_std(iVar,2))};
    tbl_all1(iVar,4) = {sprintf('%0.2f (p=%0.4f)',tbl{2,6},tbl{2,7})};
    tbl_all1(iVar,5) = {sprintf('%0.2f (p=%0.4f)',tbl{3,6},tbl{3,7})};
    tbl_all1(iVar,6) = {sprintf('%0.2f (p=%0.4f)',tbl{4,6},tbl{4,7})};    
    title(sprintf('%s',varLabels{iVar}),'Interpreter','none')
    set(gca,'LineWidth',2,'FontSize',14,'XColor','k','YColor','k','FontName','Calibri')
end   


