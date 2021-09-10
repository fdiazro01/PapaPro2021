% 
clear, clc

%% Parameters


task = 'video';
% group = {'P'}; % Uncomment one for a given group comparison
% group = {'C'};
group = {'P','C'};
smoothing = 'noSmooth';
nSess = 3;

correctSig = 1; % If set to 1, it will correct p values by number of ROIs

close all

% SVM parameters
numVoxel = 125;

% - svmtrain
%     - `-t` : Kernel type (0 => linear)
%     - `-c` : Cost (parameter C of C-SVC)
%     - `-r` : Coefficient in kernel function
%     - `-d` : Degree in Kernel function
%     - `-b` : Tain SVC model for probaility estimates (1 => true)
%     - `-q` : Silent mode
% - svmpredict
%     - `-b` : Predict for probaility estimates (1 => true)
%
trainOpt = '-t 0 -c 1 -r 0 -d 1 -b 0 -q';
testOpt = '-b 0';

%% Directories and contrasts


contrast_name = 'SgtC';
roi2use = 'dmPFC_L';
voxelDataSaveDir = 'voxelData'; % Directory to save results
voxelDataSaveFile = sprintf('voxelData_%s_%s_noSmooth.mat',contrast_name,roi2use);

ctr2use = 'ctrVoxelData';

rng('default') % For reproducibility

%% Use SVM to classify, using voxeldata from S-C contrast and first session


load(fullfile(voxelDataSaveDir,voxelDataSaveFile))

% Select data for a given group
grIdx = unique([S1.Group]);

pIdx1 = find([S1.Group]==grIdx(2));
cIdx1 = find([S1.Group]==grIdx(1));

if any(strcmp(group,'P'))
    if any(strcmp(group,'C'))
        idx2use = [cIdx1,pIdx1];
    else
        idx2use = pIdx1;     
    end
elseif any(strcmp(group,'C'))
    idx2use = cIdx1;
else
    error('Wrong group.')
end

S1 = S1(idx2use);
voxelData_allVox = voxelData_allVox(idx2use,:);
subjNo_list = subjNo_list(idx2use);

if strcmpi(task,'video')
    sess_list = {S1.Sess_video}';
else
    sess_list = {S1.Sess_audio}';
end

if correctSig
    alpha = 0.05/length(roiLabel); % Threshold for significance, adjusted for number of comparisons
else
    alpha = 0.05; 
end

%%

compareGroups = 1; % If set to one, it will also try to classify group for each session
mR_base = []; % Stores classification result from inter-session
mR_group = []; % Stores classification result from inter-group

for iROI = 1:length(roiLabel)
    roiName = roiLabel{iROI};
    fprintf('Extracting data for %s - ROI:%s...\n',contrast_name,roiName)
    idx = find(contains(roiLabel,roiName));
       
    voxelData = [];
    for i=1:length(idx)
        [dummy subjData] = getVoxelData(voxelData_allVox(:,idx),[S1.Group],sess_list,ctr2use);
        voxelData = [voxelData,dummy];
    end
    
    [~,rmIdx] = find(isnan(voxelData));
    voxelData(:,unique(rmIdx)) = [];
        
    for iSess=1:nSess
        targetIdx1 = find([subjData.session]==iSess);
        for jSess=1:nSess
            % Ignores mirror case comparisons
            if iSess>=jSess
                mR_base(iROI,iSess,jSess).predAccuracy = 0; 
                mR_base(iROI,iSess,jSess).mean = 0; 
                mR_base(iROI,iSess,jSess).sem = 0; 
                mR_base(iROI,iSess,jSess).numClass = 2;
                mR_base(iROI,iSess,jSess).predScore = 0;
                mR_base(iROI,iSess,jSess).ROIsize = size(voxelData,2);
                mR_base(iROI,iSess,jSess).classSizes = [length(targetIdx1) length(targetIdx1) length(targetIdx1)];
                mR_base(iROI,iSess,jSess).h = 0;
                mR_base(iROI,iSess,jSess).p = 1;            
                mR_base(iROI,iSess,jSess).pBinom = 1;
                mR_base(iROI,iSess,jSess).hBinom = 0; 
                mR_base(iROI,iSess,jSess).TPR = 0;
                mR_base(iROI,iSess,jSess).TNR = 0; 
                continue
            end
            targetIdx2 = find([subjData.session]==jSess);
            targetIdx = [ones(length(targetIdx1),1);zeros(length(targetIdx2),1)];
            vD = [voxelData(targetIdx1,:);voxelData(targetIdx2,:)];
            
            nFolds = length(targetIdx);

            % Run svm
            result = run_svm(vD,targetIdx,nFolds,numVoxel,trainOpt,testOpt);                   
                        
            mR_base(iROI,iSess,jSess).predAccuracy = [result.cv.fold(:).predAccuracy]; 
            mR_base(iROI,iSess,jSess).mean = mean([result.cv.fold(:).predAccuracy]); 
            mR_base(iROI,iSess,jSess).sem = std([result.cv.fold(:).predAccuracy])/sqrt(nFolds); 
            mR_base(iROI,iSess,jSess).numClass = length(unique(targetIdx));
            mR_base(iROI,iSess,jSess).predScore = result.cv.predScore;
            mR_base(iROI,iSess,jSess).ROIsize = size(voxelData,2);
            mR_base(iROI,iSess,jSess).classSizes = [length(find(targetIdx==-1)) length(find(targetIdx==0)) length(find(targetIdx==1))];            
            mR_base(iROI,iSess,jSess).TPR = result.cv.predScore(2,2)/sum(result.cv.predScore(:,2)); 
            mR_base(iROI,iSess,jSess).TNR = result.cv.predScore(1,1)/sum(result.cv.predScore(:,1)); 

            [mR_base(iROI,iSess,jSess).h mR_base(iROI,iSess,jSess).p mR_base(iROI,iSess,jSess).ci mR_base(iROI,iSess,jSess).stats] = ttest([result.cv.fold(:).predAccuracy],100/length(unique(targetIdx)),'Alpha',alpha);
            mR_base(iROI,iSess,jSess).pBinom = myBinomTest(length(find(mR_base(iROI,iSess,jSess).predAccuracy)),length(mR_base(iROI,iSess,jSess).predAccuracy),0.5,'one');
            if mR_base(iROI,iSess,jSess).pBinom <= alpha & mR_base(iROI,iSess,jSess).mean > 50
                mR_base(iROI,iSess,jSess).hBinom = 1; 
            else
                mR_base(iROI,iSess,jSess).hBinom = 0; 
            end
            
            fprintf('ROI: %s - Session %i vs %i - Mean prediction accuracy = %.3f - CI=%.3f (p binom = %.5f) - TNR=%.3f - TPR=%.3f\n',...
                    roiName,iSess,jSess,mean([result.cv.fold(:).predAccuracy]),mR_base(iROI,iSess,jSess).ci(2)-mR_base(iROI,iSess,jSess).mean,...
                    mR_base(iROI,iSess,jSess).pBinom,mR_base(iROI,iSess,jSess).TNR,mR_base(iROI,iSess,jSess).TPR);
            clear result            
            
        end
        
        
        
        if compareGroups && length(group)>1
            grIdx = unique([subjData.group]);
            targetIdx1 = find([subjData.group]==grIdx(1)&[subjData.session]==iSess);
            targetIdx2 = find([subjData.group]==grIdx(2)&[subjData.session]==iSess);
            
            targetIdx = [ones(length(targetIdx1),1);zeros(length(targetIdx2),1)];
            vD = [voxelData(targetIdx1,:);voxelData(targetIdx2,:)];
            
            nFolds = length(targetIdx);

            result = run_svm(vD,targetIdx,nFolds,numVoxel,trainOpt,testOpt);                                  
            
            mR_group(iROI,iSess).predAccuracy = [result.cv.fold(:).predAccuracy]; 
            mR_group(iROI,iSess).mean = mean([result.cv.fold(:).predAccuracy]); 
            mR_group(iROI,iSess).sem = std([result.cv.fold(:).predAccuracy])/sqrt(nFolds); 
            mR_group(iROI,iSess).numClass = length(unique(targetIdx));
            mR_group(iROI,iSess).predScore = result.cv.predScore;
            mR_group(iROI,iSess).ROIsize = size(voxelData,2);
            mR_group(iROI,iSess).classSizes = [length(find(targetIdx==-1)) length(find(targetIdx==0)) length(find(targetIdx==1))];
            mR_group(iROI,iSess).TPR = result.cv.predScore(2,2)/sum(result.cv.predScore(:,2)); 
            mR_group(iROI,iSess).TNR = result.cv.predScore(1,1)/sum(result.cv.predScore(:,1)); 

            [mR_group(iROI,iSess).h mR_group(iROI,iSess).p mR_group(iROI,iSess).ci mR_group(iROI,iSess).stats] = ttest([result.cv.fold(:).predAccuracy],100/length(unique(targetIdx)),'Alpha',alpha);
            mR_group(iROI,iSess).pBinom = myBinomTest(length(find(mR_group(iROI,iSess).predAccuracy)),length(mR_group(iROI,iSess).predAccuracy),0.5,'one');
            if mR_group(iROI,iSess).pBinom <= alpha & mR_group(iROI,iSess).mean > 50
                mR_group(iROI,iSess).hBinom = 1; 
            else
                mR_group(iROI,iSess).hBinom = 0; 
            end
            
            fprintf('ROI: %s - Group Comparison - Session %i - Mean prediction accuracy = %.3f - CI=%.3f (p binom = %.5f) - TNR=%.3f - TPR=%.3f\n',...
                    roiName,iSess,mean([result.cv.fold(:).predAccuracy]),mR_group(iROI,iSess).ci(2)-mR_group(iROI,iSess).mean,...
                    mR_group(iROI,iSess).pBinom,mR_group(iROI,iSess).TNR,mR_group(iROI,iSess).TPR)  
            clear result
            
        end
    end
    
end


