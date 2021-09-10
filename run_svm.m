% Runs svm classification

% Inputs
% vD                    Voxel data (num subjects x num voxels)
% targetIdx             Labels for each subject (num subjects x 1)
% nFolds                Number of folds for crossvalidation
% numVoxel              Number of voxels to use for classification. If empty, it
%                       will use all the voxels.  
% featSelectMethod      Method to use to select features.
%                       0: ANOVA F values.
%                       1: Principal feature analysis.
%                       2: Recursive feature elimination using ftSel_SVMRFECBR.m

function result = run_svm(vD,targetIdx,nFolds,numVoxel,trainOpt,testOpt,subjList,useBootstrap)

    if ~exist('trainOpt','var')
        trainOpt = '-t 0 -c 1 -r 0 -d 3 -b 0 -q';
    end
    
    if ~exist('testOpt','var')
        testOpt = '-b 0';
    end
    
    if ~exist('subjList','var')
        subjList = [];
    end       
    
    if ~exist('useBootstrap','var')
        useBootstrap = 0;
    end    

    if ~exist('nFolds','var')
        nFolds = length(targetIdx);
    end
    
    if ~exist('numVoxel','var')
        numVoxel = 0;
    end
    
    nBoot = 100;
    result.cv.predScore = zeros(2,2); % Initializes confusion matrix for this fold
    
    nIdx = floor(size(vD,1)/nFolds);   
    crossIdx = zeros(nFolds,size(vD,1));
    if ~isempty(subjList)    
        sList = unique(subjList);         
        for i=1:length(sList)
            crossIdx1(i,:) = contains(subjList,sList{i});
        end
        if nFolds>=size(crossIdx1,1)
            nFolds = size(crossIdx1,1);            
            crossIdx = crossIdx1;
        else
            for i=1:nFolds
                rIdx = randperm(size(crossIdx1,1),nIdx);
                crossIdx(i,:) = logical(sum(crossIdx1(rIdx,:)));
            end
        end
    else
        if ~mod(size(vD,1),nFolds) % If the subject number is evenly divislbe by the number of folds, then it is simpler
            rIdx = randperm(size(vD,1));
            rIdx = reshape(rIdx,[],nIdx);
            for i=1:nFolds            
                crossIdx(i,rIdx(i,:)) = 1;
            end
        else
            extraIdx = mod(size(vD,1),nFolds);
            rIdx = randperm(size(vD,1));            
            rIdx1 = reshape(rIdx(1:end-extraIdx),[],nIdx);
            rIdx2 = rIdx(end-extraIdx+1:end);
            for i=1:nFolds            
                crossIdx(i,rIdx1(i)) = 1;
            end
            crossIdx(end+1,rIdx2) = 1;
            nFolds = nFolds+1;
        end
    end
    
    crossIdx = logical(crossIdx);
    result.cv.crossIdx = crossIdx;
    result.cv.nFolds = nFolds;
    k = 0; 
    for n = 1:nFolds
           
        trainInd = ~crossIdx(n,:);        
        testInd = crossIdx(n,:);

        xTrain = vD(trainInd, :);
        xTest  = vD(testInd, :);
        tTrain = targetIdx(trainInd, :);
        tTest  = targetIdx(testInd, :);
        
        if numVoxel
            fVals = anova_fvals(zscore(xTrain), tTrain); % Requires BrainDecoderToolbox 2
            [xTrain selectIdx] = select_top(xTrain, fVals, numVoxel); % Requires BrainDecoderToolbox 2
            xTest = select_top(xTest, fVals, numVoxel); % Requires BrainDecoderToolbox 2            
        else
            selectIdx = 1:numVoxel;
        end
        
        result.cv.fold(n).selectIdx = selectIdx;                
                
        trainResult = svmtrain(tTrain, xTrain, trainOpt); % Requires libsvm
        [~,predLabel, predAccuracy,dummy]=evalc('svmpredict(tTest, xTest, trainResult, testOpt)'); % To avoid verbose output
        result.cv.fold(n).predAccuracy = predAccuracy(1);
        for i=1:length(predLabel)
            result.cv.predScore(tTest(i)+1,predLabel(i)+1) = 1+result.cv.predScore(tTest(i)+1,predLabel(i)+1);
        end
    end


end