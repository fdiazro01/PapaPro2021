clear, clc

[scriptpath scriptname] = fileparts(mfilename('fullpath'));
% Implement preprocessing from BrainDecoderToolbox

% Parameters


task = 'video';
smoothing = 's4';
% smoothing = 'noSmooth';
TR = 3; 
nRun = 2; % Number of runs per protocol
nSess = 6; % Number of total sessions per subject
niftiFolder = 'Nifti';
moveFile = 'rp_*.txt'; 

contrast_list = {'SvsC','S1vsC1','S2vsC2'};


% Import the data
[S subjNo_list_all ~] = read_subject_data('data.xlsx');
[S1 subjNo_list] = read_subject_list('subjList_allSessions.xlsx');


% Directories and contrasts
workDir = []; % Directory where SPM 1st level analysis data is saved
voxelDataSaveDir = []; % Directory to save the voxel data

stimDir = []; % Directory where the stimuli presentation order information is saved
stimFileName = 'stimOrder_run';

roiBaseDir = []; % Directory with the saved ROI mask files


if strcmp(smoothing,'noSmooth')
    epiFile = 'wraf*';
else
    epiFile = [smoothing 'wraf*'];
end

if ~exist(voxelDataSaveDir,'dir'), mkdir(voxelDataSaveDir), end


save_data = 1;
signal_mode = 'PercentSignalChange';

%% Preload ROI mask data

clear roiData

% Load gray matter mask file
maskFile = fullfile('mean_s4GM_bin_mask_50pc.nii');
M = spm_vol(maskFile);
[m1 m2] = spm_read_vols(M); % m1 contains volume with voxel values, m2 contains coordinates in mm

roiFiles = dir(fullfile(roiBaseDir,'rdmPFC_L')); % Load left dmPFC mask
for i = 1:length(roiFiles)    
    R = spm_vol(fullfile(roiFiles(i).folder,fullfile(roiFiles(i).name)));
    [a b] = spm_read_vols(R); % a contains volume with voxel values, b contains coordinates in mm
    mB1 = m1.*a; % Mask the ROI image with the Gray Matter mask 
    rmIdx = isnan(mB1); % Remove NaN values
    mB1(rmIdx) = [];
    b(:,rmIdx) = [];
    c = find(mB1); % Find the mask voxels      
    mVD = mB1(c); % Extract voxels in the mask with their coordinates
    roiData_all(i).XYZmm = b(:,c); % Find the coordinates in mm of the masked voxels
    roiLabel{i} = roiFiles(i).name(2:end-4);   
    roiData_all(i).XYZ = round([b(1,c)' b(2,c)' b(3,c)' ones(size(b(:,c),2),1)]*(inv(M.mat)'))'; % Transform coordinates in mm to cube 
    roiData_all(i).XYZ(4,:) = [];
end


%% Load data and calculate correlation between two ROIs

for iCon=1:length(contrast_list)
    
    % Contrast to use. 
    contrast2use = contrast_list{iCon};

    if strcmp(contrast2use,'SvsC')
        % Index of exp and control stimulus
        stimExpIdx = [1 3];
        stimConIdx = [2 4];
    elseif strcmp(contrast2use,'S1vsC1')
        stimExpIdx = [1];
        stimConIdx = [2];
    elseif strcmp(contrast2use,'S2vsC2')
        stimExpIdx = [3];
        stimConIdx = [4];
    else
        error('Wrong contrast.\n')
    end
    
    for iSubj = 1:length(subjNo_list)
       
        subjName = subjNo_list{iSubj};     
        stimFiles = dir(fullfile(stimDir,subjName,task,[stimFileName '*.mat']));
        sessNo = S1(iSubj).Sess_video;
        if strcmpi(task,'video')
            sessNo = S1(iSubj).Sess_video;
        elseif strcmpi(task,'audio')
            sessNo = S1(iSubj).Sess_audio;
        end
        
        if length(stimFiles) ~= length(sessNo)*nRun, error('Number of stimulus files does not match'), end
        
        nSess = length(sessNo);
        
        for iROI = 1:length(roiLabel)

            roiName = roiLabel{iROI}; 
            roiData = roiData_all(iROI); 
            
            for iSess=1:length(stimFiles)
                
                fprintf('Extracting data for subj %s - Session %i - ROI:%s...\n',subjName,iSess,roiName)

                % Stimuli presentation order
                stim = load(fullfile(stimDir,subjName,task,[stimFileName num2str(iSess) '.mat']));
                onsets = cell2mat([stim(:).onsets]);        
                expOnsets = onsets(:,stimExpIdx); %#ok<*AGROW>
                conOnsets = onsets(:,stimConIdx);            
                d = stim.durations{1}/TR; % Durations MUST be the same. If not, change here
                expOnsets = expOnsets(:)/TR; % Change units from seconds to slices
                conOnsets = conOnsets(:)/TR; % Change units from seconds to slices
                nStimPerRun = size(expOnsets,2);

                % Movement parameters
                Mfile = dir(fullfile(stimDir,subjName,task,niftiFolder,num2str(iSess),moveFile));
                M = load(fullfile(Mfile.folder,Mfile.name));

                % Brain scan files
                Fsess = dir(fullfile(stimDir,subjName,task,niftiFolder,num2str(iSess),epiFile));
                Fsess = strcat({Fsess.folder},{'\'},{Fsess.name})';
            
                % Extract voxel data
                voxelData = spm_get_data(Fsess, roiData.XYZ);  
                
                % Stimulus indices vectors
                idx = zeros(size(voxelData,1),1);
                expOnsets = repmat(expOnsets,1,d)+repmat(1:d,length(expOnsets),1);
                idx(expOnsets(:)) = 1;
                conOnsets = repmat(conOnsets,1,d)+repmat(1:d,length(conOnsets),1);
                idx(conOnsets(:)) = -1;
                            
                % Shift the data 2 volumes (6s) to account for hemodynamic
                % delay
                [voxelData, ind] = shift_sample(voxelData,'ShiftSize', 2);
                idx = idx(ind);
                M = M(ind,:);

                % Regressout movement parameters, DC removal and linear
                % detrend, reduce outliers and normalize voxel data               
                voxelData = regressout(voxelData,'Regressor',M,'LinearDetrend','on'); 
                voxelData = reduce_outlier(voxelData,'Dimension',1); 
                voxelData = normalize_sample(voxelData,'Mode',signal_mode,'Baseline',idx==0);

                expVoxelData = voxelData(idx==1,:);
                conVoxelData = voxelData(idx==-1,:);
                bslVoxelData = voxelData(idx==0,:);
                
                expVoxelData_all(iSess).data = expVoxelData; 
                conVoxelData_all(iSess).data = conVoxelData;
                bslVoxelData_all(iSess).data = bslVoxelData;

            end  
            
            % Adds together the different run data from the same session
            for i=1:nSess                
                if sessNo(i)
                    dummy1 = [];
                    dummy2 = [];
                    dummy3 = [];
                    for j=1:nRun
                        dummy1 = [dummy1;expVoxelData_all((i-1)*nRun+j).data];
                        dummy2 = [dummy2;conVoxelData_all((i-1)*nRun+j).data];
                        dummy3 = [dummy3;bslVoxelData_all((i-1)*nRun+j).data];
                    end
                    voxelData_allVox(iSubj,iROI).sess(sessNo(i)).expVoxelData = mean(dummy1,1);
                    voxelData_allVox(iSubj,iROI).sess(sessNo(i)).conVoxelData = mean(dummy2,1);
                    voxelData_allVox(iSubj,iROI).sess(sessNo(i)).bslVoxelData = mean(dummy3,1);
                    voxelData_allVox(iSubj,iROI).sess(sessNo(i)).ctrVoxelData = mean(dummy1,1)-mean(dummy2,1);

                    voxelData_all(iSubj,iROI).sess(sessNo(i)).expVoxelData = dummy1;
                    voxelData_all(iSubj,iROI).sess(sessNo(i)).conVoxelData = dummy2;
                    voxelData_all(iSubj,iROI).sess(sessNo(i)).bslVoxelData = dummy3;
                    voxelData_all(iSubj,iROI).sess(sessNo(i)).ctrVoxelData = dummy1-dummy2;                    
                end
            end
            
            voxelData_allVox(iSubj,iROI).Subj = subjName; %#ok<*SAGROW>
            voxelData_allVox(iSubj,iROI).ROI = roiName; %#ok<*SAGROW>

            voxelData_all(iSubj,iROI).Subj = subjName; %#ok<*SAGROW>
            voxelData_all(iSubj,iROI).ROI = roiName; %#ok<*SAGROW>
            
            clear expVoxelData_all conVoxelData_all bslVoxelData_all
            
         end    
                 
    end
    
    if save_data
        scriptname_target = [scriptname '_' datestr(datetime('now'),'yymmddTHHMMSS') '.mat'];
        copyfile(fullfile(scriptpath,[scriptname '.m']),fullfile(voxelDataSaveDir,scriptname_target));
        save(fullfile(voxelDataSaveDir,['ROI_data_' signal_mode '_' contrast2use '_allVolAllVox_' datestr(datetime('now'),'yymmddTHHMMSS') '.mat']),'voxelData_all','roiLabel','subjNo_list','S1','-v7.3')
        save(fullfile(voxelDataSaveDir,['ROI_data_' signal_mode '_' contrast2use '_meanVolAllVox_' datestr(datetime('now'),'yymmddTHHMMSS') '.mat']),'voxelData_allVox','roiLabel','subjNo_list','S1') 
    end

end

