%%
addpath(genpath('./'));
addpath(genpath('../multiarea_analysis'))

%%
clear;
load('mycc.mat');
setup_colors;
 
warning off;

%% dataset information
pbase = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\Felix\Data aligned to behavior\Resident intruder test';

exp_dir = 'Social exploration';
dir_names = {'03197', '03199', '03209', '17836', '17837', '32585'};

% exp_dir = 'AVP biosensor social exploration';
% dir_names = {'B2WT1S3', 'B2WT2S1', 'B2WT5S3', 'WT4S1'};

result_name = 'social_interaction_svm_auc';
svm_eval = 'auc';

nset = length(dir_names);
tw_sec = 6;

for dataid = 1:nset

    fprintf('processing dataset %d, %s\n', dataid, dir_names{dataid});
    spath = fullfile(pbase, exp_dir, 'results', dir_names{dataid});
    if ~exist(spath)
        mkdir(spath);
    end
    
    %% load data
    data = cell(1,2);
    label = cell(1,2);
    num_data = zeros(1,2);
    
    % load trace
    fpath = fullfile(pbase, exp_dir, dir_names{dataid});

    % load behavior time stamp data
    fname = dir(fullfile(fpath, 'peak_AUC_*.csv'));
    f_data = csvread(fullfile(fpath, fname(1).name), 1, 1);
    f_data = f_data(1:end-1, :);  % last row is the avg
    num_data(1) = size(f_data,1);
    data{1} = f_data;
    label{1} = ones(num_data(1),1);

    % extract features for all detected transients as control
    data{2} = extract_transient_features(fpath, tw_sec);
    num_data(2) = size(data{2},1);
    label{2} = 2*ones(num_data(2),1);

    
    %% set parameters
    num_shuff = 1000;
    num_rep = 20;
    perc_train = 0.7;
    perc_test = 1 - perc_train;

    svm_auc = zeros(num_rep, 1);
    svm_auc_shuff = zeros(num_shuff, num_rep);

    %% repeat random splits
    for rep_idx = 1:num_rep
        
        % split trials into cross-validation sets
        idx_train = cell(2,1);
        idx_test = cell(2,1);
        min_num = min(num_data);
        for k = 1:2
            idx_rand = randperm(num_data(k));
            N = round(num_data(k)*perc_train);
            if N==num_data(k); N = N-1; end
            idx_train{k} = idx_rand(1:N);
            idx_test{k} = idx_rand(N+1:num_data(k));
        end

        
        % k-fold training
        x_train = cat(1, data{1}(idx_train{1},:), data{2}(idx_train{2},:));
        y_train = cat(1, label{1}(idx_train{1}), label{2}(idx_train{2}));
        C = [0, sum(y_train==2); sum(y_train==1), 0];
        svm_b = fitclinear(x_train, y_train, 'ClassNames', [1,2], 'cost', C);
        svm_w = [svm_b.Bias; svm_b.Beta];

        % auc
        x_test =  cat(1, data{1}(idx_test{1},:), data{2}(idx_test{2},:));
        y_test = cat(1, label{1}(idx_test{1}), label{2}(idx_test{2}));
        if strcmp(svm_eval, 'auc')
            yhat = x_test*svm_w(2:end) + svm_w(1);
            [~,~,~,svm_auc(rep_idx)] = perfcurve(y_test, yhat, 2);
        elseif strcmp(svm_eval, 'acc')
            yhat = predict(svm_b, x_test);
            svm_auc(rep_idx) = mean(y_test==yhat);
        end

        % shuffled model
        N_test = size(x_test,1);  N_train = size(x_train,1);
        auc_s = zeros(1,num_shuff);
        parfor s = 1:num_shuff

            % fit shuffled model
            y_shuff = y_train;
            y_shuff = y_shuff(randperm(N_train));
            svm_shuff = fitclinear(x_train, y_shuff, 'ClassNames', [1,2], 'cost', C);
            if strcmp(svm_eval, 'auc')
                yhat = x_test * svm_shuff.Beta;
                [~,~,~,auc_s(s)] = perfcurve(y_test, yhat, 2);
            elseif strcmp(svm_eval, 'acc')
                yhat = predict(svm_shuff, x_test);
                auc_s(s) = mean(y_test==yhat);
            end

        end
        svm_auc_shuff(:,rep_idx) = auc_s; 
    end
    
    %% plot
%     figure; hold on;
%     plot(svm_auc)
%     v_upper = quantile(svm_auc_shuff, 0.95, 1);
%     v_lower = quantile(svm_auc_shuff, 0.05, 1);
%     v_s = mean(svm_auc_shuff, 1);
%     hold on; plot(v_s, 'color', 0.3*[1 1 1])
%     plot(v_upper, 'color', 0.7*[1 1 1])
%     plot(v_lower, 'color', 0.7*[1 1 1])
%     pval = signrank(v_upper(:), svm_auc(:), 'tail', 'left');
%     title(sprintf('p = %2.4f', pval));
    
   
    %% save
    save(fullfile(spath, [result_name '.mat']), 'svm_auc', 'svm_auc_shuff', 'num_data', '-v7.3');
    
end



