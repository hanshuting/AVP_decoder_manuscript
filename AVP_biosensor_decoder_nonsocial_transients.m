%%
addpath(genpath('./'));
addpath(genpath('../multiarea_analysis'))

%%
clear; clc
load('mycc.mat');
setup_colors;

warning off;

%% dataset information
pbase = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\Felix';
fname = 'AVP biosensor decoder non-social transients';
fpath = fullfile(pbase, fname);

exp_dir = 'WT mice';
dir_names = {'WT1', 'WT2', 'WT4', 'WT5', 'WT65', 'WT67', 'WT69'};

% exp_dir = 'HE mice';
% dir_names = {'HE66', 'HE68', 'HE70', 'HE71', 'HE72'};

result_name = 'AVP_biosensor';
svm_eval = 'auc';

% set parameters
num_shuff = 1000;
num_rep = 20;
perc_train = 0.7;
perc_test = 1 - perc_train;

nset = length(dir_names);

for dataid = 1:nset

    fprintf('processing dataset %d, %s\n', dataid, dir_names{dataid});
    spath = fullfile(fpath, exp_dir, 'results', dir_names{dataid});
    if ~exist(spath)
        mkdir(spath);
    end
    
    %% load data
    data = cell(1,2);
    
    % load transients
    fname = fullfile(fpath, exp_dir, [dir_names{dataid} '.csv']);
    f_data = readtable(fname);
    data{1} = cat(2, f_data.non_social, f_data.non_social_1);
    data{2} = cat(2, f_data.Social, f_data.Social_1);

    % remove empty values
    data{1} = data{1}(~isnan(data{1}(:,1)), :);
    data{2} = data{2}(~isnan(data{2}(:,1)), :);
    num_data = cellfun(@(x) size(x,1), data);

    label = cell(1,2);
    label{1} = ones(num_data(1),1);
    label{2} = 2*ones(num_data(2),1);
    
    % figure;
    % scatter(data{1}(:,1), data{1}(:,2), 'b*')
    % hold on; scatter(data{2}(:,1), data{2}(:,2), 'ko')
    
    
    %% repeat random splits
    svm_auc = zeros(num_rep, 1);
    svm_auc_shuff = zeros(num_shuff, num_rep);

    % each split takes 70% data for training, 30% for testing
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
    % figure; hold on;
    % plot(svm_auc, 'linewidth', 1.5)
    % v_upper = quantile(svm_auc_shuff, 0.95, 1);
    % v_lower = quantile(svm_auc_shuff, 0.05, 1);
    % v_s = mean(svm_auc_shuff, 1);
    % hold on; plot(v_s, 'color', 0.3*[1 1 1])
    % plot(v_upper, 'color', 0.7*[1 1 1])
    % plot(v_lower, 'color', 0.7*[1 1 1])
    % pval = signrank(v_upper(:), svm_auc(:), 'tail', 'left');
    % title(sprintf('p = %2.4f', pval));
    
   
    %% save
    save(fullfile(spath, [result_name '.mat']), 'svm_auc', 'svm_auc_shuff', 'num_data', '-v7.3');
    
end



