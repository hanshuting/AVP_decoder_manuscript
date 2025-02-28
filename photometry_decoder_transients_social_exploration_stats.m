%%
addpath(genpath('./'));
addpath(genpath('../multiarea_analysis'))

%%
clear; clc
load('mycc.mat');
setup_colors;

warning off;

%% dataset information
pbase = 'W:\Helmchen Group\Neurophysiology-Storage-03\Han\data\Felix\Data aligned to behavior\Resident intruder test';
exp_dir = 'Social exploration';
dir_names = {'03197', '03199', '03209', '17836', '17837', '32585'};

result_name = 'social_interaction_svm_auc';

num_rep = 20;
num_shuff = 1000;

nset = length(dir_names);
svm_auc = zeros(nset,num_rep);
svm_auc_shuff = zeros(nset,num_rep,num_shuff);
num_data = zeros(nset,2);

% load data 
for dataid = 1:nset
    spath = fullfile(pbase, exp_dir, 'results', dir_names{dataid});
    ld = load(fullfile(spath, [result_name '.mat']));
    svm_auc(dataid,:) = ld.svm_auc;
    svm_auc_shuff(dataid,:,:) = ld.svm_auc_shuff';
    num_data(dataid,:) = ld.num_data;
end

%% plot
v = cell(1,2); 
v{1} = quantile(nanmean(svm_auc_shuff,2), 0.95, 3);
v{2} = mean(svm_auc,2);

thr = 3;
for n = 1:2;  v{n} = v{n}(all(num_data>=thr, 2));  end

cc = {mycc.gray, mycc.blue};
figure; set(gcf, 'color', 'w'); hold on; w = 0.5;
plot([0.5 2.5], [0.5 0.5], 'k:');
for k = 1:2
    h = boxplot(v{k}, 'width', w, 'position', k, 'color', cc{k});
    setBoxStyle(h, 1);
end
plot([1;2]*ones(1,length(v{1})), cell2mat(v)', 'color', mycc.gray_light);
xlim([0.5 2.5]); ylim([0 1]); box off;
set(gca, 'xtick', 1:2, 'xticklabel', {'Shuffled', 'Decoder'}, 'xticklabelrotation', 45);
ylabel('Decoder AUC')
pval = signrank(v{1}, v{2}, 'tail', 'left');
title(sprintf('p = %1.4f', pval));

