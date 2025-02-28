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

exp_dir = cell(1,2);
dir_names = cell(1,2);

exp_dir{1} = 'WT mice';
dir_names{1} = {'WT1', 'WT2', 'WT4', 'WT5', 'WT65', 'WT67', 'WT69'};

exp_dir{2} = 'HE mice';
dir_names{2} = {'HE66', 'HE68', 'HE70', 'HE71', 'HE72'};

result_name = 'AVP_biosensor';

num_rep = 20;
num_shuff = 1000;

% load data 
svm_auc = cell(1,2);
svm_auc_shuff = cell(1,2);
num_data = cell(1,2);
for n = 1:2
    nset = length(dir_names{n});
    svm_auc{n} = zeros(nset,num_rep);
    svm_auc_shuff{n} = zeros(nset,num_rep,num_shuff);
    num_data{n} = zeros(nset,2);
    for dataid = 1:nset
        spath = fullfile(fpath, exp_dir{n}, 'results', dir_names{n}{dataid});
        ld = load(fullfile(spath, [result_name '.mat']));
        svm_auc{n}(dataid,:) = ld.svm_auc;
        svm_auc_shuff{n}(dataid,:,:) = ld.svm_auc_shuff';
        num_data{n}(dataid,:) = ld.num_data;
    end
end



%% plot
v = cell(1,2); 
vs = cell(1,2); 
for n = 1:2
    vs{n} = quantile(nanmean(svm_auc_shuff{n},2), 0.95, 3);
    v{n} = mean(svm_auc{n},2);
end

cc = {{mycc.gray, mycc.blue}, {mycc.gray, mycc.red}};
figure; set(gcf, 'color', 'w'); hold on; w = 0.5;
plot([0.5 4.5], [0.5 0.5], 'k:');
ym = zeros(1,2);
for n = 1:2
    % shuffled
    h = boxplot(vs{n}, 'width', w, 'position', (n-1)*2+1, 'color', cc{n}{1});
    yl = setBoxStyle(h, 1);
    ym(n) = max(ym(n), yl(2));
    % data
    h = boxplot(v{n}, 'width', w, 'position', (n-1)*2+2, 'color', cc{n}{2});
    yl = setBoxStyle(h, 1);
    ym(n) = max(ym(n), yl(2));
    % connect by lines
    plot(((n-1)*2+[1;2])*ones(1,length(v{n})), cat(2, vs{n}, v{n})', 'color', mycc.gray_light);
end
xlim([0.5 4.5]); ylim([0 1]); box off;
set(gca, 'xtick', 1:4, 'xticklabel', {'Shuffled', 'WT', 'Shuffled', 'HE'});
ylabel('Decoder AUC')
ym = ym + 0.05;

p_wt = signrank(vs{1}, v{1}, 'tail', 'left');
plot([1, 2], ym(1)*[1, 1], 'k');
text(1.5, ym(1)+0.02, sprintf('p = %1.2f', p_wt));

p_he = signrank(vs{2}, v{2}, 'tail', 'left');
plot([3, 4], ym(2)*[1, 1], 'k');
text(3.5, ym(2)+0.02, sprintf('p = %1.2f', p_he));

p_all = ranksum(v{1}, v{2});
plot([2, 4], 0.05+max(ym)*[1, 1], 'k');
text(3, max(ym)+0.1, sprintf('p = %1.2f', p_all));=
