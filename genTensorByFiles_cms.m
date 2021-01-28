%% Author: Jing Ma, Emory University
function [X, Xs, dim, nonzero_ratio, K, cutoffs, Pcutoffs, Dcutoffs, Xcutoff] = genTensorByFiles_cms()
% X: observered big tensor
% Xs: cell of observered small tensor (each tensor is hold by one hospital)
% dim: dimension of observered big tensor
% nonzero_ratio: Non zero elements ratio of observered big tensor
% K: number of hopsital (number of small tensor)
% cutoffs: divide number of the first dimension of big tensor (patient mode)
%% load data
max_count = 50; % threshold of count (elements' value of tensor)
K = 5;  % number of hospitals

% Read MIMICIII dataset
fileName=strcat('cms_data/cms3order.csv');
count_mat=csvread(fileName,0,0); % count, patient, medi, diag
count_mat=uint32(count_mat);
% count_mat((count_mat(:,1)>max_count),1)=max_count;
sz=max(count_mat); % size of the tensor
numP=sz(2); % number of patients
dim = sz(2:end);
sum_tensors = cell(1,K);
%% compute cutoffs
tsicu=strcat('cms_data/cms3order-1.csv');
count_mat_tsicu=csvread(tsicu,0,0); % count, patient, medi, diag
count_mat_tsicu=uint16(count_mat_tsicu);
count_mat_tsicu((count_mat_tsicu(:,1)>max_count),1)=max_count;
sum_tensors{1} = count_mat_tsicu;

sicu=strcat('cms_data/cms3order-2.csv');
count_mat_sicu=csvread(sicu,0,0); % count, patient, medi, diag
count_mat_sicu=uint16(count_mat_sicu);
count_mat_sicu((count_mat_sicu(:,1)>max_count),1)=max_count;
sum_tensors{2} = count_mat_sicu;

micu=strcat('cms_data/cms3order-3.csv');
count_mat_micu=csvread(micu,0,0); % count, patient, medi, diag
count_mat_micu=uint16(count_mat_micu);
count_mat_micu((count_mat_micu(:,1)>max_count),1)=max_count;
sum_tensors{3} = count_mat_micu;

csru=strcat('cms_data/cms3order-4.csv');
count_mat_csru=csvread(csru,0,0); % count, patient, medi, diag
count_mat_csru=uint16(count_mat_csru);
count_mat_csru((count_mat_csru(:,1)>max_count),1)=max_count;
sum_tensors{4} = count_mat_csru;

ccu=strcat('cms_data/cms3order-5.csv');
count_mat_ccu=csvread(ccu,0,0); % count, patient, medi, diag
count_mat_ccu=uint16(count_mat_ccu);
count_mat_ccu((count_mat_ccu(:,1)>max_count),1)=max_count;
sum_tensors{5} = count_mat_ccu;

%{
nicu=strcat('Fulldata3/nicu_count.csv');
count_mat_nicu=csvread(nicu,0,0); % count, patient, medi, diag
count_mat_nicu=uint16(count_mat_nicu);
count_mat_nicu((count_mat_nicu(:,1)>max_count),1)=max_count;
sum_tensors{6} = count_mat_nicu;
%}

%% Sparse tensor
% Sparse tensor of full data
X=sptensor(count_mat(:,2:4), count_mat(:, 1), sz(2:end));
nonzero_ratio = size(X.subs,1)/prod(size(X));
Xs = cell(1,K);
for k=1:K
    Xs{k} = sptensor(sum_tensors{k}(:, 2:4), sum_tensors{k}(:,1), sz(2:end));
end

%% cutoff rows for each ICU
cutoffs = cell(1,K);
for k=1:K
    cutoffs{k}=unique(sum_tensors{k}(:,2));
end
%% non-zero values of Procedures and Diagnoses for each ICU
Pcutoffs = cell(1,K);
for k=1:K
    Pcutoffs{k}=unique(sum_tensors{k}(:,3));
end
Dcutoffs = cell(1,K);
for k=1:K
    Dcutoffs{k}=unique(sum_tensors{k}(:,4));
end

%% non-zero values for the big tensor
Xcutoff = cell(1,3);
for n=1:3
    Xcutoff{n} = unique(count_mat(:,n+1));
end