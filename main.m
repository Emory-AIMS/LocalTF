%% Jing Ma
% DPFact
clc; close all; clearvars;
addpath(genpath('./tensor_toolbox'));


%% Set hyper-parameters
[X, Xs, dim, nonzero_ratio, K, cutoffs, Pcutoffs, Dcutoffs] = genTensorByFiles_cms; % generate tensor from the input files

%% Server: set algorithm parameters
rank = 50;
maxepoch = 50;
tao = 2; % local update times
rho = 5; % quadratic penalty term
eta_p = 0.01;  % learning rate for local SGD
rmses=zeros(1, maxepoch);  %the rmse of each epoch
communication = 0;
lambda_list = [1,1.8,3.2,1.8,1.5,0.6];
%lambda_list = [2,2,2,2,2];
eta_p1 = eta_p;
eta_p2 = eta_p/10^8;
eta_p3 = eta_p2;


%% Set clients workspace
for k=1:K
    client(k).X=Xs{k}; % observed tensor
end

%% Non zero elements index of Observed Tensor (X)
Kindices = cell(1,K);
for k=1:K
    Kindices{k} = [client(k).X.subs, client(k).X.vals];
end

GmatrixInit = cell(1,3); % Initial Global factor matrix
Gmatrix = cell(1,3);
scale = cell(1,3);

%% Server: initialization (mode 2 and 3)
for n = [2,3,1] % n=1 initialize at institutions
    GmatrixInit{n} = rand(dim(n),rank);
    %         saveGmatrixInit{n} = GmatrixInit{n};
    if n==1
        GmatrixInit{n} = GmatrixInit{n}*scale{2}*scale{3};
    else
        %scale{n} = sum(sum(GmatrixInit{n}));
        scale{n} = 1;
        GmatrixInit{n} = GmatrixInit{n}/scale{n};
    end
end

%% Scale rho, eta_p, Lmatrix, Gmatrix
rho2 = rho*(scale{2}^2);
rho3 = rho*(scale{3}^2);

%%
add_dps = {'on'};
%better prediction with dp, probably because dp = l2 reg, thus jeopardize
%the prediction
add_l21norms = {'on'};
for i1 = 1:length(add_dps)
    for i2 = 1:length(add_l21norms)
        add_dp = add_dps{i1};
        l21norm = add_l21norms{i2};
        Gmatrix{1} = GmatrixInit{1};
        Gmatrix{2} = zeros(dim(2),rank);
        Gmatrix{2}(Pcutoffs{1},:)=GmatrixInit{2}(Pcutoffs{1},:);
        Gmatrix{3} = zeros(dim(3),rank);
        Gmatrix{3}(Dcutoffs{1},:)=GmatrixInit{3}(Dcutoffs{1},:);

        %% Hospitals: compute statistics for X / initialization
        % client= a structure contains client i's information, dim, A, u, normX.
        for k=1:K
            client(k).dim=size(client(k).X); %data
            client(k).Ai=cell(1,3); % initialize 3 factor matrices
            % n = 1
            client(k).Ai{1}=zeros(dim(1),rank); % initialize A_k^(1) first to be zeros
            client(k).Ai{1}(cutoffs{k},:) = Gmatrix{1}(cutoffs{k},:);

            client(k).Ai{2}=Gmatrix{2};
            client(k).Ai{3}=Gmatrix{3};

        end

        fileName = sprintf('mimic%d_dp=%s_l21norm=multi_', rank, add_dp);
        fileID = fopen(strcat(fileName,'.txt'), 'w');

        T = ktensor(Gmatrix);
        normresidual= double(norm(plusKtensor(X, -T)));
        old_rmse=double(normresidual/(sqrt(nnz(X))));
        fprintf(fileID, '0, 0, %g, 0\n', old_rmse);
        
        batch = 3;
        
        %% main loop
        for epoch = 1:maxepoch
            % do local update;
            Ai = cell(1,K);
            tic;
            client_old = client;
            Gmatrix_old = Gmatrix;
            
            %Bmat = Gmatrix{2};
            %Cmat = Gmatrix{3};
            
            
            %cost = whos('Bmat');
            %cost2 = whos('Cmat');
            %disp(cost)
            %disp(cost2)
            
            parfor k=1:K
                indices = Kindices{k};
                lambda = lambda_list(k);
                % send Global matrix to hospitals
                 % send to K hospitals
%                 lambda = lambda_list(k);
            
                client(k).Ai= LocalUpdate(indices, tao, Gmatrix, client(k).Ai, rho2, rho3, eta_p1, eta_p2, eta_p3, cutoffs{k}, Pcutoffs{k}, Dcutoffs{k}, client(k).X, rank, add_dp, lambda, l21norm);
                Ai{k} = client(k).Ai;
            end
            
            if epoch == 1 || epoch == 50 || mod(epoch, batch) == 0
                % communication cost for A
                % sum up to get mode 1 global matrix
                tmp_G1 = zeros(dim(1), rank);
                for k=1:K
                    tmp_G1=tmp_G1+client(k).Ai{1};
                end
                Gmatrix{1}= tmp_G1;
                % update global factor matrix
                for n=2:3
                    tmp_Gn = zeros(dim(n), rank);
                    if n==2
                        for k=1:K
                            diff = rho2 * (client(k).Ai{n}-Gmatrix{n});
                            tmp_Gn=tmp_Gn+diff;
                        end
                        gradient = tmp_Gn;
                        Gmatrix{n} = Gmatrix{n} + eta_p2 * gradient;
                    else
                        for k=1:K
                            diff = rho3 * (client(k).Ai{n}-Gmatrix{n});
                            tmp_Gn=tmp_Gn+diff;
                        end
                        gradient = tmp_Gn;
                        Gmatrix{n} = Gmatrix{n} + eta_p3 * gradient;
                    end
                end
                for k=1:K            
                    client(k).Ai{2}=Gmatrix{2};
                    client(k).Ai{3}=Gmatrix{3};
                end
                Gbytes=whos('Gmatrix');
                communication=communication+Gbytes.bytes*K;
            end
            % compute the result every epoch
            T = ktensor(Gmatrix);
            normresidual= norm(plusKtensor(X, -T));
            rmses(epoch)= normresidual/(sqrt(nnz(X)));
            
            %{
            if (abs(old_rmse-rmses(epoch)) < 10^-3 && (epoch == 1 || epoch == 50 || mod(epoch, batch) == 0))
                disp([epoch, toc, rmses(epoch)])
                fprintf(fileID, '%d, %0.4f, %g, %g\n', epoch, toc, rmses(epoch));
                break;
            else
                disp([epoch, toc, rmses(epoch)])
                old_rmse = rmses(epoch);
                fprintf(fileID, '%d, %0.4f, %g, %g\n', epoch, toc, rmses(epoch));
            end
            %}
            %{
            if (old_rmse-rmses(epoch) < 0)
                %eta_p1 = eta_p1 * 0.5;
                eta_p2 = eta_p2 * 0.1;
                eta_p3 = eta_p3 * 0.1;
                
                client = client_old;
                Gmatrix = Gmatrix_old;
                %l21_norm = {'off'};
                %lambda_list = lambda_list *2;
            end
            %}
            disp([epoch, toc, rmses(epoch)])
            old_rmse = rmses(epoch);
            fprintf(fileID, '%d, %0.4f, %g, %g\n', epoch, toc, rmses(epoch));
            
        end
        fclose(fileID);

    end
end

matrixA = Gmatrix{1};
%T = readtable('Fulldata3/target.csv');
%T = T{:,:};
csvwrite('Amatrix_cms_b2_1.csv',matrixA);