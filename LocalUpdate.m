%% Jing Ma
% Permutation-based SGD for mimic-iii data
function Lmatrix= LocalUpdate(indices, tao, Gmatrix, Lmatrix, rho2, rho3, eta_p1, eta_p2, eta_p3, cutoffs, Pdim, Ddim, Ltensor, rank, add_dp, lambda, l21norm)
%% main loop
for iter = 1: tao
    % randomly pick non-zero values
    indices = indices(randperm(size(indices, 1)), :);
    
    for p = 1:length(indices)
        u = indices(p,1);
        v = indices(p,2);
        w = indices(p,3);
        R = double(indices(p,4));
        
        % update A{1}(i,:) which is a_i
        gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:));
        Lmatrix{1}(u,:) = Lmatrix{1}(u,:) - eta_p1*gradient;
        
        % update A{2}(j,:) which is b_j
        %gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{3}(w,:)) + (rho2) * (Lmatrix{2}(v,:)-Gmatrix{2}(v,:));
        gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{3}(w,:));
        Lmatrix{2}(v,:) = Lmatrix{2}(v,:) - eta_p2*gradient;
        
        % update A{3}(k,:) which is c_k
        %gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{2}(v,:)) + (rho3) * (Lmatrix{3}(w,:)-Gmatrix{3}(w,:));
        gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{2}(v,:));
        Lmatrix{3}(w,:) = Lmatrix{3}(w,:) - eta_p3*gradient;
        
    end
        
    %% L21 Norm
    if strcmp(l21norm, 'on')
        % mode 1 factor matrix
        for r =1:rank
            Lmatrix{1}(:,r) = Lmatrix{1}(:,r) * max(0,(1-lambda/(norm(Lmatrix{1}(:,r)))));
        end
    end    
end

if strcmp(add_dp, 'on')
    %% Calculate L2 and L3 based on factor matrix
    Matrix1 = zeros(size(Lmatrix{1}));
    Matrix1(cutoffs,:) = Lmatrix{1}(cutoffs,:);
    Matrix2 = zeros(size(Lmatrix{2}));
    Matrix2(Pdim,:) = Lmatrix{2}(Pdim,:);
    Matrix3 = zeros(size(Lmatrix{3}));
    Matrix3(Ddim,:) = Lmatrix{3}(Ddim,:);
    
    tmpTensor = cell(1,3);
    tmpTensor{1} = Matrix1;
    tmpTensor{2} = Matrix2;
    tmpTensor{3} = Matrix3;
    
    % calculate factor matrix's gradient norm
    gradient = cell(1,2);
    for n=2:3
        piitpii=ones(rank,rank);
        for nn=[1:n-1, n+1:3]
            piitpii=piitpii .*(tmpTensor{nn}' * tmpTensor{nn});% compute \Pi^t\Pi
        end
        term1 = mttkrp(Ltensor, tmpTensor,n);
        term2 = tmpTensor{n} * piitpii;
        gradient{n-1} = -term1+term2;
    end
    l2 = norm(gradient{1});
    l3 = norm(gradient{2});
    
    
    %% Add Gaussian noise to achieve (epsilon, delta)-differential privacy
    % calculate l2_sensitivity for each factor matrix
    l2_sensitivity_L2 = 2*tao*l2*eta_p2;  % l2 sensitivity for mode 2 factor matrix
    l2_sensitivity_L3 = 2*tao*l3*eta_p3;  % l2 sensitivity for mode 3 factor matrix
    
    % Assign Privacy Budget for this hospital
    %epsilon_P = 0.35;
    %epsilon_D = 0.35;
    epsilon_P = 0.035;
    epsilon_D = 0.035;
    delta_P = 0.00001;
    delta_D = 0.00001;
    
    % add noise to each row of mode2 matrix(P)
    c_P = sqrt(2*log(1.25/delta_P));
    sigma_P = c_P*l2_sensitivity_L2/epsilon_P;   % calibrate noise
    var_P = sigma_P^2;
    noise_matrix2 = zeros(size(Lmatrix{2}));
    noise_matrix2(Pdim,:) = var_P*randn(length(Pdim),rank);
    Lmatrix{2} = Lmatrix{2} + noise_matrix2;
    
    % add noise to each row of mode3 matrix(D)
    c_D = sqrt(2*log(1.25/delta_D));
    sigma_D = c_D*l2_sensitivity_L3/epsilon_D;  % calibrate noise
    var_D = sigma_D^2;
    noise_matrix3 = zeros(size(Lmatrix{3}));
    noise_matrix3(Ddim,:) = var_D*randn(length(Ddim),rank);
    Lmatrix{3} = Lmatrix{3} + noise_matrix3;
    
end

end
