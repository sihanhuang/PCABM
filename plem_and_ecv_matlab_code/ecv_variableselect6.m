addpath(fullfile('.','cpl_m_code'))
delete(gcp('nocreate'))

% rng(2000);

% tic, fprintf('%-40s','Setting up the model ...')
n = 500;
K = 2;    % number of communities
oir = 0.5;    % "O"ut-"I"n-"R"atio 

compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
compErr2 = @(c,e) compARI(compCM(c,e,K));

inWei = [1 1];   % relative weight of in-class probabilities
%   [lowVal, lowProb] = deal(0.2, 0.9); % An example of Degree corrected block model %lowProb = rho
  [lowVal, lowProb] = deal(1,0); % An example of original block model

c_rho = 2;
rho = c_rho*(log(n))^1.5 / n;   %dc+(log n)^1.5/n: totally fail: with pertb/score, 0.45; wt pertb, 0
                          %no dc, (log n)^1.5/n: ari = nmi = 1
p = 2;
c_gam = 2;
gamma = c_gam * [1;0];
r = 0.5;  % correlation between correlated z and c
epsilon_L = 0.02;     % improvement threshold in likelihood when a new covariate is added

fprintf("r=%2.2f, c_rho = %2.2f, c_gam = %2.2f \n",r,c_rho,c_gam)




MuI_overr = zeros(10,5);
ARI_overr = zeros(10,5);
gamhmean_overr = zeros(10,5,2);
gamhvar_overr = zeros(10,5,2);
selectpropor_overr = zeros(10,2);
for r_ind = 1:10
    r = (r_ind-1)/10;
experiment_times = 2;
Shat_lik = zeros(experiment_times,p);
MuI_specresult = zeros(experiment_times, 5);
ARI_specresult = zeros(experiment_times, 5);
gamh_specresult = zeros(experiment_times, 5, 2) * NaN;
%  parpool(10)
%  parfor realization = 1:experiment_times  
for realization = 1:experiment_times
% tic, fprintf('%-40s','Generating data ...')
mo = CAdcBlkMod(n,K, lowVal, lowProb, rho); % create a base model
mo = mo.genP(oir, inWei);  % generate the edge probability matri
mo = mo.CAgen_latent;

% 3 types of covariates: 1. indpt from c; 2. z dpt on c with gamma = 0; 3.
% z dpt on c with gamma != 0.
cvt = zeros(n,n,p);
% %%%% an example of correlated covariates 
%      cvtup = triu( randn(n,n), 1 ) * 0.3;
     cvtup = triu( poissrnd(0.1, n,n), 1 );
     cvt(:,:,1) = cvtup + cvtup';
%      cvtup = triu( randn(n,n), 1 ) * 0.3;
     cvtup = triu( poissrnd(0.1, n,n), 1 );
     cvt(:,:,2) = cvtup + cvtup' + 0.6 * r / (sqrt(1-r^2)) * ( mo.P(mo.c,mo.c)/mo.rhon - 1.5 ) ;

mo = mo.CAgenData(cvt, gamma);        % generate data (Adj. matrix "As" and the labels "c")
mo = mo.removeZeroDeg;  % remove zero degree nodes
nnz = size(mo.As, 1);
% fprintf('%3.5fs\n',toc)

e0 = ones(nnz,1);

%%%%%% ecv for selecting variables
Seled = [];  % set of selected variables
not_Seled = 1:p;  % set of not selected variables
Lik_old = -999999;
Lik_new = -99999;
Lik_history = Lik_new * ones(p,1);
Nrep = 5;    %folds of cv

while(Lik_new - Lik_old > epsilon_L * abs(Lik_old)) 
    Lik_old = Lik_new;
lst = length(Seled)+1;
gammah_sub = zeros(p,lst);
for d = find(not_Seled)
    Seled_temp = Seled;
    Seled_temp( lst ) = d;
    gammah_sub(d,:) = CA_estimgamma(mo.As, 1, e0, mo.cvt(:,:,Seled_temp));
end

% l_notseled = length(not_Seled);    %not_Seled is always p dim
Ldm_lik = zeros(p,Nrep)-inf;
% LKm_lik_scaled = zeros(p,Nrep);
% LKm_se = zeros(p,Nrep);

for m=1:Nrep
    p_subsam = 0.9;
    subOmega = binornd(1,p_subsam,nnz,nnz);
    
   for d = find(not_Seled) 
     Seled_temp = Seled;
     Seled_temp( lst ) = d;
     expcvt =exp( reshape( (reshape(mo.cvt(:,:,Seled_temp) , nnz*nnz, lst)...
         * gammah_sub(d,:)'), nnz, nnz)); %n*n matrix of exp(Zij^T gammah)
     A1 = mo.As ./ expcvt;
    subsam_A1 = A1 .* subOmega;
    subsam_As = mo.As .* subOmega;
    subsam_expcvt = expcvt .* subOmega;
    [U,S,V] = svds(subsam_A1 / p, K);
%     for k = 1:Kmax
        Ahat_d = U(:,1:K) * S(1:K,1:K) * V(:,1:K)';
        opt_cvsc = struct('verbose',false,'perturb',true,...
                    'score',false,'divcvt',false,'D12',false);
        edm =  CA_SCWA(Ahat_d, K, zeros(nnz,nnz,1), 0, opt_cvsc);
            Oll = zeros(K);
            Ell = zeros(K);
            Bll = zeros(K);
            for ell1 = 1:K
                    for ell2 = 1:K
                    Oll(ell1,ell2) = sum( reshape( subsam_As(edm==ell1, edm==ell2), [], 1));
                    Ell(ell1,ell2) = sum( reshape( subsam_expcvt(edm==ell1, edm==ell2), [], 1));
                    Bll(ell1,ell2) = Oll(ell1,ell2) / Ell(ell1,ell2);
                    end
            end
            EA_hat = Bll(edm,edm) .* expcvt;
        Ldm_lik(d,m) = sum(sum( (mo.As-subsam_As) .* log(EA_hat) - EA_hat .* (1-subOmega) ));
%         LKm_lik_scaled(d,m) = sum(sum( (A1-subsam_A1).* log(EA_hat) - Bll(edm,edm) .* (1-subOmega) ));
%         LKm_se(d,m) = sum(sum( ( ( A1-Bll(edm,edm) ) .* (1-subOmega) ).^2 ));
   end      %end for d = not_Seled (k = 1:Kmax)
end         %end for m = 1:Nrep
LK_lik = mean(Ldm_lik,2)/n;
% LK_lik_scaled = mean(LKm_lik_scaled,2)/n;
% LK_se = mean(LKm_se,2)/n;

[Lik_new,dhat_lik] = max(LK_lik);
if Lik_new - Lik_old > epsilon_L * abs(Lik_old)
    Seled(lst) = dhat_lik;
    not_Seled(dhat_lik) = 0;
%     Lik_old = Lik_new;
% else
%     break
Lik_history(dhat_lik) = Lik_new;
end

% [~,dhat_lik_scaled] = max(LK_lik_scaled);
% [~,dhat_se] = min(LK_se);

end  %end for while

Lik_history = Lik_history(Seled);
Shat_lik(realization,:) = [1:p] - not_Seled;

%%%%%%% spectral clustering when adjusting all/oracle/selected/none of the covariates
init_opts = struct('verbose',false,'perturb',true,'D12',false);

    for ad_type = 1:5
%%%% adjusting all
    if ad_type == 1
    gammah_temp = CA_estimgamma(mo.As, 1, e0, mo.cvt);
    [e_temp, ~] = CA_SCWA(mo.As, mo.K, mo.cvt, gammah_temp, init_opts);
    gamh_specresult(realization,ad_type,:) = gammah_temp;
    end
%%%% adjusting oracle
    if ad_type == 2
    gammah_temp = CA_estimgamma(mo.As, 1, e0, mo.cvt(:,:,logical(gamma ~= 0)) );
    [e_temp, ~] = CA_SCWA(mo.As, mo.K, mo.cvt(:,:,logical(gamma ~= 0)), gammah_temp, init_opts);
    gamh_specresult(realization,ad_type,logical(gamma ~= 0)) = gammah_temp;
    end
%%%% adjusting selected
    if ad_type == 3
        if sum(Shat_lik(realization,:)) ~= 0
    gammah_temp = CA_estimgamma(mo.As, 1, e0, mo.cvt(:,:,logical(Shat_lik(realization,:))) );
    [e_temp, ~] = CA_SCWA(mo.As, mo.K, mo.cvt(:,:,logical(Shat_lik(realization,:))), gammah_temp, init_opts);
      gamh_specresult(realization,ad_type,logical(Shat_lik(realization,:))) = gammah_temp;
        else
           [e_temp, ~] = CA_SCWA(mo.As, mo.K, zeros(nnz,nnz,1), 0, init_opts); 
        end
    end
%%%% adjusting none
    if ad_type == 4
    [e_temp, ~] = CA_SCWA(mo.As, mo.K, zeros(nnz,nnz,1), 0, init_opts);    
    end
%%%% adjusting only z'
    if ad_type == 5
    gammah_temp = CA_estimgamma(mo.As, 1, e0, mo.cvt(:,:,logical(gamma == 0)) );
    [e_temp, ~] = CA_SCWA(mo.As, mo.K, mo.cvt(:,:,logical(gamma == 0)), gammah_temp, init_opts);    
    gamh_specresult(realization,ad_type,logical(gamma == 0)) = gammah_temp;
    end
    MuI_specresult(realization,ad_type) = compErr(mo.c, e_temp);
    ARI_specresult(realization,ad_type) = compErr2(mo.c, e_temp);
    end
    
end

% fprintf("variables selected times / total realizations") 
selectpropor_overr(r_ind,:) = sum(logical(Shat_lik),1)/experiment_times;
% fprintf("average MuI/ARI for adjusting all/oracle(gam~=0)/selected/none/corr'ed Z only")
MuI_overr(r_ind,:) = mean(MuI_specresult,1);
ARI_overr(r_ind,:) = mean(ARI_specresult,1);
gamhmean_overr(r_ind,:,:) = nanmean(gamh_specresult,1);
gamhvar_overr(r_ind,:,:) = nanvar(gamh_specresult,1);
%  corr(reshape(mo.P(mo.c,mo.c),nnz*nnz,1),reshape(mo.cvt(:,:,2),nnz*nnz,1))
 
end

% selectpropor_overr
% MuI_overr
% ARI_overr
% gamhmean_overr
% gamhvar_overr