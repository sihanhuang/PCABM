addpath(fullfile('.','cpl_m_code'))
addpath(fullfile('.','polblogs'))
rng(1)

pb_A = csvread('pb_A3.csv',0,0);
pb_opinion = csvread('pb_opin2.csv')+1;

%         d_vec = mean(pb_A);
%         d_all = mean(d_vec);
%         adjvec = 2 * d_all ./ d_vec;
%         adjvec(adjvec > 1) = 1;
%         pb_A = pb_A .* sqrt(adjvec' * adjvec);

n = size(pb_A,1);
K = 2;
p = 1;

compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
compErr2 = @(c,e) compARI(compCM(c,e,K));

zdNodes = sum(pb_A,2) == 0;  % sum(zdNodes) = 0, i.e. no 0 deg nodes

cvt = zeros(n,n,p);
cvtup = triu( log( repmat(sum(pb_A,2),1,n) ) , 1 ) + triu( log( repmat(sum(pb_A,1),n,1) ) , 1 );
 
cvt(:,:,1) = cvtup + cvtup';
% deg_vec = sum(pb_A,2);
% cvt1 = log (deg_vec * deg_vec');
 
 
tic, fprintf('%-40s','Estimating gamma ...')
pi0 = ones(K,1) / K;
[e0,~] = find(mnrnd(1,pi0,n)');
gammah = CA_estimgamma(pb_A, K, e0, cvt);
fprintf('gammahat=%f\n',gammah)
fprintf('%3.5fs\n',toc)
gammah2 = 0.465;

init_opts = struct('verbose',false,'perturb',false,...
                   'score',false,'divcvt',true,'D12',false,'Ufirst',false,'SC_r',false);
%     init_opts = struct('verbose',false,'perturb',false,...
%                    'score',true,'divcvt',false,'D12',false,'Ufirst',false,'SC_r',false);
                             % score: use not divcvt, no D^-1/2 adjacency mat
T = 20;
tic, fprintf('%-40s','Applying init. method (SC) ...') 
[e, init_dT] = CA_SCWA(pb_A, K, cvt, gammah2, init_opts);    
fprintf('%3.5fs\n',toc)
init_nmi = compErr(pb_opinion, e);
init_ari = compErr2(pb_opinion, e);
init_miscl = min( sum(e ~= pb_opinion), sum(e == pb_opinion) );

cpl_opts = struct('verbose',false,'delta_max',0.1, ...
                   'itr_num',T,'em_max',80,'track_err',true);
% using random initialization
tic, fprintf('%-40s','Applying CA plEM (random initialization) ...')
[CA_chat, CA_err, CA_dT, CA_post] = ...
     CA_plEM(pb_A, K, e0, pb_opinion, cvt, gammah , cpl_opts); 
            % why pih NaN ???   % e do not enough 2s

% % % using SCWA initialization
% tic, fprintf('%-40s','Applying CA plEM (SCWA initialization) ...')
% [CA_chat, CA_err, CA_dT, post] = ...
%      CA_plEM(pb_A, K, e, pb_opinion, cvt, gammah , cpl_opts);
 
fprintf('%3.5fs\n',toc)
CAplEM_nmi = compErr(pb_opinion, CA_chat);
CAplEM_ari = compErr2(pb_opinion, CA_chat);
CAplEM_miscl = min( sum(CA_chat ~= pb_opinion), sum(CA_chat == pb_opinion) );
fprintf(1,'Init error = %d\nInit NMI = %2.3f\nInit ARI = %2.3f\nCAplEM error = %d\nCAplEM  NMI = %2.3f\nCAplEM  ARI = %2.3f\n',...
    init_miscl,init_nmi,init_ari,CAplEM_miscl,CAplEM_nmi,CAplEM_ari)
