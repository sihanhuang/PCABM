addpath(fullfile('.','cpl_m_code'))
addpath(fullfile('.','schooldata'))

fileID = fopen('school_Ab2.txt','r'); Ab = textscan(fileID,'%f');
Ab = Ab{1,1}; Ab = reshape(Ab,777,777);
fileID = fopen('school_gender.txt','r'); gender = textscan(fileID,'%f');
gender = gender{1,1};
fileID = fopen('school_grade.txt','r'); grade = textscan(fileID,'%f');
grade = grade{1,1};
fileID = fopen('school_racedm.txt','r'); racedm = textscan(fileID,'%f');
racedm = racedm{1,1}; racedm = reshape(racedm,4,777)'; %other,white,black,hispanic
fileID = fopen('school_sch.txt','r'); sch = textscan(fileID,'%f');
sch = sch{1,1};
fileID = fopen('school_tot.txt','r'); tot = textscan(fileID,'%f');
tot = tot{1,1};

compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.
compErr2 = @(c,e) compARI(compCM(c,e,K));

% allcov = [gender, grade, sch, racedm, tot];
% corrcoef(allcov);

n = size(Ab,1);
% K = 2;

% %%%% predict gender
% c = gender + 1;
% p = 4;
% cvt = zeros(n,n,p); 
%  cvt(:,:,1) = racedm(:,2) * racedm(:,2)';
%  cvt(:,:,2) = racedm(:,3) * racedm(:,3)';
%  cvt(:,:,3) = abs(grade-grade');
%  cvt(:,:,4) = abs(sch-sch');
% %  cvt(:,:,5) = racedm(:,1) * racedm(:,1)';
% %  cvt(:,:,6) = racedm(:,4) * racedm(:,4)';
%  
% % tic, fprintf('%-40s','Estimating gamma ...')
% pi0 = ones(K,1) / K;
% [e0,~] = find(mnrnd(1,pi0,n)');
% % e0 = ones(n,1);
% gammah = CA_estimgamma(Ab, K, e0, cvt);
% 
% init_opts = struct('verbose',false,'perturb',true,...
%                    'score',false,'divcvt',true,'D12',true);
%                              % score: use not divcvt, no D^-1/2 adjacency mat
% tic, fprintf('%-40s','Applying init. method (SC) ...') 
% [e, init_dT] = CA_SCWA(Ab, K, cvt, gammah, init_opts);    
% fprintf('%3.5fs\n',toc)
% init_nmi = compErr(c, e);
% init_ari = compErr2(c, e);
% init_miscl = min( sum(e ~= c), sum(e == c) );





%% predict school
c = sch + 1;
K = 2;
p=6;
cvt = zeros(n,n,p);
cvt(:,:,1) = racedm(:,2) * racedm(:,2)';
cvt(:,:,2) = racedm(:,3) * racedm(:,3)';
cvt(:,:,3) = racedm(:,4) * racedm(:,4)';
cvt(:,:,4) = racedm(:,1) * racedm(:,1)';
cvt(:,:,5) = abs(gender - gender');
cvt(:,:,6) = log(tot+1) + log(tot+1)';

  
% tic, fprintf('%-40s','Estimating gamma ...')
pi0 = ones(K,1) / K;
[e0,~] = find(mnrnd(1,pi0,n)');
% e0 = ones(n,1);
gammah = CA_estimgamma(Ab, K, e0, cvt);


fprintf('%3.5fs\n',toc)

init_opts = struct('verbose',false,'perturb',false,...
                   'score',false,'divcvt',true,'D12',true);
                             % score: use not divcvt, no D^-1/2 adjacency mat
T = 20;
tic, fprintf('%-40s','Applying init. method (SC) ...') 
[e, init_dT] = CA_SCWA(Ab, K, cvt, gammah, init_opts);    
fprintf('%3.5fs\n',toc)
init_nmi = compErr(c, e);
init_ari = compErr2(c, e);
init_miscl = min( sum(e ~= c), sum(e == c) );

cpl_opts = struct('verbose',false,'delta_max',0.1, ...
                   'itr_num',T,'em_max',80,'track_err',true);

% % using random initialization
% tic, fprintf('%-40s','Applying CA plEM (random initialization) ...')
% [CA_chat, CA_err, CA_dT, CA_post] = ...
%      CA_plEM(pb_A, K, e0, pb_opinion, cvt, gammah , cpl_opts); 
%             % why pih NaN ???   % e do not enough 2s

tic, fprintf('%-40s','Applying CA plEM (SCWA initialization) ...')
[CA_chat, CA_err, CA_dT, post] = ...
     CA_plEM(Ab, K, e, c, cvt, gammah , cpl_opts);                                                
fprintf('%3.5fs\n',toc)
CAplEM_nmi = compErr(c, CA_chat);
CAplEM_ari = compErr2(c, CA_chat);
CAplEM_miscl = min( sum(CA_chat ~= c), sum(CA_chat == c) );
fprintf(1,'Init NMI = %3.2f\nInit ARI = %3.2f\nCAplEM  NMI = %2.3f\nCAplEM  ARI = %2.3f\n',...
    init_nmi,init_ari,CAplEM_nmi,CAplEM_ari)

% %% predict race
% [c,~] = find(racedm');
% K=4;
% bw = (c==2 | c==3);
% 
% n = sum(bw);
% c = c(bw)-1;
% grade = grade(bw);
% sch = sch(bw);
% gender = gender(bw);
% tot = tot(bw);
% Ab = Ab(bw,bw);
% K=2;
% 
% 
% 
% p = 4;
% cvt = zeros(n,n,p);
% cvt(:,:,2) = abs(grade-grade');
% cvt(:,:,1) = abs(sch-sch');
% cvt(:,:,3) = abs(gender - gender');
% cvt(:,:,4) = log(tot+1) + log(tot+1)';
% 
%   
% % tic, fprintf('%-40s','Estimating gamma ...')
% pi0 = ones(K,1) / K;
% [e0,~] = find(mnrnd(1,pi0,n)');
% % e0 = ones(n,1);
% gammah = CA_estimgamma(Ab, K, e0, cvt);
% 
% 
% fprintf('%3.5fs\n',toc)
% 
% init_opts = struct('verbose',false,'perturb',false,...
%                    'score',false,'divcvt',true,'D12',true);
%                              % score: use not divcvt, no D^-1/2 adjacency mat
% T = 20;
% tic, fprintf('%-40s','Applying init. method (SC) ...') 
% [e, init_dT] = CA_SCWA(Ab, K, cvt, gammah, init_opts);  
% e_nan = (e~=1&e~=2);
% e(e_nan) = 2;
% fprintf('%3.5fs\n',toc)
% init_nmi = compErr(c, e);
% init_ari = compErr2(c, e);
% init_miscl = min( sum(e ~= c), sum(e == c) );
% 
% cpl_opts = struct('verbose',false,'delta_max',0.1, ...
%                    'itr_num',T,'em_max',80,'track_err',true,'rdm_init_unbalance',false);
% 
% % % using random initialization
% % tic, fprintf('%-40s','Applying CA plEM (random initialization) ...')
% % [CA_chat, CA_err, CA_dT, CA_post] = ...
% %      CA_plEM(Ab, K, e0, c, cvt, gammah , cpl_opts); 
% %             % why pih NaN ???   % e do not enough 2s
% 
% % using SCWA initialization
% tic, fprintf('%-40s','Applying CA plEM (SCWA initialization) ...')
% [CA_chat, CA_err, CA_dT, post] = ...
%      CA_plEM(Ab, K, e, c, cvt, gammah , cpl_opts);                                                
% fprintf('%3.5fs\n',toc)
% 
% CAplEM_nmi = compErr(c, CA_chat);
% CAplEM_ari = compErr2(c, CA_chat);
% CAplEM_miscl = min( sum(CA_chat ~= c), sum(CA_chat == c) );
% fprintf(1,'Init NMI = %3.2f\nInit ARI = %3.2f\nCAplEM  NMI = %2.3f\nCAplEM  ARI = %2.3f\n',...
%     init_nmi,init_ari,CAplEM_nmi,CAplEM_ari)


