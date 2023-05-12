function [chatF,err,dT,Phat] = CA_plEM(As,K,e,c, cvt,gammah, varargin)
% pseudo likelihood EM algorithm for pairwise covariate adjusted BM
% As  sparse Adjacency matrix
% e   the initial labeling, n dim vec
% c   the true labeling (used only for computing/tracking error across iterations
% T   number of iterations on top of EM

% cvt  covariate Zij, n*n*p, should be symmetric (Zij = Zji)
% gammah  an estimate of gamma in exp(Zij^T gamma), p dim vec

% chatF: cluster estimation by the algorithm
% post: posterior pi_{il}

options = struct('verbose',false,'verb_level',1,'conv_crit','pl', ...
    'em_max',100,'delta_max',0,'itr_num',20,'track_err',true,'rdm_init_unbalance',true); %default options
if nargin > 6
    % process options
    optNames = fieldnames(options);
    passedOpt = varargin{end};
  
    if ~isstruct(passedOpt) 
        error('Last argument should be an options struct.')
    end
    for fi = reshape(fieldnames(passedOpt),1,[])
        if any(strmatch(fi,optNames))
            options.(fi{:}) = passedOpt.(fi{:});
        else
            error('Unrecognized option name: %s', fi{:})
        end
    end
end

if options.track_err
%     compErr = @(c,e) compARI(compCM(c,e,K)); 
     compErr = @(c,e) compMuI(compCM(c,e,K));         % use mutual info as a measure of error/sim.
end

rowSumNZ = @(X) X./repmat(sum(X,2),1,size(X,2)); % normalize row sum to 1
% swap= @(varargin) varargin{nargin:-1:1};         % swap two variables
epsilon = 1e-3;
regularize = @(x) (x+epsilon).*(x == 0) + x.*(x ~= 0);

LOG_REAL_MAX = log(realmax)-1;

T = options.itr_num;

% remove 0-degree nodes
zdNodes = sum(As,2) == 0;
nOrg = size(As,1);
% zdNum = nnz(zdIDX);
% AsOrg = As;
% eOrg = e;

if options.rdm_init_unbalance
for k = 1:K
    if sum(e==k) < 10
        fprintf('the %2.0fth cluster in e is too small\n', k)
        %random initialization if initial e is too unbalanced
        pi0 = ones(K,1) / K;
        [e,~] = find(mnrnd(1,pi0,nOrg)'); 
        break
    end
end
end

As = As(~zdNodes,~zdNodes);
e = e(~zdNodes);
cvt = cvt(~zdNodes,~zdNodes,:);
if options.track_err
    c = c(~zdNodes);  % not a very good idea ! % why?
end
n = size(As,1);
p = size(cvt,3);
expcvt =exp( reshape( (reshape(cvt, n*n, p) * gammah), n, n)); %n*n matrix of exp(Zij^T gammah)

% Compute initial Phat
% Phat_kl is from l in c to k in e
Phat = zeros(K);
for k = 1:K
    for ell = 1:K
        Phat(k,ell) = sum( reshape( As(e==k, e==ell), [], 1)) ...
                        / sum( reshape( expcvt(e==k, e==ell), [], 1)) ;

        Phat(k,ell) = regularize( Phat(k,ell) );
    end
end


% Compute initial Bjkehat %n*K
Bjkehat1 = zeros(n,K);
for k = 1:K
    Bjkehat1(:,k) = expcvt * (e == k) ;
end

% % Compute inital Lambda hat and Theta hat
% Lambdah = n*Phat*Rhat';  
% Thetah = rowSumNZ(Lambdah);  %the multinomial theta, not the degree correction

% Lambda hat becomes n*k*k, but in fact the info is contained in n*k Bjke
% and k*k Phat

% Compute inital community prior estimates
pih = zeros(K,1);
for k = 1:K
    pih(k) = sum( e == k ) / n;
    pih(k) = regularize( pih(k) );
end
%pih = rand(K,1); pih = pih/sum(pih);


% Compute block compressions
Bs = compBlkCmprss(As,e,K);

% initial the err vector
if options.track_err
    err = zeros(1,T+1);
    err(1) = compErr(c,e);
else
    err = 0;  %-1
end
% chat = zeros(n,1);

emN = options.em_max;           % max. EM steps
%deltaMax = 2/n;
if options.delta_max == 0
    switch options.conv_crit
        case 'param'
            deltaMax = 1e-3; % max. error below which EM terminates
        case 'label'
            deltaMax = 1e-2;    
        case 'pl'
            deltaMax = 1e-2;    
    end
else
    deltaMax = options.delta_max;
end
% initial chat
chat = e;
% chatOld = chat;

        % unconditional PL
        if options.verbose
            fprintf(1,'\nupl: %3d iterations\n',T)
        end
        tic
       for t = 2:(T+1)
            
            delVec = zeros(emN,1);
            OVF_FLAG = false;
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            
            % Compute Bjkehat %n*K for chat
            Bjkehat = zeros(n,K);
            for k = 1:K
                Bjkehat(:,k) = expcvt * (chat == k) ;
            end
            
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                % Z is K x n   % Bs is n*K, Bjkehat is n*K
                % Phat_lk is l in c, k in e
%  2023.5.12              Z = -Phat * Bjkehat' + log(Phat)*Bs' + ...
%  2023.5.12                  repmat(sum(log(Bjkehat) .* Bs, 2)', K, 1) ;
                Z = -Phat * Bjkehat' + log(Phat)*Bs';%2023.5.12
                % Z: pi_il = pi_l * exp(Z_li)
                Zmean = mean(Z);   %1*n
                Z = Z - repmat(Zmean,K,1);
                 
                [ZZ, OVF] = handleOverflow(Z,LOG_REAL_MAX); 
                
                U = exp( ZZ ); %K*n
                if OVF
                    OVF_FLAG = true;
                end
                
                alpha = repmat(pih(:),1,n).*U;  %K*n
                
                post_denom = sum(alpha);  %1*n                             
                
%                 alphatemp = alpha;
                
                alpha = alpha ./ regularize( repmat(post_denom,K,1) ); %pi_{li}hat
                
% 2023.5.12               plVal = sum( log(post_denom) ) + sum(Zmean);  %log pseudo likelihood  
        
                % alpha is K x n -- This is posterior prob. of labels pi_li
                % Bs is n x K
                % Bjkehat is n*K
                
                if any(isnan(alpha(:)))
                      warning('Something went wrong, pih will have NaN entries.')
                      % try to random initialize again
                      pi0 = ones(K,1) / K;
                      [e_r,~] = find(mnrnd(1,pi0,nOrg)');
                      [chatF,err,dT,post] = CA_plEM(As,K,e_r,c, cvt,gammah, options);
                      return
                 end
               
%                 pihold = pih;

                pih = mean(alpha,2); 
%                 disp(pih)               
                 if any(isnan(pih))
                      warning('Something went wrong, pih has NaN entries.')
                      % try to random initialize again
                      pi0 = ones(K,1) / K;
                      [e_r,~] = find(mnrnd(1,pi0,nOrg)');
                      [chatF,err,dT,post] = CA_plEM(As,K,e_r,c, cvt,gammah, options);
                      return
                 end
                  
            % Phat_lk: l in c, k in e
                 %Phat =  ( Bs' * alpha' ) ./ ( Bjkehat' * alpha' );%wrong!
                 
                 Phat = (alpha * Bs) ./ (alpha * Bjkehat);
                 
                 plVal = sum(sum(alpha .* (repmat(pih(:),1,n)-Phat * Bjkehat' + log(Phat)*Bs')));  % 2023.5.12
                 
%                 Lambdah = regularize( ...
%                         diag(1./ regularize(pih) )*(alpha*Bs/n) );
                

%                  [~, chat] = max(alpha',[],2);
                [~,chat] = max(alpha,[],1);
                chat = chat(:);
                
%                 if options.verb_level > 1
%                     disp(pih)
%                     disp(Lambdah)
%                 end
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;
                
                if nu ~= 1
                    delta = abs((plVal - plValOld)/plValOld);
                    CONVERGED = delta < deltaMax;
                    delVec(nu-1) = delta;
                end
                plValOld = plVal;    %check convergence of pseudo likelihood
                

                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;
            end % end while
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'b.-')
%                 
%                 if OVF_FLAG
%                     title('OVERFLOW')
%                 end
%                 pause(1)
%             end
            
            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs = compBlkCmprss(As,chat,K);
                       
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end
        end  %end for
        
       % err
        
%         if options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = pih;
        zdPrior = zdPrior/sum(zdPrior);
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = alpha;
        post(:,zdNodes) = tempP;
         
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end
        
end

function Cs = compBlkCmprss(As,e,K)
% Compute block compression
%
% As  sparse adj. matrix
% e   labeling to use
% K   number of communities

n = size(As,1);

Cs = spalloc(n,K,nnz(As));
for k = 1:K
   Cs(:,k) = sum(As(:,e == k),2);
end

end


function [ZZ,OVF] = handleOverflow(Z,LOG_MAX)
    Zmax = max(abs(Z(:)));
   
    if Zmax > LOG_MAX
        ZZ = (LOG_MAX / Zmax) * Z;
        OVF = true;
    else
        ZZ = Z;
        OVF = false;
    end
end

function print_pl_decay(nu,delta,CONVERGED)
    if nu == 1
        fprintf(1,'    .... ')
    else
        fprintf(1,'%3.5f ',delta)
        if (mod(nu,5) == 0) || CONVERGED 
            fprintf(1,'\n    .... ');
        end
    end
end