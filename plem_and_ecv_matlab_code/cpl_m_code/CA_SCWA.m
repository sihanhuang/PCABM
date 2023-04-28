function [e, dT] = CA_SCWA(As,K, cvt, gammah, varargin)
% function [e, dT, A1, expcvt] = CA_SCWA(As,K, cvt, gammah, varargin)
% Generate initial labeling: spectral clustering with adjustment
    % spectral clustering on Aij / exp(Zij^T gammahat)
% - As:    sparse adjacency matrix %sparse double 
% - K:     number of communities 
% - gammah: an estimation of gamma, p-dim col vector

options = struct('verbose',false,'perturb',false,'rhoPert',0.25, ...
                 'normU',false, 'score',false,'D12',false,'divcvt',true, ...
                 'Ufirst',false,'SC_r',false);  %default options
                 % D12: multiply the adjacency by $D^{-1/2}$ on left and
                 %      right
                 % SC_r: regularized spectral clustering (as in PCABM paper)
                 % perturbation: spectral clustering with perturbation (as
                 %       in Amini's PLEM paper)
                 % score: use not divcvt, no D^-1/2 adjacency mat
if nargin > 4
    % process options
    optNames = fieldnames(options);  %6*1 cell array
    passedOpt = varargin{end}; % struct('verbose',false,...)
  
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

n = size(As,1);
if options.divcvt
    p = size(gammah,1);
    expcvt =exp( reshape( (reshape(cvt, n*n, p) * gammah), n, n)); %n*n matrix of exp(Zij^T gammah)
    A1 = As ./ expcvt;
else
    A1 = As;
end
avgDeg1 = full(mean(sum(A1,2))); % average degree of A1

        
    if ~options.perturb
        % spectral clustering
        tic
        if options.D12
            G = infHandle(diag(sum(A1,2).^(-0.5)));
        
            L = G*A1*G;
        else
            L = A1;
        end
        fun = @(x) L*x;
        
        % opts.issym = 1;
        if options.verbose
            opts.disp = 2;
        end
%         [U, D] = eigs(L,10,'LM',opts);      
        % [U,~] = eigs(L,K,'lm',opts);
        [U,~] = eigs(L,K,'lm');
        if options.verbose
            kmopts = statset('Display','iter');
        else
            kmopts = statset('Display','off'); 
        end
        
%         if options.normU
%             U = ( U - repmat(mean(U),n,1) ) ./ repmat(var(U),n,1) ;
%         end

        if options.score
            kmIDX = kmeans( U(:,2:K) ./ repmat(U(:,1),1,K-1) ,K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        elseif options.Ufirst
            kmIDX = kmeans(U(:,1:K) , K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        else
            kmIDX = kmeans(U(:,2:K) , K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        end
        
        
        
%         kmIDX_k = zeros(K,1);
%         for smleig = 1:K
%             kmIDX_k(smleig) = sum(kmIDX == smleig);
%         end
%         if min(kmIDX_k) < 10
%             fprintf('one cluster fail in kmeans')
%             kmIDX = ( U(:,2) > 0 ) + 1;
%         end
        
        e = reshape(repmat(kmIDX,1,1)',n,[]);  %?
        dT = toc;
        
    elseif options.SC_r
        tic
        d_vec = mean(A1);
        d_all = mean(d_vec);
        adjvec = 2 * d_all ./ d_vec;
        adjvec(adjvec > 1) = 1;
        adjmat = A1 .* sqrt(adjvec' * adjvec);
        [U,~] = eigs(adjmat,K,'lm');
        if options.verbose
            kmopts = statset('Display','iter');
        else
            kmopts = statset('Display','off'); 
        end
        if options.Ufirst
            kmIDX = kmeans(U(:,1:K) , K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        else
            kmIDX = kmeans(U(:,2:K) , K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        end
         e = reshape(repmat(kmIDX,1,1)',n,[]);  %?
        dT = toc;

    else
        % spectral clusteting with perturbation
        tic
        alpha0 = (options.rhoPert)*avgDeg1;
             
        degh = sparse(sum(A1,2) + alpha0);        
        Gh = infHandle(diag(degh.^(-0.5)));
        
        bh = sparse(Gh*ones(n,1));
        bhn = (alpha0/n)*bh;
        if options.D12
            Lh = Gh*A1*Gh;
        else
            Lh = A1;
        end
        fun = @(x) Lh*x + (bh'*x)*bhn;
        
        opts.issym = 1;
        opts.fail = 'drop';
        opts.p = 25;
        if options.verbose
            opts.disp = 2;
        end
        [U, ~] = eigs(fun,n,K,'LM',opts);
        
        if options.verbose
            kmopts = statset('Display','iter'); 
        else
            kmopts = statset('Display','off'); 
        end
        kmIDX = kmeans(U(:,2:K),K,'replicates',10, ...
            'onlinephase','off','Options',kmopts);
        
        e = reshape(repmat(kmIDX,1,1)',n,[]);
          
        dT = toc;    
    end
        
     

end 


function y = infHandle(x)

y = x;
y(isinf(x)) = 0;
end
