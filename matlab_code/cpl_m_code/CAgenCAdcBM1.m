function As = CAgenCAdcBM1(clabels, Pmat, theta, cvt, gamma)
  % generate a covariate adjusted degree-corrected POISSON block model
  % theta: length = n vector
  % cvt:  covariate Zij, n*n*p, should be symmetric
  % gamma:  p dim vec
  
%     clabels = mo.c;
%     Pmat = mo.P;
%     n = length(clabels);
%     p = 3;
%     cvt = zeros(n,n,p);
%     gamma = [0;1;2];
%     theta = mo.theta;
%     tic
    
  K = size(Pmat, 1);
  csizes = zeros(1, K);  
  n = length(clabels);
  p = length(gamma);
  for i=1:K
	csizes(i) = sum(clabels == i);
  end
  cumcsizes = [0, cumsum(csizes)];
  
  
  Amean = zeros(n,n);
  for i = 1:K
      for j = 1:K
%           if j > i
            Amean( (cumcsizes(i)+1) : cumcsizes(i+1) , (cumcsizes(j)+1) : cumcsizes(j+1) )...
                = Pmat(i,j) * ones(csizes(i), csizes(j));
%           end
%           if j == i
%              Amean( (cumcsizes(i)+1) : cumcsizes(i+1) , (cumcsizes(j)+1) : cumcsizes(j+1) )...
%                 = Pmat(i,j) *  triu( ones(csizes(i), csizes(j)), 1 );
%           end
      end
  end
  expcvt =exp( reshape( (reshape(cvt, n*n, p) * gamma), n, n)); %n*n matrix of exp(Zij^T gammah)
  [~, Ic] = sort(clabels);
  Asmean(Ic,Ic) = Amean;
  Asmean = Asmean .* expcvt .* (theta * theta') ;
  
%   if max(max(Asmean)) >= 1
%       fprintf('the edge probability is larger than 1 !')
%       Asmean = min(Asmean,1);
%   end
  
  A1 = poissrnd( triu(Asmean,1), n,n );
  As = sparse(A1);
  
  As = As + As';
  
% fprintf('%3.5fs\n',toc)
end