function gammah = CA_estimgamma(As, K, e, cvt)
% estimate gamma in covariate adjusted SBM
% return: gammah, a p-dim vector
% As: n*n adjacency matrix
% K: number of clusters
% e: an initial estimate of cluster assignment
% cvt: n*n*p covariate

Okl = zeros(K);
for k = 1:K
    for ell = 1:K
        Okl(k,ell) = sum( reshape( As(e==k, e==ell), [], 1));
    end
end

% options = optimoptions('fminunc','SpecifyObjectiveGradient',true);

ler = @(gamma) le_gamma(As,K,e,cvt, gamma, Okl);

p = size(cvt,3);

options = optimoptions('fminunc','Display','none');
gammah = fminunc(ler, zeros(p,1),options);
%default: quasi-newton, bfgs


end

function l = le_gamma(As, K, e, cvt, gammah, Okl)
% % the function of log-likelihood for gamma, given an assignment e
% As = mo.As;
% e = e0;
% gammah = gamma;

n = size(As,1);
p = size(cvt,3);
zTgam = reshape( (reshape(cvt, n*n, p) * gammah), n, n);
exp_zTgam = exp(zTgam);

Ekl = ones(K);
for k = 1:K
    for ell = 1:K
        Ekl(k,ell) = sum( reshape( exp_zTgam(e==k, e==ell), [], 1));
    end
end
if min(Ekl(:)) == 0
    fprintf('Error: in estimgamma: Ekl == 0\n')
end

l = -sum(reshape(As .* zTgam, [], 1)) + sum(reshape( Okl .* log(Ekl), [], 1));

end