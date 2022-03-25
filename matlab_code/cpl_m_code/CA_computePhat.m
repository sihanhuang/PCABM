function Phat = CA_computePhat(As, K, e, cvt, gammah)
epsilon = 1e-3;
regularize = @(x) (x+epsilon).*(x == 0) + x.*(x ~= 0);

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

end