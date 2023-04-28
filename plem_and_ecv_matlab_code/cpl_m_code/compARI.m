function ARI = compARI(CM)
% normARI Computes the adjusted rand index between two clusters
    % this is permutation-invariant?
% CM   confusion matrix

N = sum(CM(:));
ai = sum(CM,2);
bi = sum(CM,1);
sn = sum(CM(:) .* (CM(:)-1))/2;
sa = sum(ai .* (ai-1) )/2;
sb = sum(bi .* (bi-1) )/2;
ARI = (sn - sa * sb / (N*(N-1)/2) ) /...
        (0.5 * (sa + sb) - sa * sb / (N*(N-1)/2) );

end