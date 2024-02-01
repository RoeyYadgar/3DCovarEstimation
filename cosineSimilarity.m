function s = cosineSimilarity(v,u)

    %Reshape v and u into a matrix - assuming the last dimension is the
    %number of vectors in v and u
    v = reshape(v,[],size(v,ndims(v)));
    u = reshape(u,[],size(u,ndims(u)));

    %normalize vector
    v = v./vecnorm(v,2,1);
    u = u./vecnorm(u,2,1);

    cosine_matrix = v'*u;
    [~,S,~] = svd(cosine_matrix);

    %thetas = acos(diag(S));
    s = diag(S);

end
