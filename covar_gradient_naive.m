function grad = covar_gradient_naive(u,proj_s,rot_s,mu)
    L = size(u,1);
    r = size(u,4);
    n = size(rot_s,3);
    rot_s_inv = transposeTensor(rot_s);

    P_u = zeros(L,L,r,n);
    PtP_u = zeros(L^3,r,n);
    for j = 1:r
        P_u(:,:,j,:) = cryo_project(u(:,:,:,j),rot_s);
        PtP_u(:,j,:) = reshape(im_backproject_arr(squeeze(P_u(:,:,j,:)),rot_s_inv),L^3,n);
    end

    P_mu = cryo_project(mu,rot_s);
    Pt_y = reshape(im_backproject_arr((proj_s-P_mu),rot_s_inv),L,L,L,1,n);


    PtP_u = reshape(PtP_u,L^3,r,n);
    Pt_y = reshape(Pt_y,L^3,n);

    L_sigma = sum(pagemtimes(reshape(PtP_u,L^3,1,r,n),reshape(PtP_u,1,L^3,r,n)),[3,4])/n;
    B = sum(pagemtimes(reshape(Pt_y,L^3,1,n),reshape(Pt_y,1,L^3,n)),3)/n;

    grad = 4*(L_sigma-B)*reshape(u,L^3,r)/(L^3);
    grad = reshape(grad,L,L,L,r);



end