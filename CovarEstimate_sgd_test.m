addpath(genpath('../KLTpicker/aspire-master'))

L = 15;
r = 2;
L_half = (L-1)/2;
x = (-L_half:L_half)/L_half; y = x; z = x;
voxel1 = double(((x-0.6).^2 + (y.^2)' + reshape(z,1,1,[]).^2) <= 0.25);
%voxel2 = double((3*x.^2 + ((y).^2)' + reshape(z,1,1,[]).^2) <= 0.5);
voxel2 = double(((x+0.6).^2 + ((y).^2)' + reshape(z,1,1,[]).^2) <= 0.25);
%voxel2 = voxel1;
%voxel2 = -voxel1;



if(~isfile('projections.mat'))
   projection_num = 5000;
   rots1 = rand_rots(projection_num);
   projs1 = cryo_project(voxel1,rots1,L); projs1 = transposeTensor(projs1);
   projs1 = projs1.*(randi(2,1,1,projection_num)-1.5)*2;
   rots2 = rand_rots(projection_num); 
   projs2 = cryo_project(voxel2,rots2,L).*(randi(2,1,1,projection_num)-1.5)*2; projs2 = transposeTensor(projs2);
   save('projections.mat','projection_num','rots1','projs1','rots2','projs2');
else
    load('projections.mat')
end


projs = cat(3,projs1,projs2);
rots = cat(3,rots1,rots2);
%meu = (voxel1+voxel2)/2;
meu = zeros(size(voxel1));
%%
learn_rate = 0.5;
learn_rate_exp = 0.998;
momentum = 0.9;
vel = [];
batch_size = 16;
n = size(projs,3);


numIter = 5000;
verbose_freq = 50;
cost_func_val = zeros(numIter/verbose_freq,1);
singular_vals = zeros(numIter/verbose_freq,r);

u_0 = randn(L,L,L,r); u_0(:,:,:,1) = voxel1;
P_u0 = zeros(L,L,r,batch_size);
PtP_u0 = zeros(L^3,r,batch_size);


reg_param = 0;
for i = 1:numIter
    s = randi(n,1,batch_size);
    proj_s = projs(:,:,s);
    rot_s = rots(:,:,s);
    rot_s_inv = transposeTensor(rot_s);
    
    for j = 1:r
        P_u0(:,:,j,:) = transposeTensor(cryo_project(u_0(:,:,:,j),rot_s));
        PtP_u0(:,j,:) = reshape(im_backproject_arr(squeeze(P_u0(:,:,j,:)),rot_s_inv),L^3,batch_size);
    end
    PtPui_uj = pagemtimes(PtP_u0,'transpose',reshape(u_0,L^3,r),'none'); %size r x r x batch_size, the element i,j,l contains inner product between projected-backprojected u_i and (non projected-backprojected) u_j for the l-th image in the batch    
    PtP_Sigma_PtP_ui = reshape(pagemtimes(PtP_u0,PtPui_uj),L,L,L,r,batch_size); 

    P_meu = transposeTensor(cryo_project(meu,rot_s));
    Pt_y = reshape(im_backproject_arr((proj_s-P_meu),rot_s_inv),L,L,L,1,batch_size);
    Pt_y_yt_P_u =sum(Pt_y.*u_0,[1,2,3]).*Pt_y; 

    grad_u0 = 4*(sum(PtP_Sigma_PtP_ui,5) - sum(Pt_y_yt_P_u,5))/(L^3 * batch_size);
    grad_u0 = grad_u0./norm(grad_u0(:)); grad_u0(:,:,:,1) = 0;
    if(any(isnan(grad_u0(:))))
        break
    end

    [u_0,vel] = sgdmupdate(u_0,grad_u0,vel,learn_rate,momentum);
    learn_rate = (learn_rate_exp) * learn_rate;
   
    %min(norm(u_0(:) - voxel1(:)),norm(u_0(:) + voxel1(:)))/norm(voxel1(:))
    v(i) = min(norm(reshape(u_0(:,:,:,2),[],1) - voxel2(:)),norm(reshape(u_0(:,:,:,2),[],1) + voxel2(:)))/norm(voxel2(:));
    v(i)
    if(mod(i,verbose_freq) == 0)
        cost_func_val(i/verbose_freq) = covar_cost_func(proj_s-P_meu,P_u0);
        singular_vals(i/verbose_freq,:) = cosineSimilarity(cat(4,voxel1,voxel2),u_0);
        display(['Cost function val ' sprintf('%0.5e',cost_func_val(i/verbose_freq))]);
        display(['Cosine Similiraity singular values ' sprintf('%f , ',singular_vals(i/verbose_freq,:))]);


        subplot(2,2,1)
        voxelSurf((abs(u_0(:,:,:,1)) > 0.5).*u_0(:,:,:,1));
        subplot(2,2,2)
        voxelSurf((abs(u_0(:,:,:,2)) > 0.5).*u_0(:,:,:,2));
        subplot(2,2,3)
        plot(v((max(i-400,1)):i))
        drawnow
    end
end

%norm_diff = min(norm(u_0(:) - voxel1(:)),norm(u_0(:) + voxel1(:)))/norm(voxel1(:));
%display(['Norm Difference from ground turth: ' sprintf('%0.5e',norm_diff)]);



voxelSurf((abs(u_0(:,:,:,1)) > 0.5).*u_0(:,:,:,1))



%save('result3.mat','u_0','cost_func_val','singular_vals','numIter','batch_size','learn_rate','momentum','vel')

