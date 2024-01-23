addpath(genpath('../KLTpicker/aspire-master'))

L = 15;
r = 1;
L_half = (L-1)/2;
x = (-L_half:L_half)/L_half; y = x; z = x;
voxel1 = double(((x-0.6).^2 + (y.^2)' + reshape(z,1,1,[]).^2) <= 0.25);
%voxel2 = double((3*x.^2 + ((y).^2)' + reshape(z,1,1,[]).^2) <= 0.5);
voxel2 = double(((x+0.6).^2 + ((y).^2)' + reshape(z,1,1,[]).^2) <= 0.25);
voxel2 = voxel1;
%voxel2 = -voxel1;



if(~isfile('projections.mat'))
   projection_num = 5000;
   rots1 = rand_rots(projection_num);
   projs1 = cryo_project(voxel1,rots1,L);
   projs1 = projs1.*(randi(2,1,1,projection_num)-1.5)*2;
   rots2 = rand_rots(projection_num);
   projs2 = cryo_project(voxel2,rots2,L).*(randi(2,1,1,projection_num)-1.5)*2;
   save('projections.mat','projection_num','rots1','projs1','rots2','projs2');
else
    load('projections.mat')
end


projs = cat(3,projs1,projs2);
rots = cat(3,rots1,rots2);
%meu = (voxel1+voxel2)/2;
meu = zeros(size(voxel1));
%%
learn_rate = 1e-4*2*100;
momentum = 0.5;
vel = [];
batch_size = 1;

% learning_rate_exp = [1.01 ;  0.998 ; 0.995]; learning_rate_exp_ind = [200 ; 600];
% exp_val = learning_rate_exp(1);

n = size(projs,3);

u_0 = randn(L,L,L,r);
P_u0 = zeros(L,L,r,batch_size);
PtP_u0 = zeros(L^3,r,batch_size);
%u_0 = cat(4,voxel1,voxel2);
%u_0 = voxel1;

reg_param = 0;
for i = 1:1600
    s = randi(n,1,batch_size);
    proj_s = projs(:,:,s);
    rot_s = rots(:,:,s);
    
    for j = 1:r
        P_u0(:,:,j,:) = cryo_project(u_0(:,:,:,j),rot_s);
        PtP_u0(:,j,:) = reshape(im_backproject_arr(squeeze(P_u0(:,:,j,:)),rot_s),L^3,batch_size);
    end
    PtPui_uj = pagemtimes(PtP_u0,'transpose',reshape(u_0,L^3,r),'none'); %size r x r x batch_size, the element i,j,l contains inner product between projected-backprojected u_i and (non projected-backprojected) u_j for the l-th image in the batch    
    PtP_Sigma_PtP_ui = reshape(pagemtimes(PtP_u0,PtPui_uj),L,L,L,r,batch_size); 

    P_meu = cryo_project(meu,rot_s);
    Pt_y = reshape(im_backproject_arr((proj_s-P_meu),rot_s),L,L,L,1,batch_size);
    Pt_y_yt_P_u =sum(Pt_y.*u_0,[1,2,3]).*Pt_y; 

    grad_u0 = 4*(sum(PtP_Sigma_PtP_ui,5) - sum(Pt_y_yt_P_u,5))/(L^3 * batch_size);
    %grad_u0 = 4*sum((sum(PtP_u0.*u_0,[1 2 3]).*PtP_u0 + reg_param*u_0 - sum(Pt_y.*u_0,[1 2 3]).*Pt_y),4) / (L^3 * batch_size);
    if(any(isnan(grad_u0(:))))
        break
    end
    %u_0 = u_0 - 1e-11 * grad_u0;
    [u_0,vel] = sgdmupdate(u_0,grad_u0,vel,learn_rate,momentum);
    
    
    
    %display(['Norm error ' sprintf('%0.5e',norm(u_0(:) - (voxel1(:)-voxel2(:))/2)/norm((voxel1(:)-voxel2(:))))])

    learn_rate = (0.999+0.001) * learn_rate;
   
    
    if(mod(i,100) == 0)
        display(['Cost function val ' sprintf('%0.5e',covar_cost_func(proj_s-P_meu,P_u0))]);
    end
end



%voxelSurf((abs(u_0) > 0.4).*u_0)





