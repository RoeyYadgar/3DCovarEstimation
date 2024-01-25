addpath(genpath('../KLTpicker/aspire-master'))

L = 15;
L_half = (L-1)/2;
x = (-L_half:L_half)/L_half; y = x; z = x;
voxel1 = double((x.^2 + (y.^2)' + reshape(z,1,1,[]).^2) <= 0.5);
%voxel2 = double((3*x.^2 + ((y).^2)' + reshape(z,1,1,[]).^2) <= 0.5);
%voxel2 = double((abs(x) + abs((y))' + reshape(abs(z),1,1,[])) <= 0.8);
voxel2 = -voxel1;

if(~isfile('projections.mat'))
   projection_num = 5000;
   rots1 = rand_rots(projection_num);
   projs1 = cryo_project(voxel1,rots1,L,'double');
   rots2 = rand_rots(projection_num);
   projs2 = cryo_project(voxel2,rots2,L,'double');
   save('projections.mat','projection_num','rots1','projs1','rots2','projs2');
else
    load('projections.mat')
end


projs = cat(3,projs1,projs2);
rots = cat(3,rots1,rots2);
meu = (voxel1+voxel2)/2;
%%
%learn_rate = 5/10;
%momentum = 0.9;
learn_rate = 0.5;
momentum = 0.9;
vel = [];
batch_size = 8;
n = size(projs,3);

numIter = 1500;
verbose_freq = 50;
cost_func_val = zeros(numIter/verbose_freq,1);
norm_err = zeros(numIter/verbose_freq,1);

u_0 = randn(L,L,L);
%u_0 = voxel1;

reg_param = 0;
for i = 1:numIter
    s = randi(n,1,batch_size);
    proj_s = projs(:,:,s);
    rot_s = rots(:,:,s);
    rot_s_inv = transposeTensor(rot_s);
    
    P_u0 = cryo_project(u_0,rot_s);
    PtP_u0 = im_backproject_arr(P_u0,rot_s_inv);
    P_meu = cryo_project(meu,rot_s);
    Pt_y = im_backproject_arr((proj_s-P_meu),rot_s_inv);
    
    grad_u0 = 4*sum((sum(PtP_u0.*u_0,[1 2 3]).*PtP_u0 + reg_param*u_0 - sum(Pt_y.*u_0,[1 2 3]).*Pt_y),4) / (L^3 * batch_size);
    if(any(isnan(grad_u0(:))))
        break
    end
    [u_0,vel] = sgdmupdate(u_0,grad_u0,vel,learn_rate,momentum);
    
    
    

    
    learn_rate = (0.997) * learn_rate;
  
    if(mod(i,verbose_freq) == 0)
        cost_func_val(i/verbose_freq) = covar_cost_func(proj_s-P_meu,reshape(P_u0,L,L,1,batch_size));
        norm_err(i/verbose_freq) = min(norm(u_0(:) - voxel1(:)),norm(u_0(:) + voxel1(:)))/norm(voxel1(:));
        display(['Cost function val ' sprintf('%0.5e',cost_func_val(i/verbose_freq))]);
        display(['Norm Difference from ground turth: ' sprintf('%0.5e',norm_err(i/verbose_freq))]);
    end
    
end


%norm_diff = min(norm(u_0(:) - voxel1(:)),norm(u_0(:) + voxel1(:)))/norm(voxel1(:));
%display(['Norm Difference from ground turth: ' sprintf('%0.5e',norm_diff)]);

voxelSurf((abs(u_0) > 0.8).*u_0)



%save('result1.mat','u_0','norm_err','cost_func_val','numIter','batch_size','learn_rate','momentum')

%%
