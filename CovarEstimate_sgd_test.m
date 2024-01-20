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
   projs1 = cryo_project(voxel1,rots1,L);
   rots2 = rand_rots(projection_num);
   projs2 = cryo_project(voxel2,rots2,L);
   save('projections.mat','projection_num','rots1','projs1','rots2','projs2');
else
    load('projections.mat')
end


projs = cat(3,projs1,projs2);
rots = cat(3,rots1,rots2);
meu = (voxel1+voxel2)/2;
%%
learn_rate = 1e-4*2*100;
momentum = 0.5;
vel = [];
batch_size = 1;

% learning_rate_exp = [1.01 ;  0.998 ; 0.995]; learning_rate_exp_ind = [200 ; 600];
% exp_val = learning_rate_exp(1);

n = size(projs,3);

u_0 = ones(L,L,L);
%u_0 = (voxel1-voxel2)/2 + 0*randn(L,L,L);
reg_param = 0;
for i = 1:1600*3
    s = randi(n,1,batch_size);
    proj_s = projs(:,:,s);
    rot_s = rots(:,:,s);
    
    P_u0 = cryo_project(u_0,rot_s);
    PtP_u0 = im_backproject_arr(P_u0,rot_s);
    P_meu = cryo_project(meu,rot_s);
    Pt_y = im_backproject_arr((proj_s-P_meu),rot_s);
    
    %grad_u0 = 4*(sum(PtP_u0(:).*u_0(:)).*PtP_u0 + reg_param*u_0 - sum(Pt_y(:).*u_0(:)).*Pt_y) / L^3;
    grad_u0 = 4*sum((sum(PtP_u0.*u_0,[1 2 3]).*PtP_u0 + reg_param*u_0 - sum(Pt_y.*u_0,[1 2 3]).*Pt_y),4) / (L^3 * batch_size);
    if(any(isnan(grad_u0(:))))
        break
    end
    %u_0 = u_0 - 1e-11 * grad_u0;
    [u_0,vel] = sgdmupdate(u_0,grad_u0,vel,learn_rate,momentum);
    
    
    %display(['Norm error ' sprintf('%0.5e',norm(u_0(:) - (voxel1(:)-voxel2(:))/2)/L^3)])
    display(['Norm error ' sprintf('%0.5e',norm(u_0(:) - (voxel1(:)-voxel2(:))/2)/norm((voxel1(:)-voxel2(:))))])

    % if(mod(i,100) == 0)
    %     learn_rate = learn_rate / 10;
    % end
    learn_rate = (0.999+0.001) * learn_rate;
    % learn_rate = exp_val * learn_rate;
    % if(~isempty(find(i == learning_rate_exp_ind)))
    %     exp_val = learning_rate_exp(find(i == learning_rate_exp_ind) + 1);
    % end
    
end



voxelSurf((abs(u_0) > 0.4).*u_0)



%voxelSurf(abs(fftshift(fftshift(fftshift(fftn(u_0),1),2),3)) > 10)

%%
% p.rot_matrices = rots1;
% p.ctf = ones(L,L,1);
% p.ctf_idx = ones(1,size(rots1,3));
% p.ampl = ones(1,size(rots1,3));
% p.shifts = zeros(2,size(rots1,3));
% 
% cryo_mean_kernel_f(L,p)