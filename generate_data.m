run('~/aspire/initpath');
E_orig = C4_params;
L = 100;
k = 360;
thetas = linspace(0,360,k);
vols = single(zeros(L^3,length(thetas)));
for i = 1:length(thetas)
    theta = thetas(i);
    E = E_orig;
    coor_rot = rotz(theta) * E_orig(:,5:7)';
    E(:,5:7) = coor_rot';

    f_handle = @() E(1:8,:);
    
    
    vols(:,i) = single(reshape(cryo_gaussian_phantom_3d(f_handle,L,1),[],1));
   
    
end
vols = vols + single(reshape(cryo_gaussian_phantom_3d('micky',L,1),[],1));
%WriteMRC(reshape(vols,L,L,[]),1,p)
%c = cov(vols');
%[V,D] = eigs(c,10);
save("data/vols.mat","vols");
