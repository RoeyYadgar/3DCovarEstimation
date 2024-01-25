function ims = cryo_project_arr(vol, rot_matrice)
    
    L = size(vol,1);
    n = size(vol,4);
    k = size(rot_matrice,3);
    ims = zeros(L,L,k,n);
    for i = 1:n
        ims(:,:,:,i) = cryo_project(vol(:,:,:,i),rot_matrice);
    end
   
end