function preprocess(filename,L,pixelSize)

ims = ReadMRC([filename '.mrcs']);
starfile = readSTAR([filename '.star']);

centered_images = zeros(size(ims));

for i = 1:size(ims,3)
	shiftx = starfile(2).data{i}.rlnOriginXAngst;
	shifty = starfile(2).data{i}.rlnOriginYAngst;

	centered_images(:,:,i) = reshift_image(ims(:,:,i),[-shiftx -shifty]/pixelSize);
end

ds_images = cryo_downsample(centered_images,L,1);
newPixelSize = pixelSize*size(centered_images,1)/L;
WriteMRC(ds_images,newPixelSize,[filename '_preprocessed_L' num2str(L) '.mrcs']);

%% Estimate Power spectrum

%half_L = floor((L-1)/2);
%x = (-half_L):half_L; y = x;
%[X,Y] = meshgrid(x,y);

%R = sqrt(X.^2 + Y.^2);
%used_inds = find(R > half_L);

%psd = cryo_epsdS(ds_images,used_inds,half_L/2);

%psd = psd(1:2:end,1:2:end);
%save('psd.mat','psd');
