d = 101;
u = ((randi(2,d,1))-1.5)*2;

% s = ((randi(2,1000,1))-1.5)'*2;
% sigma = cov((u.*s)');
% B = ifft(fft(sigma,d,1),d,2);
%%
u_0 = randn(size(u));
k = 4;
P = @(u,n) P_opt(u,n,k);
Pt = @(Pu,n) PT_opt(Pu,n,k,d);
reg_param = 0;
for i = 1:10000
    n = randi(k/2);
    %n = mod(i,k); n(n==0) = k;
    % PtP_u0 = PT_opt(P_opt(u_0,n),n,d);
    PtP_u0 = Pt(P(u_0,n),n);
    y = P(u,n);
    Pt_y = Pt(y,n);
    u_0 = u_0 - 0.001 * (PtP_u0 * PtP_u0.' - Pt_y*Pt_y.' + reg_param*u+0*u_0')*u_0;

end
%plot(real(u_0))
%plot(real((PtP_u0 * PtP_u0' - Pt_y*Pt_y')*u_0))
% u_0' * (PtP_u * PtP_u' - Pt_y*Pt_y') %grad 

plot(unwrap(angle(fft(u_0)./fft(u))))
plot(real(u_0./u))
function Pu = P_opt(u,n,k)
    fftu = fft(u);
    dc = fftu(1);
    fftu = reshape(fftu(2:end),[],k);
    Pu = [dc ; fftu(:,n) ; fftu(:,end-n+1)];
end

function u = PT_opt(Pu,n,k,d)
    % fftu = zeros(d,1);
    % fftu(n) = Pu(1);
    % fftu(end-n+1) = Pu(2);
    % u = ifft(fftu);
    dc = Pu(1);
    Pu = reshape(Pu(2:end),[],2);
    fftu = zeros((d-1)/k,k);
    fftu(:,n) = Pu(:,1);
    fftu(:,end-n+1) = Pu(:,2);
    u = ifft([dc ; fftu(:)]);
end