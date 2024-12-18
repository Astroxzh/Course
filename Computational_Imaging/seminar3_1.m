N = 2^6; dx = 100e-9;
L = N*dx;
x = (-N/2:N/2-1)/dx; z = x;
[X,Z] = meshgrid(x);

%wavevector
k = 2*pi/633e-9;

%source coordinate
xs = 0;
zs = -1.25*L;

%exp1, spherical wave
R = sqrt((X-xs).^2 +(Z-zs).^2);
Uin = exp(1i*k*R)./(1i*k*R);

%exp2, plane wave
% Uin = exp(1i*k*Z);

figure(1)
hsvplot(Uin)
axis image off
title('U_{in}')


%% determing random source points

% cmap = jet(50);
% cmap = [aleph]
rng(0)
idx = 1:10:N^2; numScatterers = length(idx); idx = min(idx+randi([1 4],size(idx)),N^2);
aleph = 1e-2*ones(numScatterers,1);
source = zeros(N,N);
source(idx) = aleph;
R = sqrt(X.^2+Y.^2);


%% normalization factor

G = mean(aleph)*exp(1i*k*R) ./ (1i*k*R+eps);

%method 1: Foldy Lax
M = zeros(N^2,N^2);
for loop = 1:length(idx)
    R = sqrt((X-X(idx(loop))).^2 +(Z-Z(idx(loop))).^2);
    %2D Green's function:Henkel
    G = aleph(loop) *1i*pi/besselh(0,k*R);
    G(idx(loop)) = 0;
    M(:,idx(loop)) = aleph(loop) * G(:);
end

Ufoldy = reshape(sparse(eye(N^2)-M) \ Uin(:), [N,N]);

%% Born appro.

