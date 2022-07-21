%function [aold,Vold,time,timevi,countsub,countvi,Values,Valuesvi] = accvdppi(deltax,ncont,mmfile)

x = 2; %bounds to solve in a square [-x x][-x x]
deltax = 0.02; %gridspacing
xnode = -x : deltax : x;

[X,Y] = ndgrid(xnode);

h = .05 * deltax;
ncont = 200;

lambda = .001; % discount factor, needs to be small to mimic the HJB equation.
dim  = length(xnode);

Vold = ones(dim, dim);
inside = ones(dim, dim, ncont);
inside(2:dim-1, 2:dim-1, :) = 0;
inside1 = squeeze(inside(:,:,1));
Vold(X.^2 + Y.^2 == 0) = 0;
Tg = Vold;

writematrix(xnode,'xnode.csv')

aold = zeros(dim, dim); %initial control

count = 0;
subc = 0;
cb = 20; % control bounds to look for a control variables in [-cb cb]
Acont = -cb:(2 * cb / (ncont - 1)):cb;


Varr = zeros(dim, dim);
Disc = zeros(dim, dim);

T = zeros(dim, dim, ncont);
Valuesvi = zeros(dim, dim, dim);


Values = zeros(dim, dim, dim);
diff1 = 1;
count = 1;
tol = 1 / 5 * deltax^2;

gamma = 0.5;
p = 2;
Disc = 0.5 * (X.^2 + Y.^2);

tic
while(diff1 > tol)
VI = Vold;

diff = 1;

arrx = X + h * Y ;
arry = Y + h * (X.^3 + aold);
countsub(count) = 0;

while (diff > 10 * tol)
    Varr = interpn(X, Y, Vold, arrx, arry, 'linear', 60); %assigns 5 when outside
    %disp(Varr)
    Vnew = exp(-lambda * h) * Varr + h * (Disc + gamma * abs(aold).^p);
    Vnew(Tg == 0) = 0;
    diff = max(max(abs(Vnew - Vold))); 
    %disp(diff)
    Vold = Vnew;   
    countsub(count) = countsub(count) + 1;
end


%Values(:, : , count) = Vold;

for k = 1:ncont
    arrx = X + h * Y;
    arry = Y + h * (X.^3 + Acont(k));
    Varr=interpn(X, Y, Vold, arrx, arry, 'linear', 1);
    T(:,:,k) = exp(-lambda * h) * Varr + h * (Disc + gamma * abs(Acont(k)).^p);
end
%disp(T(:,:,ncont))
%disp(sum(T, 'all'))

T(inside == 1) = 5;
[~, donde] = min(T, [], 3);
    
%disp(donde)
 for i = 1:ncont
     aold(donde == i) = Acont(i);
 end
 
    %disp(aold)
    %disp(Varr)
    diff1 = max(max(abs(VI - Vold)))
    %disp(diff1)
    count = count + 1;
end

time = toc;
Values(:, :, [count:dim]) = [];

figure
surf(Vold)

writematrix(Vold,'exact_solution.csv')


figure
surf(aold)

writematrix(aold,'exact_control.csv')

%SDRE solution
Pi=@(x1,x2)[ ((x1^4 + 1)^(1/2)*(2*(x1^4 + 1)^(1/2) + 2*x1^2 + 1)^(1/2))/2 (x1^4 + 1)^(1/2)/2 + x1^2/2;
    (x1^4 + 1)^(1/2)/2 + x1^2/2 (2*(x1^4 + 1)^(1/2) + 2*x1^2 + 1)^(1/2)/2 ];

Xv = X(:);
Yv = Y(:);
Vex = [];
for i = 1:length(Xv)
    Vex(i) = [Xv(i) Yv(i)] * Pi(Xv(i),Yv(i)) * [Xv(i) Yv(i)]';
end

Vex = reshape(Vex, size(X));

figure
surf(abs(Vold - Vex))
