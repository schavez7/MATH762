function True_Solution

x = linspace(0,2*pi,25);
y = x;

[X,Y] = meshgrid(x,y);

figure(1)
mesh(X,Y,0*X)
axis([0 2*pi 0 2*pi -1 1])
for i = 2:100
    time = i*pi/50;
    solution = sin(X).*sin(Y).*sin(time);
    mesh(X,Y,solution)
    axis([0 2*pi 0 2*pi -1 1])
    disp(time)
    pause(0.1)
end
