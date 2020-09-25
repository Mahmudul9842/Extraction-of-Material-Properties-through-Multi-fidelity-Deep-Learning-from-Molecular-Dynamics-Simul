function yL=low(xL)
m=length(xL);
yL=[];
for i=1:m
if 0<=xL(i) && xL(i)<=0.5
    yL(i)=(.5*(6*xL(i)-2).^2)*sin(12*xL(i)-4)+10*(xL(i)-.5)-5;
elseif .5<xL(i) && xL(i)<=1
    yL(i)=3+(.5*(6*xL(i)-2).^2)*sin(12*xL(i)-4)+10*(xL(i)-.5)-5;
else
end
end
end