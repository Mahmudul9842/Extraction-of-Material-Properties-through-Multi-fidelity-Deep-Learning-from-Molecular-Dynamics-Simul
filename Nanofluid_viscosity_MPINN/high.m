function yH=high(xH)

m=length(xH);
yL=[];
for i=1:m
if 0<=xH(i) && xH(i)<=0.5
    yH(i)=2*low(xH(i))-20*xH(i)+20;
elseif .5<xH(i) && xH(i)<=1
    yH(i)=4+2*low(xH(i))-20*xH(i)+20;
else
end
end
end