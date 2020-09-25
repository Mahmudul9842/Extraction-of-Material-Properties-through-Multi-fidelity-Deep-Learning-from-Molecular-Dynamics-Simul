function [Jalpha grad_alpha]=Calculate_J(alpha,h_Lin,h_NL,Reg_Lin, Reg_NL,JL,yH)

h=h_Lin*alpha+h_NL*(1-alpha);
Jalpha=sum(sum((yH-h).^2))/(2*length(h(1,:)));



grad_alpha=(1/length(h(1,:)))*sum(sum((h_Lin*alpha+h_NL*(1-alpha)-yH).*(h_Lin-h_NL)));

end