function [X] = featureNormalize(X_norm, mu, sigma)
% FEATURENORMALIZE(X) returns a normalized version of X where
% the mean value of each feature is 0 and the standard deviation
% is 1. This is often a good preprocessing step to do when
% working with learning algorithms.


t = ones(1,length(X_norm));
X=X_norm.*(t * sigma)+(t * mu) ; % Vectorized

end