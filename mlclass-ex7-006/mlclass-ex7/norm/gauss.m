X = linspace(-10, 4);
mu = [-3 -3]';
sig = ([4 2] .^ (0.5))';

Y = zeros(length(mu), length(X));

for i=1:length(mu)
    Y (i,:) = normpdf(X, mu(i), sig(i));
end

