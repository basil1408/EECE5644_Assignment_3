close all
clear all
N=[10,100,1000,10000];
prior_true = [0.25,0.25,0.25,0.25];
mu_true = [-10 0 10 20;0 0 0 0];
Sigma_true(:,:,1) = [3 1;1 2];
Sigma_true(:,:,2) = [12 1;1 2];
Sigma_true(:,:,3) = [7 1;1 16];
Sigma_true(:,:,4) = [15 1;1 6];
for i=1:4
    n=N(1,i);
    x = randGMM(n,prior_true,mu_true,Sigma_true);
    x=x';
    hold off
    figure(n)
    for v=1:n
        plot(x(v,1),x(v,2),'r*');
        hold on
    end
    for j=1:6
        for k=1:9
            [test train]=split(n,x,k);
            model1 = fitgmdist(train,j,'RegularizationValue',0.1);
            prior=model1.ComponentProportion;
            mu1=model1.mu;
            mu1=mu1';
            Sigma1=model1.Sigma;
            z=(evalGMM(test,prior,mu1,Sigma1));
        end
        logLikelihood(i,j) = sum(log(z));
        
    end
    
end
Q=[1 2 3 4 5 6];
figure(5)
for j=1:6
    plot(Q(1,j),abs(logLikelihood(1,j)),'bo');
    hold on
end
title("10 Samples");
xlabel("Number of Distribution");
ylabel("Absolute value of log likelihood");
hold off
figure(6)
for j=1:6
    plot(Q(1,j),abs(logLikelihood(2,j)),'bo');
    hold on
end
title("100 Samples");
xlabel("Number of Distribution");
ylabel("Absolute value of log likelihood");
hold off
figure(7)
for j=1:6
    plot(Q(1,j),abs(logLikelihood(3,j)),'bo');
    hold on
end
title("1000 Samples");
xlabel("Number of Distribution");
ylabel("Absolute value of log likelihood");
hold off
figure(8)
for j=1:6
    plot(Q(1,j),abs(logLikelihood(4,j)),'bo');
    hold on
end
title("10000 Samples");
xlabel("Number of Distribution");
ylabel("Absolute value of log likelihood");
hold off
function x = randGMM(N,prior,mu,Sigma)
d = size(mu,1); 
cum_prior = [0,cumsum(prior)];
u = rand(1,N); 
x = zeros(d,N); 
labels = zeros(1,N);
for m = 1:length(prior)
    ind = find(cum_prior(m)<u & u<=cum_prior(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,prior,mu,Sigma)
x=x';
gmm = zeros(1,size(x,2));
for m = 1:length(prior) % evaluate the GMM on the grid
    gmm = gmm + prior(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end

end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
function [q w]=split(n,x,fold_num)
t=n/10;
y=fold_num*t;
p=y+t;
y=y+1;
q=x(y:p,:);
w=[x(1:(y-1),:) ;x((p+1):n,:)];
end