clear all
close all
P1=0.3;
P2=0.7;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
N1=0;
N2=0;
for i=1:999
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else 
        x2(:,k)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        k=k+1;
    end
end
label1=zeros(N1,1);
label2=ones(N2,1);
label=[label1 ;label2];
mu1hat = mean(x1,2); 
S1hat = cov(x1');
mu2hat = mean(x2,2);
S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1;
y2 = w'*x2;
figure(1),
 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
axis equal,
title('Scatter plot before Fisher LDA');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
db=(mu1hat+mu2hat)/2;
y = w'*db;

hold on
M1=size(y1);
M2=size(y2);
j=1;
k=1;
for i=1:M1(1,2)
    if y1(1,i)>y
        new_label1(i)=0;
        y1new(j)=y1(1,i);
        j=j+1;
    else
        new_label1(i)=1;
        miss1(k)=y1(1,i);
        k=k+1;
    end
end
l=1;
m=1;
for i=1:M2(1,2)
    if y2(1,i)<y
        
        y2new(l)=y2(1,i);
        new_label2(i)=1;
        l=l+1;
    else
        new_label2(i)=0;
        miss2(m)=y2(1,i);
        m=m+1;
    end
end
L=size(miss1);
K=size(miss2);
J=size(y1new);
G=size(y2new);
error= L(1,2)+K(1,2);
E=error/999;
error= error
probability_of_error_FLDA=E
probability_of_error_percentage=E*100
figure(2) 
plot(y1new(1,:),zeros(1,J(1,2)),'b.');
hold on;
plot(y2new(1,:),zeros(1,G(1,2)),'r.');
hold on
plot(miss1(1,:),zeros(1,L(1,2)),'ro');
hold on;
plot(miss2(1,:),zeros(1,K(1,2)),'bo');
axis equal;

xline(y);
title('Plot after Fisher LDA');
ylabel('Fisher LDA projection vector','FontSize',10);
xlabel('Fisher LDA scores projection','FontSize',10);
legend('Class 1 as Class 1','Class 2 as Class 2','Class1 as Class 2','Class 2 as Class 1','Decision boundary');
grid on
hold off
new_label=[new_label1 new_label2];
new_label=new_label';
c=confusionmat(label,new_label)

clear all
close all
P1=0.3;
P2=0.7;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
N1=0;
N2=0;
for i=1:999
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j(i)=0;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        N2=N2+1;
        j(i)=1;
    end
end
e= [x;j];
e=e';
X=x';
j=j';
y=j; 
plotData(X,y);
xlabel('Feature 1')
ylabel('Feature 2')
% Specified in plot order
legend('Class 1', 'Class 2')
grid on
axis equal
hold off;
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
X = [ones(m, 1) X];
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
% Plot Boundary
plotDecisionBoundary(theta, X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Feature 1')
ylabel('Feature 2')
% Specified in plot order
legend('Class 1', 'Class 2','Decision Boundary')
hold off;
% Compute accuracy on our training set
p = predict(theta, X);
P=0;
for i=1:999
    if p(i)~=y(i)
        P=P+1;
    end
end
fprintf('Number of errors: %f\n', P);
fprintf('probability of error for Logistic regression: %f\n', mean(double(p ~= y)));
fprintf('probability of error percentage: %f\n', mean(double(p ~= y))*100);
c=confusionmat(y,p)

clear all
close all
P1=0.3;
P2=0.7;
cov1=[3 1;1 0.8];
cov2=[3 1;1 0.8];
mu1=[0;0];
mu2=[3;3];
j=1;
k=1;
for i=1:999
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x(:,i)= mvnrnd(mu1,cov1,1);
       label(i)=0;
    else 
        x(:,i)= mvnrnd(mu2,cov2,1);
        label(i)=1;
    end
end
label=label';
x=x';
figure;

for i=1:999
    if label(i)==0
        plot(x(i,1),x(i,2),'r+');
        hold on
    else
        plot(x(i,1),x(i,2),'b*');
        hold on;
    end
end
title('Actual Scatter plot of data generated');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on
hold off
%{
Now we take the samples in x and predict their class using MAP
Classification rule. We use the concept of dichotomizer.
%}
c=1;
d=1;
for i=1:999
    e=x(i,:);
    a= mvnpdf(e',mu1,cov1);
    b= mvnpdf(e',mu2,cov2);
    g= log(a/b)+log(P1/P2);     % g is the discriminant function
    if g>0
        q1(c,:)=x(i,:);         %q1 contains elements classified as class 1
        c=c+1;
        h(i)=0;                 %h contains the infered labels
    else
        q2(d,:)=x(i,:);         %q2 contains elements classified as class 2
        d=d+1;
        h(i)=1;
    end
end
figure(2)
plot(q2(:,1),q2(:,2),'+');
hold on
plot(q1(:,1),q1(:,2),'o');
title('Infered Scatter plot of data using MAP');
ylabel('Feature 2','FontSize',10);
xlabel('Feature 1','FontSize',10);
legend('Class 1','Class 2');
grid on;
hold off
q=[q1;q2];

r=0;
h=h';
z=eq(label,h);
for i=1:900
    if z(i)==0
        r=r+1;
    end
end
number_of_errors=r
probability_of_error_MAP =(r/999)
probability_of_error_percentage =(r/999)*100
c=confusionmat(label,h)

function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
J = (-1 / m) * sum(y.*log(sigmoid(X * theta)) + (1 - y).*log(1 - sigmoid(X * theta)));
temp = sigmoid (X * theta);
error = temp - y;
grad = (1 / m) * (X' * error); 

end
function plotData(X, y)
figure; 
hold on;
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), '+');
plot(X(neg, 1), X(neg, 2), 'o');
hold off;

end
function plotDecisionBoundary(theta, X, y)
plotData(X(:,2:3), y);
hold on
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y)
legend('Feature 1', 'Feature 2', 'Decision Boundary')
grid on
axis equal
hold off

end
function p = predict(theta, X)
m = size(X, 1); % Number of training examples
% You need to return the following variables correctly
p = zeros(m, 1);
p = round(sigmoid(X * theta));

end

function g = sigmoid(z)
g = zeros(size(z));
g = 1./(1 + exp(-1*z));
end



