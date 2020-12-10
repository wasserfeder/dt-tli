clear;
clc;

load('data/Naval/naval_preproc_data_online.mat')
t_min = 0;
t_max = t(end);

num_signals = size(data,1);
min_signals = zeros(1,num_signals);
max_signals = zeros(1,num_signals);

for i=1:num_signals
    min_signals(i) = min(data(i,:));
    max_signals(i) = max(data(i,:));
end

min_pi = min(min_signals);
max_pi = max(max_signals);

% meshgrid and surf

r = 0.01;
h = max_pi - min_pi;
figure(1)
for t0=0:5:t_max
    for t1=0:5:t_max
        [X,Y,Z] = cylinder(r);
        X = X + t0;
        Y = Y + t1;
        Z = Z * h;
        Z = Z + min_pi;
        surf(X,Y,Z);
        hold on;
    end
end






