%% This function performs K-Means Clustering on the data
% data - the data on which the clustering is to be performed
% k - the number of desired clusters
% centers - the vector containing the locations of the cluster centers
% clusterVariance - the vector containing the variances of each cluster
% dMax - the maximum distance between two centers

function [centers, centerVariance, dMax] = kMeansClustering(data, k, showPlots)

maxVal = 10^7;

%% initialization of vectors
[numVals, ~] = size(data);
% randomly assign initial cluster centers
centers = data(randsample(numVals, k));
centerPopulation = zeros(size(centers));
centerVariance = zeros(size(centers));
centerRadius = zeros(size(centers));
dataClusterIdx = zeros(size(data));
centerDistances = zeros(k, k);

%% display cluster formation 
if showPlots == 1
    figure(1)
    plot(data, data, 'y.');
    hold on
    plot(centers, centers, 'b*');
    pause(0.1);
end

%% perform clustering
stopClustering = 0;
while stopClustering == 0
    % calculate distance matrix
    distMatrix = maxVal * ones(numVals, k);
    for i = 1:k
        distMatrix(:, i) = (data - centers(i, 1)).^2;
    end
    
    % find the closest center
    for j = 1:numVals
        tempMinVal = min(distMatrix(j, :));
        idx = find(distMatrix(j, :) == tempMinVal);
        dataClusterIdx(j) = idx(1,1);
    end
    
    % update cluster centers
    tempCenters = zeros(size(centers));
    for i = 1:k
        [idr, ~] = find(dataClusterIdx == i);
        tempCenters(i, 1) = mean(data(idr, 1));
        maxVal = max(data(idr, 1));
        dist1 = abs(tempCenters(i, 1) - maxVal);
        minVal = min(data(idr, 1));
        dist2 = abs(tempCenters(i, 1) - minVal);
        radius = max(dist1, dist2);
        centerRadius(i, 1) = radius;
        centerVariance(i, 1) = var(data(idr, 1));
        centerPopulation(i, 1) = size(idr, 1);
    end
    
    % display cluster formation
    if showPlots == 1
        figure(1)
        clf(1)
        plot(data, data, 'y.');
        hold on
        plot(tempCenters, tempCenters, 'b*');
        hold on
        for i = 1:k
            circle(tempCenters(i, 1), tempCenters(i, 1), centerRadius(i, 1));
            hold on
        end
        pause(0.5);
    end
    
    % stop criterion if none of the centers has changed
    if nnz(centers == tempCenters) == k
        stopClustering = 1;
    end
    centers = tempCenters;
end

%% Calculate variances and distances
for i = 1:k
    % assign variances to clusters with 0 variance
    sumVariance = sum(centerVariance);
    if centerPopulation(i, 1) == 1
        centerVariance(i, 1) = sumVariance/(k-1);
    end
    
    % find dMax
    for j = 1:k
        if i == j
            centerDistances(i, j) = 0;
        else
            centerDistances(i, j) = (centers(i, 1) - centers(j, 1))^2;
        end
    end
end
dMax = max(centerDistances(:));
end

%% This function was taken from matlab central
function circle(x,y,r)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp);
end

