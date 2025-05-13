% Last edit: 11/03/25
% Brain_X_SmaRt Group: Bowen Zhi, Xuanyun Qiu, Shiwei Liu, Ruiping Chi

function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    
    groupSize = 20;
    smoothingWindow = 50;
    
    processedTrial = binAndSquare(test_data, groupSize, 1); % preprocessing
    smoothedTrial = calculateFiringRates(processedTrial, groupSize, smoothingWindow); % preprocessing with smoothing
    totalTimeSteps = size(test_data.spikes, 2);
    % totalNeurons = size(smoothedTrial(1, 1).rates, 1);
    
    % extract relevant parameters from the model
    if totalTimeSteps <= 560
        
        idx = (totalTimeSteps/groupSize) - (320/groupSize) + 1;
        lowSpikeNeurons = modelParameters.lowFirers{1};
        smoothedTrial.rates(lowSpikeNeurons, :) = [];
        
        firingData = reshape(smoothedTrial.rates, [], 1);
        % totalNeurons = totalNeurons - length(lowSpikeNeurons);
    
        
        classifierWeights = modelParameters.classify(idx).wLDA_kNN;
        % pcaDimensions = modelParameters.classify(idx).dPCA_kNN;
        ldaDimensions = modelParameters.classify(idx).dLDA_kNN;
        optimalWeights = modelParameters.classify(idx).wOpt_kNN;
        meanFiringRates = modelParameters.classify(idx).mFire_kNN;
        
        firingDataTest = optimalWeights' * (firingData - meanFiringRates); 
        
        predictedLabel = classifyKNN(firingDataTest, classifierWeights);
        modelParameters.actualLabel = predictedLabel;
        if predictedLabel ~= modelParameters.actualLabel
            predictedLabel = modelParameters.actualLabel;
        end
    
    else % keep using the parameters derived from the longest training data length
        predictedLabel = modelParameters.actualLabel;
        % idx = 1;
        lowSpikeNeurons = modelParameters.lowFirers{1};
        smoothedTrial.rates(lowSpikeNeurons, :) = [];
        
        firingData = reshape(smoothedTrial.rates, [], 1);
        % totalNeurons = totalNeurons - length(lowSpikeNeurons);
    end
    
    
    if totalTimeSteps <= 560
        
        % apply time shift
        % totalNeurons = size(smoothedTrial(1, 1).rates, 1) - length(lowSpikeNeurons);
    
        idx = (totalTimeSteps/groupSize) - (320/groupSize) + 1;
        avgX = modelParameters.averages(idx).avX(:, predictedLabel);
        avgY = modelParameters.averages(idx).avY(:, predictedLabel);
        meanFiring = modelParameters.pcr(predictedLabel, idx).fMean;
        pcaWeightsX = modelParameters.pcr(predictedLabel, idx).bx;
        pcaWeightsY = modelParameters.pcr(predictedLabel, idx).by;
        x = (firingData - mean(meanFiring))' * pcaWeightsX + avgX;
        y = (firingData - mean(meanFiring))' * pcaWeightsY + avgY;
        
        try
            x = x(totalTimeSteps, 1);
            y = y(totalTimeSteps, 1);
        catch
            x = x(end, 1);
            y = y(end, 1);
        end
    
    elseif totalTimeSteps > 560 % continue using model derived from the longest training time
    
        % apply time shift to the data
        % totalNeurons = size(smoothedTrial(1, 1).rates, 1) - length(lowSpikeNeurons);
        
        avgX = modelParameters.averages(13).avX(:, predictedLabel);
        avgY = modelParameters.averages(13).avY(:, predictedLabel);
        % meanFiring = modelParameters.pcr(predictedLabel, 13).fMean;
        pcaWeightsX = modelParameters.pcr(predictedLabel, 13).bx;
        pcaWeightsY = modelParameters.pcr(predictedLabel, 13).by;
        
        x = (firingData(1:length(pcaWeightsX)) - mean(firingData(1:length(pcaWeightsX))))' * pcaWeightsX + avgX;
        y = (firingData(1:length(pcaWeightsY)) - mean(firingData(1:length(pcaWeightsY))))' * pcaWeightsY + avgY;
        
        try
            x = x(totalTimeSteps, 1);
            y = y(totalTimeSteps, 1);
        catch
            x = x(end, 1);
            y = y(end, 1);
        end
    end

end


function processedTrial = binAndSquare(trial, groupSize, shouldSquare)

    processedTrial = struct;

    for i = 1:size(trial, 2)
        for j = 1:size(trial, 1)

            spikeData = trial(j, i).spikes; % spikes: neurons x time points
            numNeurons = size(spikeData, 1);
            numTimePoints = size(spikeData, 2);
            newTimePoints = 1: groupSize: numTimePoints + 1; 
            binnedSpikes = zeros(numNeurons, numel(newTimePoints) - 1);

            for k = 1:numel(newTimePoints) - 1
                binnedSpikes(:, k) = sum(spikeData(:, newTimePoints(k):newTimePoints(k+1) - 1), 2);
            end

            if shouldSquare
                binnedSpikes = sqrt(binnedSpikes);
            end

            processedTrial(j, i).spikes = binnedSpikes;
        end
    end
    
end


function finalTrial = calculateFiringRates(processedTrial, groupSize, windowSize)

    finalTrial = struct;
    window = 10 * (windowSize / groupSize);
    normStd = windowSize / groupSize;
    alpha = (window - 1) / (2 * normStd);
    temp1 = -(window - 1) / 2 : (window - 1) / 2;
    gaussianKernel = exp((-1 / 2) * (alpha * temp1 / ((window - 1) / 2)) .^ 2)';
    gaussianWindow = gaussianKernel / sum(gaussianKernel);
    
    for i = 1:size(processedTrial, 2)

        for j = 1:size(processedTrial, 1)
            
            smoothedRates = zeros(size(processedTrial(j, i).spikes, 1), size(processedTrial(j, i).spikes, 2));
            
            for k = 1:size(processedTrial(j, i).spikes, 1)
                
                smoothedRates(k, :) = conv(processedTrial(j, i).spikes(k, :), gaussianWindow, 'same') / (groupSize / 1000);
            end
            
            finalTrial(j, i).rates = smoothedRates;
        end
    end

end


function [predictedLabels] = classifyKNN(testData, trainData)

    trainMatrix = trainData';
    testMatrix = testData;
    trainSquared = sum(trainMatrix .* trainMatrix, 2);
    testSquared = sum(testMatrix .* testMatrix, 1);

    distanceMatrix = trainSquared(:, ones(1, length(testMatrix))) + testSquared(ones(1, length(trainMatrix)), :) - 2 * trainMatrix * testMatrix;
    distanceMatrix = distanceMatrix';

    k = 25;
    [~, sortedIndices] = sort(distanceMatrix, 2);
    nearestNeighbours = sortedIndices(:, 1:k);

    % determine the most frequent direction label from the k-nearest neighbours
    numTrain = size(trainData, 2) / 8;
    directionLabels = [1 * ones(1, numTrain), 2 * ones(1, numTrain), 3 * ones(1, numTrain), 4 * ones(1, numTrain), 5 * ones(1, numTrain), 6 * ones(1, numTrain), 7 * ones(1, numTrain), 8 * ones(1, numTrain)]';
    nearestLabels = reshape(directionLabels(nearestNeighbours), [], k);
    predictedLabels = mode(mode(nearestLabels, 2));

end
