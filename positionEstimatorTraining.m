% Last edit: 11/03/25
% Brain_X_SmaRt Group: Bowen Zhi, Xuanyun Qiu, Shiwei Liu, Ruiping Chi

function [modelParameters] = positionEstimatorTraining(training_data)

    numDirections = 8;
    groupSize = 20;
    binSize = 50;
    numTrials = length(training_data);
    preprocessedData = preprocessData(training_data, groupSize, 1);
    smoothedData = calculateFiringRates(preprocessedData, groupSize, binSize); 
    
    modelParameters = struct;
    startTime = 320;
    endTime = 560;
    counter = 1;
    timePoints = (startTime:groupSize:endTime) / groupSize;
    neuronsToRemove = {};
    
    % Compile firing data for PCA
    numNeurons = size(smoothedData(1, 1).rates, 1);
    for i = 1:numDirections
        for j = 1:numTrials
            for k = 1:endTime / groupSize
                firingData(numNeurons * (k-1) + 1 : numNeurons * k, numTrials * (i-1) + j) = smoothedData(j, i).rates(:, k);     
            end
        end
    end
    
    % Identify and remove neurons with low firing rates for numerical stability in PCA
    lowFiringNeurons = [];
    for neuron = 1:numNeurons
        avgRate = mean(mean(firingData(neuron:98:end, :)));
        if avgRate < 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end
    clear firingData
    neuronsToRemove{end+1} = lowFiringNeurons; 
    modelParameters.lowFirers = neuronsToRemove;
    
    % Process data for time windows
    for timeWindow = timePoints
        numNeurons = size(smoothedData(1, 1).rates, 1);
        for i = 1:numDirections
            for j = 1:numTrials
                for k = 1:timeWindow
                    firingData(numNeurons * (k-1) + 1:numNeurons * k, numTrials * (i-1) + j) = smoothedData(j, i).rates(:, k);     
                end
            end
        end
        removeIdx = [];
        for neuronIdx = lowFiringNeurons
            removeIdx = [removeIdx, neuronIdx:98:length(firingData)];
        end 
        firingData(removeIdx, :) = [];
        numNeurons = length(firingData) / (endTime / groupSize);
    
        % Supervised labeling for LDA
        directionLabels = [ones(1, numTrials), 2*ones(1, numTrials), 3*ones(1, numTrials), 4*ones(1, numTrials), ...
                           5*ones(1, numTrials), 6*ones(1, numTrials), 7*ones(1, numTrials), 8*ones(1, numTrials)];
    
        [principalComponents, ~] = performPCA(firingData);
    
        % Implement LDA on the PCA reduced firing data
        betweenClassMatrix = zeros(size(firingData, 1), numDirections);
        for i = 1:numDirections
            betweenClassMatrix(:, i) = mean(firingData(:, numTrials * (i-1) + 1 : i * numTrials), 2);
        end
        scatterBetween = (betweenClassMatrix - mean(firingData, 2)) * (betweenClassMatrix - mean(firingData, 2))';
        totalScatter = (firingData - mean(firingData, 2)) * (firingData - mean(firingData, 2))';
        scatterWithin = totalScatter - scatterBetween;
    
        pcaDimensions = 30;
        ldaDimensions = 6;
    
        [ldaVectors, ldaValues] = eig(((principalComponents(:, 1:pcaDimensions)' * scatterWithin * principalComponents(:, 1:pcaDimensions))^-1) * (principalComponents(:, 1:pcaDimensions)' * scatterBetween * principalComponents(:, 1:pcaDimensions)));
        [~, sortIdx] = sort(diag(ldaValues), 'descend');
        optimizedProjection = principalComponents(:, 1:pcaDimensions) * ldaVectors(:, sortIdx(1:ldaDimensions));
    
        classificationWeights = optimizedProjection' * (firingData - mean(firingData, 2));
    
        % Store the results for kNN classification
        modelParameters.classify(counter).wLDA_kNN = classificationWeights;
        modelParameters.classify(counter).dPCA_kNN = pcaDimensions;
        modelParameters.classify(counter).dLDA_kNN = ldaDimensions;
        modelParameters.classify(counter).wOpt_kNN = optimizedProjection;
        modelParameters.classify(counter).mFire_kNN = mean(firingData, 2);
        counter = counter + 1;
    end
    
    % Prepare data for PCR
    [xPos, yPos, xResampled, yResampled] = getResampledPositionData(training_data, numDirections, numTrials, groupSize);
    xTestInterval = xResampled(:, [startTime:groupSize:endTime] / groupSize, :);
    yTestInterval = yResampled(:, [startTime:groupSize:endTime] / groupSize, :);
    
    timeDivision = repelem(groupSize:groupSize:endTime, numNeurons);
    testTimes = startTime:groupSize:endTime;
    
    for i = 1:numDirections
        currentX = squeeze(xTestInterval(:, :, i));
        currentY = squeeze(yTestInterval(:, :, i));
    
        for j = 1:((endTime - startTime) / groupSize + 1)
            normalizedX = currentX(:, j) - mean(currentX(:, j));
            normalizedY = currentY(:, j) - mean(currentY(:, j));
    
            windowedFiring = firingData(timeDivision <= testTimes(j), directionLabels == i);
            [eigenVectors, ~] = performPCA(windowedFiring);
            pcaProjection = eigenVectors(:, 1:pcaDimensions)' * (windowedFiring - mean(windowedFiring, 1));
    
            % Regression coefficients for position estimation
            Bx = (eigenVectors(:, 1:pcaDimensions) * inv(pcaProjection * pcaProjection') * pcaProjection) * normalizedX;
            By = (eigenVectors(:, 1:pcaDimensions) * inv(pcaProjection * pcaProjection') * pcaProjection) * normalizedY;
    
            modelParameters.pcr(i, j).bx = Bx;
            modelParameters.pcr(i, j).by = By;
            modelParameters.pcr(i, j).fMean = mean(windowedFiring, 1);
            modelParameters.averages(j).avX = squeeze(mean(xPos, 1));
            modelParameters.averages(j).avY = squeeze(mean(yPos, 1));
        end
    end

end


function trialData = preprocessData(inputData, binSize, applySqrt)

    trialData = struct;
    for i = 1:size(inputData, 2)
        for j = 1:size(inputData, 1)
            allSpikes = inputData(j, i).spikes;
            numNeurons = size(allSpikes, 1);
            numPoints = size(allSpikes, 2);
            newTimePoints = 1:binSize:numPoints + 1;
            binnedSpikes = zeros(numNeurons, numel(newTimePoints) - 1);

            for k = 1:numel(newTimePoints) - 1
                binnedSpikes(:, k) = sum(allSpikes(:, newTimePoints(k):newTimePoints(k+1) - 1), 2);
            end

            if applySqrt
                binnedSpikes = sqrt(binnedSpikes);
            end

            trialData(j, i).spikes = binnedSpikes;
            trialData(j, i).handPos = inputData(j, i).handPos(1:2, :);
            trialData(j, i).bin_size = binSize;
        end
    end
end

function trialFinal = calculateFiringRates(processedData, groupSize, scaleWindow)

    trialFinal = struct;
    win = 10 * (scaleWindow / groupSize);
    normStd = scaleWindow / groupSize;
    alpha = (win - 1) / (2 * normStd);
    temp1 = -(win - 1) / 2 : (win - 1) / 2;
    gaussianTemp = exp((-1 / 2) * (alpha * temp1 / ((win - 1) / 2)) .^ 2)';
    gaussianWindow = gaussianTemp / sum(gaussianTemp);

    for i = 1:size(processedData, 2)
        for j = 1:size(processedData, 1)
            smoothedRates = zeros(size(processedData(j, i).spikes, 1), size(processedData(j, i).spikes, 2));

            for k = 1:size(processedData(j, i).spikes, 1)
                smoothedRates(k, :) = conv(processedData(j, i).spikes(k, :), gaussianWindow, 'same') / (groupSize / 1000);
            end

            trialFinal(j, i).rates = smoothedRates;
            trialFinal(j, i).handPos = processedData(j, i).handPos;
            trialFinal(j, i).bin_size = processedData(j, i).bin_size;
        end
    end
end

function [principalComp, evals, sortedIdx, eigVec] = performPCA(data)

    dataCentered = data - mean(data, 2);
    covarianceMatrix = dataCentered' * dataCentered / size(data, 2);
    [eigVec, evals] = eig(covarianceMatrix);
    [~, sortedIdx] = sort(diag(evals), 'descend');
    eigVec = eigVec(:, sortedIdx);
    principalComp = dataCentered * eigVec;
    principalComp = principalComp ./ sqrt(sum(principalComp .^ 2));
    evals = diag(evals(sortedIdx));
end

function [xn, yn, xrs, yrs] = getResampledPositionData(data, numDirections, numTrials, groupSize)

    trialCell = struct2cell(data);
    sizes = [];
    for i = 2:3:numTrials * numDirections * 3
        sizes = [sizes, size(trialCell{i}, 2)];
    end
    maxSize = max(sizes);

    xn = zeros(numTrials, maxSize, numDirections);
    yn = xn;

    for i = 1:numDirections
        for j = 1:numTrials
            xn(j, :, i) = [data(j, i).handPos(1, :), data(j, i).handPos(1, end) * ones(1, maxSize - sizes(numTrials * (i - 1) + j))];
            yn(j, :, i) = [data(j, i).handPos(2, :), data(j, i).handPos(2, end) * ones(1, maxSize - sizes(numTrials * (i - 1) + j))];
            xrs(j, :, i) = xn(j, 1:groupSize:end, i);
            yrs(j, :, i) = yn(j, 1:groupSize:end, i);
        end
    end
end