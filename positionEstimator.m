function [x, y] = positionEstimator(test_data, modelParameters)
    bin_size = 20;
    max_time = 320;  % 强制统一时间长度
    
    % 动态计算分箱数量
    num_bins = floor(max_time / bin_size);
    binned_test = zeros(98, num_bins);
    
    % 分箱处理（严格对齐训练长度）
    for bin = 1:num_bins
        start = (bin-1)*bin_size +1;
        finish = bin*bin_size;
        if finish > size(test_data.spikes,2)
            finish = size(test_data.spikes,2);
        end
        binned_test(:,bin) = mean(test_data.spikes(:,start:finish),2);
    end
    
    % 截断测试数据分箱数以匹配训练维度
    training_data = zeros(98, num_bins, 8);
    for ang = 1:8
        orig_cols = size(modelParameters.param(ang).firing_rates,2);
        valid_cols = min(orig_cols, num_bins);
        training_data(:,1:valid_cols,ang) = modelParameters.param(ang).firing_rates(:,1:valid_cols);
    end
    
    % 调用修正后的KNN函数
    angle = knn_predicted_angles(training_data, binned_test(:,1:valid_cols), 1);
    
    % 位置预测逻辑保持不变
    dt = bin_size;
    X = modelParameters.param(angle).dynamics;
    current_bin = floor(size(test_data.spikes,2)/dt);
    
    if current_bin >= size(X,2)
        x = X(1,end); y = X(3,end);
    else
        if isempty(test_data.decodedHandPos)
            x = test_data.startHandPos(1) + dt*X(2,1);
            y = test_data.startHandPos(2) + dt*X(4,1);
        else
            x = X(1,current_bin); y = X(3,current_bin);
        end
    end
end
   