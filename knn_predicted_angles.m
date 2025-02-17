function predicted_angle = knn_predicted_angles(training_train, test_data, K)
    % 扩展测试数据维度以匹配训练数据
    test_data_expanded = repmat(test_data, [1, 1, size(training_train,3)]);
    
    % 计算欧氏距离平方
    dist = sum((training_train - test_data_expanded).^2, [1,2]);
    
    % 获取前K个最小距离的索引
    [~, idx] = mink(dist(:), K);  
    
    % 修正角度标签计算逻辑
    num_angles = size(training_train, 3);
    angle_labels = mod(idx-1, num_angles) + 1;
    
    % 统计频率
    freq = zeros(1, num_angles);
    for k = 1:K
        freq(angle_labels(k)) = freq(angle_labels(k)) + 1;
    end
    [~, predicted_angle] = max(freq);
end