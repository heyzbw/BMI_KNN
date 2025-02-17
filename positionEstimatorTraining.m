function modelParameters = positionEstimatorTraining(training_data)
    dt = 20;
    modelParameters = struct('param',struct(),'bin_size',dt);
    
    % 保持原最短时间计算逻辑
    min_len = 1000*ones(1,8);
    for j=1:8
        for k=1:size(training_data,1)
            min_len(j) = min(min_len(j), size(training_data(k,j).spikes,2));
        end
        modelParameters.param(j).size_min = min_len(j);
    end
    
    % 动态计算分箱数量
    for j=1:8
        num_bins = floor(min_len(j)/dt);
        
        % 运动动力学计算
        dyn = zeros(6,num_bins);
        for dim=1:3
            pos = zeros(size(training_data,1),min_len(j));
            for k=1:size(training_data,1)
                pos(k,:) = training_data(k,j).handPos(dim,1:min_len(j));
            end
            avg_pos = mean(pos);
            vel = [diff(avg_pos)*0.5, 0];
            
            for b=1:num_bins
                idx = (b-1)*dt+1 : min(b*dt,min_len(j));
                dyn(2*dim-1,b) = mean(avg_pos(idx));
                dyn(2*dim,b) = mean(vel(idx));
            end
        end
        modelParameters.param(j).dynamics = dyn;
        
        % 发放率计算（维度对齐关键修正）
        fr = zeros(98,num_bins);
        for n=1:98
            spikes = zeros(size(training_data,1),min_len(j));
            for k=1:size(training_data,1)
                spikes(k,:) = training_data(k,j).spikes(n,1:min_len(j));
            end
            avg = mean(spikes);
            
            for b=1:num_bins
                idx = (b-1)*dt+1 : min(b*dt,min_len(j));
                fr(n,b) = mean(avg(idx));
            end
        end
        modelParameters.param(j).firing_rates = fr;
    end
end
    
    