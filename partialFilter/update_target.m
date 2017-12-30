function [new_target_histgram,pre_probability]=update_target(target_histgram,Sample_histgram,Sample_probability,pre_probability,Estimate,N,v_count,loop,resample_judge)
    if(resample_judge==0)
        %使用直接选择排序从N个粒子中选择N/5个大权值粒子，得到这些粒子的位置
        n=N/5;
        [b,location]=sort(Sample_probability,'descend');
        %将这前n个大权值的粒子的权值存储在model_probability
        model_probability  = b(1:n);
        sum_model_probability=sum(model_probability);

        %权值归一化
        model_probability=model_probability./sum_model_probability;
        
        %求平均模板
        Sample_histgram = Sample_histgram(location(1:n),:);
        model_probability = repmat(model_probability',1,size(Sample_histgram,2));
        average_target = model_probability.*Sample_histgram;
        average_target = sum(average_target);
        
    %若是重采样则对所有的粒子求平均模板
    else
        model_probability = repmat(Sample_probability',1,size(Sample_histgram,2));
        average_target = model_probability.*Sample_histgram;
        average_target = sum(average_target);        
    end
    %得到更新的模板
    new_target_histgram=0.2*average_target+0.8*target_histgram;

    if(loop<=10)
      pre_probability(loop)=Estimate(loop).probability;
    else
        for k=1:9
            pre_probability(k)=pre_probability(k+1);
        end
        pre_probability(10)=Estimate(loop).probability;
end

