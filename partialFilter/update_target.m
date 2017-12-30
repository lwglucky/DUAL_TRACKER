function [new_target_histgram,pre_probability]=update_target(target_histgram,Sample_histgram,Sample_probability,pre_probability,Estimate,N,v_count,loop,resample_judge)
    if(resample_judge==0)
        %ʹ��ֱ��ѡ�������N��������ѡ��N/5����Ȩֵ���ӣ��õ���Щ���ӵ�λ��
        n=N/5;
        [b,location]=sort(Sample_probability,'descend');
        %����ǰn����Ȩֵ�����ӵ�Ȩֵ�洢��model_probability
        model_probability  = b(1:n);
        sum_model_probability=sum(model_probability);

        %Ȩֵ��һ��
        model_probability=model_probability./sum_model_probability;
        
        %��ƽ��ģ��
        Sample_histgram = Sample_histgram(location(1:n),:);
        model_probability = repmat(model_probability',1,size(Sample_histgram,2));
        average_target = model_probability.*Sample_histgram;
        average_target = sum(average_target);
        
    %�����ز���������е�������ƽ��ģ��
    else
        model_probability = repmat(Sample_probability',1,size(Sample_histgram,2));
        average_target = model_probability.*Sample_histgram;
        average_target = sum(average_target);        
    end
    %�õ����µ�ģ��
    new_target_histgram=0.2*average_target+0.8*target_histgram;

    if(loop<=10)
      pre_probability(loop)=Estimate(loop).probability;
    else
        for k=1:9
            pre_probability(k)=pre_probability(k+1);
        end
        pre_probability(10)=Estimate(loop).probability;
end

