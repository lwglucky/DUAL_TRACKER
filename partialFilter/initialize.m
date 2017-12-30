function [Sample_Set,Sample_probability,Estimate,target_histgram]=initialize(x,y,hx,hy,H,S,V,N,image_boundary_x,image_boundary_y,v_count,new_sita)
       for i=1:1:N
        Sample_Set(i).x=x;
        Sample_Set(i).y=y;    
       end
% Sample_Set = repmat([x,y] , N , 1);
%�õ���HSV�ռ����ɫֱ��ͼ
target_histgram=histgram(x,y,hx,hy,H,S,V,image_boundary_x,image_boundary_y,v_count);

%Ԥ��Ŀ�������λ��
Estimate(1).x=x;
Estimate(1).y=y;

%����ģ��ʱʹ��
Estimate(1).histgram=target_histgram;
Estimate(1).probability=weight(Estimate(1).histgram,target_histgram,new_sita,v_count);


%����Ȩֵ��ʼ��
initial_probability=1/N;%weight(target_histgram,target_histgram,new_sita,v_count)
Sample_probability = repmat(initial_probability,N,1);
%     for i=1:N
%      Sample_probability(i)=initial_probability;       
%     end

