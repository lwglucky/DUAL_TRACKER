function [Sample_Set,after_prop]=reproduce(Sample_Set,vx,vy,image_boundary_x,image_boundary_y,I,N,sigma_x,sigma_y,runtime)
    after_prop=I;

    %ǰ��֡ͼ���ƽ���ٶ���Ϊ��ǰ֡���ٶ�
    %���У���һ֡�͵ڶ�֡�����������
    if(runtime==1)
        pvx=vx(3);
        pvy=vy(3);
    end
    if(runtime==2)
        pvx=(vx(2)+vx(3))/2;
        pvy=(vy(2)+vy(3))/2;
    end
    pvx=(((vx(1)+vx(2)+vx(3))/3));
    pvy=(((vy(1)+vy(2)+vy(3))/3));

    %�õ���˹�ֲ����� 
    rn=random('Normal',pvx,sigma_x,1,10*N);
    rn2=random('Normal',pvy,sigma_y,1,10*N);

%�õ�����״̬ת�Ʒ��̣������ӽ���ת��
    count=1;
    i=1;
    while count<=N
        current_x=Sample_Set(count).x;
        current_y=Sample_Set(count).y;
        rand_x=round(rn(i));
        rand_y=round(rn2(i));
        Sample_Set(count).x=current_x+rand_x;
        Sample_Set(count).y=current_y+rand_y;     
         count=count+1;
        i=i+1;
    end

%ʵʱ��ʾ������ÿһ֡�ľ���״̬
    for  i=1:N
           if Sample_Set(i).x<=0
               Sample_Set(i).x=1;
           end
           if Sample_Set(i).y<=0;
               Sample_Set(i).y=1;
           end
           if Sample_Set(i).x>image_boundary_x;
               Sample_Set(i).x=image_boundary_x;
           end
           if Sample_Set(i).y>image_boundary_y;
               Sample_Set(i).y=image_boundary_y;
           end
           after_prop(Sample_Set(i).y,Sample_Set(i).x,1)=255; 
           after_prop(Sample_Set(i).y,Sample_Set(i).x,2)=0;
           after_prop(Sample_Set(i).y,Sample_Set(i).x,3)=0;
    end;
end
 
