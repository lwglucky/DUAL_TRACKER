clear;
clc;


%数据的初始化
%在HSV空间中，将三个颜色分量合成为一维特征向量时，一维向量的大小
v_count=193;
N=400;
%采样粒子的个数
n=52;%179;%400;%52;
%视频序列中的图像帧数
first=1;
%第一帧图像的名称序号
new_sita=0.20;
%（new_sita）^2表示颜色信息的高斯分布方差。
vx=[0,0,0];
vy=[0,0,0];
%得出目标的移动速度
runtime=0;%求取目标速度的时候用
struct_index=0;%存储结构体的指数
%产生随机粒子的方差
sigma_x=3.5;
sigma_y=3.5;
%求前10帧图像与目标模板的相似度
pre_probability=zeros(1,10);
%判断是否进行了重采样
resample_judge=0;

%得到目标模板的初始数据
video_path = 'D:\dataset\Benchmark\Basketball\img\';
outpath = 'd:\output\';
img_files = dir([video_path '*.jpg']);

I = imread([video_path img_files(1).name]);
imshow(I);
rect = getrect();
rect(1) = 198;  rect(2) = 213;
%153.0000  172
rect(3) = 34;  rect(4) = 81;
x1 = rect(1); 
x2 = rect(1) + rect(3);
y1 = rect(2);
y2 = rect(2) + rect(4);
%得到初始跟踪目标的中心坐标点
 x=round((x1+x2)/2);
 y=round((y1+y2)/2);
%得到描述目标轮廓的椭圆的长短半轴的平方
hx=((x2-x1)/3)^2;
hy=((y2-y1)/3)^2;
sizeimage=size(I);
image_boundary_x=int16(sizeimage(2));
image_boundary_y=int16(sizeimage(1));

%将第一帧用来选择被跟踪目标的图片存入指定的文件夹中
F = getframe;
mkdir([outpath 'result']);
image_source=strcat(outpath,'0001.jpg');
imwrite(F.cdata,image_source);

%在第一帧中手动选定的目标进行初始化操作
[H S V]=rgb_to_rank(I);
[Sample_Set,Sample_probability,Estimate,target_histgram]=initialize(...
    x,y,hx,hy,H,S,V,N,...
    image_boundary_x,image_boundary_y,...
    v_count,new_sita);
pre_probability(1)=Estimate(1).probability;


%从第二帧往后循环迭代的进行下去
for loop=2:numel(img_files)
    struct_index=struct_index+1;
    a=num2str(loop+first-1);
    b=[a,'.jpg'];
    b = [video_path img_files(loop).name];
    I=imread(b);
    [H,S,V]=rgb_to_rank(I);
    %产生随机粒子
    [Sample_Set,after_prop]=reproduce(Sample_Set,vx,vy, ... 
                image_boundary_x,image_boundary_y,...
                I,N,sigma_x,sigma_y,runtime);
    
    %得出被跟踪目标的在当前帧的预测位置
    [Sample_probability,Estimate,vx,vy,TargetPic,Sample_histgram]= ...
        evaluate(Sample_Set,Estimate,target_histgram,new_sita,...
        loop,after_prop,H,S,V,N,image_boundary_x,image_boundary_y,...
        v_count,vx,vy,hx,hy,Sample_probability);
    %模板更新时和重采用判断时，都要用到归一化的权值Sample_probability
    
    %模板跟新
    if(loop<=10)%前10帧属于特殊情况，需要额外进行处理
    %    sum_probability = sum(pre_probability(1:loop-1));
        mean_probability=mean(pre_probability(1:loop-1));% sum_probability/(loop-1);
    else%直接求取均值
        mean_probability=mean(pre_probability);
    end
    mean_probability;
    Estimate(loop).probability;
    if(Estimate(loop).probability>mean_probability)
        [target_histgram,pre_probability]=...
            update_target(target_histgram,Sample_histgram,...
                Sample_probability,pre_probability,Estimate,N,v_count,loop,resample_judge);
   %不进行模板更新，但是要对pre_probability进行更新操作
    else if(loop>10)
        for k=1:9
            pre_probability(k)=pre_probability(k+1);
        end
        pre_probability(10)=Estimate(loop).probability;
     else 
            pre_probability(loop)=Estimate(loop).probability;
        end
    end
    
    
    resample_judge=0;
    
    
    %判断是否需要重采样
    back_sum_weight=sum(Sample_probability.^2);
    sum_weight=1/back_sum_weight;
    if(sum_weight<N/2)
        %重采样过程
        usetimes=reselect(Sample_Set,Sample_probability,N);
        [Sample_Set,Sample_probability]=assemble(Sample_Set,usetimes,Sample_probability,N);%进行线性组合
        resample_judge=1;
    end
    
    
    %得到目标运动的轨迹
if(struct_index==1)
    routine.x=round(Estimate(loop).x);
    routine.y=round(Estimate(loop).y);
else
    routine(struct_index).x=round(Estimate(loop).x);
    routine(struct_index).y=round(Estimate(loop).y);
end
i=1;
j=1;
while(j<=struct_index)
    for new_x=routine(j).x-i:routine(j).x+i
       for new_y=routine(j).y:routine(j).y+i
            TargetPic(new_y,new_x,1)=0;
            TargetPic(new_y,new_x,2)=0;
            TargetPic(new_y,new_x,3)=255;
       end
    end   
    j=j+1;
end

%画出每一帧图像中跟踪目标的预测中心点
i=1;
for new_x=round(Estimate(loop).x)-i:round(Estimate(loop).x+i)
       for new_y=round(Estimate(loop).y)-i:round(Estimate(loop).y+i)
          TargetPic(new_y,new_x,1)=255;
          TargetPic(new_y,new_x,2)=255;
          TargetPic(new_y,new_x,3)=255;
       end
end

     imshow(TargetPic);
     F = getframe;
     image_source=strcat(outpath,num2str(loop),'.jpg');
     imwrite(F.cdata,image_source);  
end


%在灰度图中绘制出被跟踪目标的轨迹
im_routine=redraw_routine(image_boundary_x,image_boundary_y,routine,struct_index);
figure;
title('被跟踪目标的运动轨迹图');
hold on;
imshow(im_routine);

