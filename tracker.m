function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)

	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
    resize_image = 0;
% 	if resize_image,
% 		pos = floor(pos / 2);
% 		target_sz = floor(target_sz / 2);
%     end


    runtime=0;%求取目标速度的时候用
	%在HSV空间中，将三个颜色分量合成为一维特征向量时，一维向量的大小
    v_count=193;
    N=400;%采样粒子的个数
    new_sita=0.20;%颜色信息的高斯分布方差。
    vx=[0,0,0]; %得出目标的移动速度
    vy=[0,0,0];
    sigma_x=5.5;%产生随机粒子的方差
    sigma_y=5.5;
    pre_probability=zeros(1,10);%求前10帧图像与目标模板的相似度
    %判断是否进行了重采样
    resample_judge=0;
    
	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
    
	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path , resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision

	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
        rgbim = im;
		
% 		if resize_image,
% 			im = imresize(im, 0.5);
%         end                
        if size(im,3) > 1,
			im = rgb2gray(im);
		end
		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)			
            
            [H,S,V]=rgb_to_rank(rgbim);
            %产生随机粒子
            [Sample_Set,after_prop]=reproduce(Sample_Set,vx,vy, ... 
                image_boundary_x,image_boundary_y,...
                rgbim,N,sigma_x,sigma_y,runtime);
            %得出被跟踪目标的在当前帧的预测位置
            [Sample_probability,Estimate,vx,vy,after_prop,Sample_histgram]= ...
                evaluate(Sample_Set,Estimate,target_histgram,...
                new_sita,frame,after_prop,H,S,V,N,...
                image_boundary_x,image_boundary_y,v_count,...
                vx,vy,hx,hy,Sample_probability);
            %模板更新时和重采用判断时，都要用到归一化的权值Sample_probability
    
            routine.x=round(Estimate(frame).x);
            routine.y=round(Estimate(frame).y);
            npos = [ routine.y , routine.x];
            patch = get_subwindow(im, npos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
% 			nresponse = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            pa = sum((Estimate(frame).histgram.*target_histgram).^0.5); %Estimate(loop).histgram.*;
            nresponse = real(ifft2(model_alphaf .* kzf));
            pab = max(nresponse(:));
 
            patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
 			response = real(ifft2(model_alphaf .* kzf));
%            response= max(real(ifft2(model_alphaf .* kzf)));  
%             pb = sum((real(ifft2(model_alphaf .* kzf)).*real(yf)).^0.5);
%             [pa,pb]
            if (max(nresponse(:))>max(response(:)))
                response = nresponse;
                pos = npos;
%                 disp(['revise' num2str(frame)]);
            end
            
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            
%             t_histgram=histgram(pos(2),pos(1),hx,hy,H,S,V,image_boundary_x,image_boundary_y,v_count);
%             Sample_histgram=zeros(N,v_count);
%             New_Sample_probability =  zeros(1,N);
%             for i=1:N
%                 Sample_histgram(i,:)=histgram((Sample_Set(i).x),(Sample_Set(i).y),hx,hy,H,S,V,image_boundary_x,image_boundary_y,v_count); 
%                 New_Sample_probability(i)=Sample_probability(i)*weight(target_histgram,Sample_histgram(i,:),new_sita,v_count);
%             end
%             Total_probability = sum(New_Sample_probability);
%             New_Sample_probability=New_Sample_probability./Total_probability;
%             tsampleset =  reshape(cell2mat(struct2cell(Sample_Set)),2,size(Sample_Set,2))';
%             x = New_Sample_probability*tsampleset(:,1);
%             y = New_Sample_probability*tsampleset(:,2);
%             t2_histgram=histgram(round(x),round(y),hx,hy,H,S,V,image_boundary_x,image_boundary_x,v_count);
%             pba=sum((t2_histgram.*t_histgram).^0.5); %weight(t2_histgram,t_histgram,new_sita,v_count);  
%             
%             if (pa^2*pba/(pb^2*pab)<1)
%                 pos = npos;
%                 disp(['revise' num2str(frame)]);
%             end
		end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
        if(frame==1)
            hx=(target_sz(2)/3)^2;
            hy=(target_sz(1)/3)^2;
            hx = target_sz(2)*2/4;
            hy = target_sz(1)*2/4;
            image_boundary_x=size(rgbim,2);
            image_boundary_y=size(rgbim,1);
%            rgb_patch = get_subwindow(rgbim, pos, window_sz);
            [H S V]=rgb_to_rank(rgbim);
            [Sample_Set,Sample_probability,Estimate,target_histgram]= ...
                initialize( pos(2),pos(1) , ...
                hx,hy,H,S,V,N, ...
                image_boundary_x,image_boundary_y ,v_count,new_sita);
            pre_probability(1)=Estimate(1).probability;
            after_prop = im;
        end
        
        if (frame>1)
  %          [pos(2)  round(Estimate(frame).x)  pos(1)  round(Estimate(frame).y)]
            Estimate(frame).x = pos(2);
            Estimate(frame).y= pos(1);
            Estimate(frame).histgram=histgram(pos(2),pos(1),hx,hy,H,S,V,image_boundary_x,image_boundary_y,v_count);
            Estimate(frame).probability=weight(target_histgram,Estimate(frame).histgram,new_sita,v_count); 
            vx(1:3) = [vx(1:2) pos(2)-Estimate(frame-1).x];
            vy(1:3) = [vy(1:2) pos(1)-Estimate(frame-1).y];
            for i=1:N
                %求每一个粒子确定的候选模板在HSV空间的颜色直方图
                Sample_probability(i)=Sample_probability(i)*weight(target_histgram,Sample_histgram(i,:),new_sita,v_count);
            end
            Sample_probability=Sample_probability./sum(Sample_probability);
        end
        
        %%粒子模板跟新
        if(frame<=10)%前10帧属于特殊情况，需要额外进行处理
            mean_probability = mean(pre_probability(1:frame-1));%sum_probability/(frame-1);
        else%直接求取均值
            mean_probability=mean(pre_probability);
        end
        if(Estimate(frame).probability>mean_probability)
            [target_histgram,pre_probability]= ...
                update_target(target_histgram,Sample_histgram,...
                Sample_probability,pre_probability,Estimate,...
                N,v_count,frame,resample_judge);
%不进行模板更新，但是要对pre_probability进行更新操作
        else
            if(frame>10)
                pre_probability(1:9) = pre_probability(2:10);
                pre_probability(10)=Estimate(frame).probability;
            else 
                pre_probability(frame)=Estimate(frame).probability;
            end
        end
        resample_judge=0;
        back_sum_weight=sum(Sample_probability.^2);
        sum_weight=1/back_sum_weight;
        if(sum_weight<N/2)
            %重采样过程
            usetimes=reselect(Sample_Set,Sample_probability,N);
            [Sample_Set,Sample_probability]= ...
                assemble(Sample_Set,usetimes,Sample_probability,N,pos);%进行线性组合
            resample_judge=1;
        end
        %%
        xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		kf = gaussian_correlation(xf, xf, kernel.sigma);
        
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization,
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box , after_prop);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end

