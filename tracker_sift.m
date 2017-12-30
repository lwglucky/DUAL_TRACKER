function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)

	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	
    kcfnum =5;
    sub_target_sz = floor(target_sz / kcfnum);
    sub_window_sz = floor(sub_target_sz * (1 + padding));
    sub_output_sigma = sqrt(prod(sub_target_sz)) * output_sigma_factor / cell_size;
    sub_yf = fft2(gaussian_shaped_labels(sub_output_sigma, floor(sub_window_sz / cell_size)));
    sub_cos_window = hann(size(sub_yf,1)) * hann(size(sub_yf,2))';
    sub_pos = zeros(kcfnum,2);
    
	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision

	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
        end                

		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			patch = get_subwindow(im, pos, window_sz);
            
            objsize = (target_sz)*4/7;
            objsize = floor(objsize);
            objpatch = get_subwindow(im, pos, objsize);        
        
            nresponse=[];
            ndescriptor = siftKeyPoint(objpatch);
            if (size(ndescriptor,1)>2 && size(descriptor,1)>2)
                [matches , matchcount] = getMatches(descriptor, ndescriptor); 
                matchcount,frame
                if (matchcount>0)
                    dxy = diff_matchs(matches(1:matchcount))
                    npos = pos +  ceil(dxy) ;
                
                    patch = get_subwindow(im, npos, window_sz);
                    zf = fft2(get_features(patch, features, cell_size, cos_window));
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                    nresponse = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
                end
            end
%             top_left_pt = [pos(2)-objsize(2)/2,pos(1)-objsize(1)/2];
%             kp = kp + repmat(top_left_pt,size( kp, 1),1);
%             kp = floor(kp);
%             idx = sub2ind(size(im),kp(:,2) , kp(:,1))';
%             im(idx) = 0;
            patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            
%             if (~isempty(nresponse) && max(response(:))<max(nresponse(:)))
%                 response = nresponse;
%                 disp('revise');
%             end
            
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
		end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
        
        objsize = (target_sz)*4/7;
        objsize = floor(objsize);
        objpatch = get_subwindow(im, pos, objsize);        
        
        descriptor = siftKeyPoint(objpatch);
%         top_left_pt = [pos(2)-objsize(2)/2,pos(1)-objsize(1)/2];
%         kp = kp + repmat(top_left_pt,size( kp, 1),1);
%         kp = floor(kp);
%         idx = sub2ind(size(im),kp(:,2) , kp(:,1))';
%         im(idx) = 0;
%         patch = get_subwindow(im, pos, window_sz);
        
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
			stop = update_visualization(frame, box , im);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end

