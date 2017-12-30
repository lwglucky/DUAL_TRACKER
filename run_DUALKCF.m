
%
%  High-Speed Tracking with Kernelized Correlation Filters
%
%  Joao F. Henriques, 2014
%  http://www.isr.uc.pt/~henriques/
%
%  Main interface for Kernelized/Dual Correlation Filters (KCF/DCF).
%  This function takes care of setting up parameters, loading video
%  information and computing precisions. For the actual tracking code,
%  check out the TRACKER function.
%
%  RUN_TRACKER
%    Without any parameters, will ask you to choose a video, track using
%    the Gaussian KCF on HOG, and show the results in an interactive
%    figure. Press 'Esc' to stop the tracker early. You can navigate the
%    video using the scrollbar at the bottom.
%
%  RUN_TRACKER VIDEO
%    Allows you to select a VIDEO by its name. 'all' will run all videos
%    and show average statistics. 'choose' will select one interactively.
%
%  RUN_TRACKER VIDEO KERNEL
%    Choose a KERNEL. 'gaussian'/'polynomial' to run KCF, 'linear' for DCF.
%
%  RUN_TRACKER VIDEO KERNEL FEATURE
%    Choose a FEATURE type, either 'hog' or 'gray' (raw pixels).
%
%  RUN_TRACKER(VIDEO, KERNEL, FEATURE, SHOW_VISUALIZATION, SHOW_PLOTS)
%    Decide whether to show the scrollable figure, and the precision plot.
%
%  Useful combinations:
%  >> run_tracker choose gaussian hog  %Kernelized Correlation Filter (KCF)
%  >> run_tracker choose linear hog    %Dual Correlation Filter (DCF)
%  >> run_tracker choose gaussian gray %Single-channel KCF (ECCV'12 paper)
%  >> run_tracker choose linear gray   %MOSSE filter (single channel)
%


function res= run_KCFMY(seq, res_path, bSaveImage)

    addpath(genpath('/home/ww/toolbox/'));
    addpath(('../../rstEval'));
    addpath(('./sift/'));
    addpath(('./partialFilter/'));
    close all



	%parameters according to the paper. at this point we can override
	%parameters based on the chosen kernel or feature type
	kernel.type = 'gaussian';
    feature_type='hog';
	
	features.gray = false;
	features.hog = false;
	
	padding = 1.5;  %extra area surrounding the target
	lambda = 1e-4;  %regularization
	output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
	
	switch feature_type
	case 'gray',
		interp_factor = 0.075;  %linear interpolation factor for adaptation

		kernel.sigma = 0.2;  %gaussian kernel bandwidth
		
		kernel.poly_a = 1;  %polynomial kernel additive term
		kernel.poly_b = 7;  %polynomial kernel exponent
	
		features.gray = true;
		cell_size = 1;
		
	case 'hog',
		interp_factor = 0.02;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
		
	otherwise
		error('Unknown feature.')
	end


	%assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')


	
		
	
		%running in benchmark mode - this is meant to interface easily
		%with the benchmark's code.
		
		%get information (image file names, initial position, etc) from
		%the benchmark's workspace variables

		target_sz = seq.init_rect(1,[4,3]);
		pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
		img_files = seq.s_frames;
		video_path = [];
        show_visualization=1;
		
		%call tracker function with all the relevant parameters
		[positions,duration]= tracker(video_path, img_files, pos, target_sz, ...
			padding, kernel, lambda, output_sigma_factor, interp_factor, ...
			cell_size, features, show_visualization);
		
		%return results to benchmark, in a workspace variable
		rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
		rects(:,3) = target_sz(2);
		rects(:,4) = target_sz(1);
		res.type = 'rect';
		res.res = rects;
        res.fps=(seq.len-1)/duration;
		assignin('base', 'res', res);
		
%      closematlabpool;
end

function  [pool] = startmatlabpool(size)  
    pool=[];  
    isstart = 0;  
    if isempty(gcp('nocreate'))==1  
        isstart = 1;  
    end  
    if isstart==1  
    if nargin==0  
        pool=parpool('local');  
    else  
        try  
            pool=parpool('local',size);%matlabpool('open','local',size);  
        catch ce  
            pool=parpool('local');%matlabpool('open','local');  
            size = pool.NumWorkers;  
            display(ce.message);  
            display(strcat('restart. wrong  size=',num2str(size)));  
        end  
    end  
else  
    display('matlabpool has started');  
    if nargin==1  
        closematlabpool;  
        startmatlabpool(size);  
    else  
        startmatlabpool();  
    end  
    end
end



function [] = closematlabpool  
if isempty(gcp('nocreate'))==0  
    delete(gcp('nocreate'));  
end  
end
