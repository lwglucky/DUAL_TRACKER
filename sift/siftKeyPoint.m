function descriptor = siftKeyPoint( im )
%SIFTKEYPOINT 此处显示有关此函数的摘要
%   此处显示详细说明
    retScaleSpace = scaleSpace(im,4,3);
    octaveStack = retScaleSpace{1}; 
    accumSigmas = retScaleSpace{2}; 
	octaveDOGStack = calculateDog(octaveStack);
	keypoints = calculateKeypoints(octaveDOGStack, im);
    
    orientationDef = defineOrientation(keypoints, octaveDOGStack, ...
               octaveStack, im, accumSigmas);    
    descriptor = localDescriptor_v3(orientationDef, keypoints, ...
               accumSigmas, size(im,1)*2, size(im,2)*2); 
    
%     keypointDescriptor = keypoints{1};
%     keypt = [];
%     image = im;
%     for octave = 1:size(keypointDescriptor, 1)
%         %for each keypoints layer 
%         for kptLayer = 1:size(keypointDescriptor,2)
%             [rowKpt colKpt] = find(keypointDescriptor{octave,kptLayer} == 1);
%             if(octave==1)
%                 rowKpt = round(rowKpt/2); 
%                 colKpt = round(colKpt/2); 
%             end
%             if(octave>2)                        
%                 rowKpt = rowKpt * (2^(octave-2)); 
%                 colKpt = colKpt * (2^(octave-2)); 
%             end
%             rowKpt(rowKpt==1) = 2;
%             colKpt(colKpt==1) = 2;
% 
%             keypt = [keypt ;[colKpt rowKpt ]];
%         end 
%     end 
end

