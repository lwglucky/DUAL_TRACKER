%function that calculates the keypoints and filters out the ones on edges
%and with low contrast

function calculateKeypoints=calculateKeypoints(dogDescriptors, originalImage)
	
    contrastLimit = 0.03;
    isKeypoint = false; 
%	keypointsMap = cell(size(dogDescriptors,1), size(dogDescriptors,1)-2);
	keypointsMap = cell(size(dogDescriptors,1), size(dogDescriptors{1},4)-2);
	cant = 0;

%    size(dogDescriptors{1},3)
    
	for octave = 1:size(dogDescriptors,1)
		for layer = 2:(size(dogDescriptors{octave},4)-1)
%            keypointsMap{octave}{layer-1} = zeros(size(dogDescriptors{octave},1), size(dogDescriptors{octave},2));
            keypointsMap{octave,layer-1} = zeros(size(dogDescriptors{octave},1), size(dogDescriptors{octave},2));

			%each of the points are to be compared - if it takes too long, do it in C++ 
			for row = 2:size(dogDescriptors{octave},1)-1
				for column = 2:size(dogDescriptors{octave},2)-1
					%checks if it is maxima
                    isKeypoint = false; 
					if dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column,1,layer) && ...
                        dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column-1,1,layer+1) && ... 
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column-1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column-1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row+1,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row-1,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) > dogDescriptors{octave}(row,column,1,layer+1)
                    
                        if(keypointsMap{octave,layer-1}(row,column) == 0)
                            keypointsMap{octave,layer-1}(row,column) = 1; 
                            cant = cant + 1; 
                            isKeypoint = true; 
                        end 
                    end 
                    
                    
					%checks if it is minima 
					if dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column,1,layer) && ...
                        dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column-1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column+1,1,layer) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column-1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column+1,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column,1,layer-1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column-1,1,layer+1) && ... 
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column-1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column-1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row+1,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row-1,column+1,1,layer+1) && ...
						dogDescriptors{octave}(row,column,1,layer) < dogDescriptors{octave}(row,column,1,layer+1)
                    
                        if(keypointsMap{octave,layer-1}(row,column) == 0)
                            keypointsMap{octave,layer-1}(row,column) = 1; 
        					cant = cant + 1; 
                            isKeypoint = true; 
                        end 
                    end 
                    
                    
                    %checks the contrast - for now just the simplest way,
                    %without doing tailor expansion 
                    if(isKeypoint==true) 
                        
                        if(abs(dogDescriptors{octave}(row,column,1,layer))<contrastLimit)
                            keypointsMap{octave,layer-1}(row,column) = 0; 
                            isKeypoint = false; 
                            cant = cant - 1; 
                        end
                        
                    end
                
                    %checks the points on the ridges 
                    if(isKeypoint==true)
                        %used material to calculate derivatives: http://www.sci.utah.edu/~tolga/ece6532/Derivatives
                        %also wiki: http://en.wikipedia.org/wiki/Edge_detection
                        DerivativeYY = (dogDescriptors{octave}(row-1,column,1,layer) + ...
                               dogDescriptors{octave}(row+1, column, 1, layer) - ... 
                               2.0*dogDescriptors{octave}(row, column, 1, layer)); 
                           
                        DerivativeXX = (dogDescriptors{octave}(row,column-1,1,layer) + ...
                               dogDescriptors{octave}(row, column+1, 1, layer) - ... 
                               2.0*dogDescriptors{octave}(row, column, 1, layer)); 
                           
                           
                        DerivativeXY = (dogDescriptors{octave}(row-1,column-1,1,layer) + ...
                               dogDescriptors{octave}(row+1,column+1,1,layer) - ... 
                               dogDescriptors{octave}(row+1, column-1, 1, layer) - ... 
                               dogDescriptors{octave}(row-1, column+1, 1, layer))/4; 
                           
                        trTerm = DerivativeXX + DerivativeYY;
                        
                        DeterminantH = DerivativeXX * DerivativeYY - DerivativeXY*DerivativeXY; 
                        
                        if(DeterminantH<0)
                           % DeterminantH
                        end
                        
                        ratio = (trTerm*trTerm)/DeterminantH; 
                        
                        %r=10 is the value proposed in section 4.1 of Lowe
                        %paper, however experimentally 5 seems to be better
                        %ratio
                        threshold = ((5+1)^2)/5;
                        if(ratio>=threshold || DeterminantH<0)
                            keypointsMap{octave,layer-1}(row,column) = 0;                             
                            isKeypoint = false; 
                            cant = cant -1; 
                        end 
                    end 
                    
                end 
			end 
		end 
    end 
    
    
    
    
    withPointsImage = originalImage; 
    
    

    returnData = cell(4,1); 
    returnData{1} = keypointsMap;
    returnData{2} = withPointsImage;
    returnData{3} = dogDescriptors;
    %number of keypoints 
    returnData{4} = cant;
    
    
    
%    qtyDep
  %  cant
    
    
    
    calculateKeypoints = returnData; 
    
end 