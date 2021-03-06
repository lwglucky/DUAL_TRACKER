function plotDot = plotDot(image, row, col, octave)
        
        relRow = row; 
        relCol = col; 
        
        if(octave==1)
            relRow = round(row/2); 
            relCol = round(col/2); 
        end

        if(octave>2)                        
            relRow = row * (2^(octave-2)); 
            relCol = col * (2^(octave-2)); 
        end
        if(relRow==1)
            relRow = 2; 
        end
        if(relCol==1)
            relCol = 2; 
        end
        if (size(image,3)==3)
            image(relRow,relCol,1) = 255; 
            image(relRow,relCol,2) = 255; 
            image(relRow,relCol,3) = 0; 
            image(relRow-1,relCol-1,1) = 255; 
            image(relRow-1,relCol-1,2) = 255; 
            image(relRow-1,relCol-1,3) = 0; 
            image(relRow+1,relCol+1,1) = 255; 
            image(relRow+1,relCol+1,2) = 255; 
            image(relRow+1,relCol+1,3) = 0; 
            image(relRow-1,relCol+1,1) = 255; 
            image(relRow-1,relCol+1,2) = 255; 
            image(relRow-1,relCol+1,3) = 0; 
            image(relRow+1,relCol-1,1) = 255; 
            image(relRow+1,relCol-1,2) = 255; 
            image(relRow+1,relCol-1,3) = 0; 
        else 
            image(relRow,relCol) = 255;  
            image(relRow-1,relCol-1) = 255; 
            image(relRow+1,relCol+1) = 255; 
            image(relRow-1,relCol+1) = 255;  
            image(relRow+1,relCol-1) = 255;  
        end
                
        
        plotDot = image; 
    end 
