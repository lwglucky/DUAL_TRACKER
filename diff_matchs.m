function dxy = diff_matchs( matches )
    dxys = [];
    for match = 1:size(matches,1)
        desc1 = matches(match).descriptorIm1; 
        desc2 = matches(match).descriptorIm2; 
        
        octave1 = desc1.octave; 
        xPos1 = desc1.kptX; 
        yPos1 = desc1.kptY; 
        
        if(octave1==1)
            xPos1 = round(xPos1/2); 
            yPos1 = round(yPos1/2); 
        end

        if(octave1>2)                        
            xPos1 = xPos1 * (2^(octave1-2)); 
            yPos1 = yPos1 * (2^(octave1-2)); 
        end
        
        octave2 = desc2.octave; 
        xPos2 = desc2.kptX; 
        yPos2 = desc2.kptY; 
        
        if(octave2==1)
            xPos2 = round(xPos2/2); 
            yPos2 = round(yPos2/2); 
        end

        if(octave2>2)                        
            xPos2 = xPos2 * (2^(octave2-2)); 
            yPos2 = yPos2 * (2^(octave2-2)); 
        end
        
        dxys = [dxys; xPos2-xPos1, yPos2-yPos1];
    end
    dxy = sum(dxys)/(size(dxys,1));
end

