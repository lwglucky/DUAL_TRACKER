classdef sub_kcf
    properties
        pos = [0,0];
        cell_size = [];
        target_sz = [];
        window_sz = [];
        output_sigma ;
        padding ;
        lambda;        
        sub_yf;
        sub_cos_window;
        features,
    end
    
    methods
        function outpos = kcf(self,im)
            patch = get_subwindow(im, self.pos, self.window_sz);
            xf = fft2(get_features(patch, self.features, self.cell_size, self.cos_window));
        end
    end
end


