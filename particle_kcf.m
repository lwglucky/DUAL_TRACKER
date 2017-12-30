function outkcf = particle_kcf( self , im , pos , frame )
        if frame > 1,
            patch = get_subwindow(im, self.pos, self.window_sz);
            zf = fft2(get_features(patch, self.features, self.cell_size, self.cos_window));						
            kzf = gaussian_correlation(zf, self.model_xf, self.kernel.sigma);
            response = real(ifft2(self.model_alphaf .* kzf));  %equation for fast detection

            [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                vert_delta = vert_delta - size(zf,1);
            end
            if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                horiz_delta = horiz_delta - size(zf,2);
            end
            self.pos = self.pos + self.cell_size * [vert_delta - 1, horiz_delta - 1];
        end
        if (frame==1) 
            self.pos = pos;
        end
        patch = get_subwindow(im, self.pos, self.window_sz);
        xf = fft2(get_features(patch, self.features, self.cell_size, self.cos_window));
        kf = gaussian_correlation(xf, xf, self.kernel.sigma);
        alphaf = self.yf ./ (kf + self.lambda);
        if frame == 1,  %first frame, train with a single image
            self.model_alphaf = alphaf;
            self.model_xf = xf;
        else
            %subsequent frames, interpolate model
            self.model_alphaf = (1 - self.interp_factor) * self.model_alphaf + self.interp_factor * alphaf;
            self.model_xf = (1 - self.interp_factor) * self.model_xf + self.interp_factor * xf;
        end
        outkcf = self;
end

