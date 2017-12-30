function bin=histgram(x,y,hx,hy,H,S,V,image_boundary_x,image_boundary_y,v_count)
bin=zeros(1,v_count); 
matrix=1:1:v_count;
a=(hx+hy)^0.5;  %a is adjust parament 
pixel_distance=a;   %%a�ǹ�һ������
d=0;
r=0;
e=round((hx)^0.5);
f=round((hy)^0.5);

for pixel_x=(x-e):(x+e)
     for pixel_y=(y-f):(y+f)
            if(((x-pixel_x)^2/hx+(y-pixel_y)^2/hy)<=1&&pixel_x<=image_boundary_x&&pixel_x>0&&pixel_y<=image_boundary_y&&pixel_y>0)
                 bin_id=matrix(H(pixel_y,pixel_x)*4+S(pixel_y,pixel_x)*3+V(pixel_y,pixel_x)+1);
                 %(min=0*16+0*4+0+1=1,max=45*4+3*3+3+1=193)
                 pixel_distance=((double(x-pixel_x).^2)+(double(y-pixel_y).^2)).^0.5;   
                 r=pixel_distance/a; 
                 k=0.75*(1-r^2); 
                 bin(bin_id)=bin(bin_id)+k;   
                 d=d+k;                       
            end
     end
end
f=1/d;
time=1:1:v_count;
bin(time)=f*bin(time);