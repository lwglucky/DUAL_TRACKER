function w=weight(p,q,sita,m)
% simi=0;
% for i=1:m;
%  simi=simi+(p(i)*q(i))^0.5;  %%����q��Ŀ��ģ�����ɫֱ��ͼ��p����ĳһ���ĺ�ѡ�������ɫֱ��ͼ��ͨ��Bhattacharyyaϵ�����������ǵ����Ƴ̶�
% end
simi = sum((p.*q).^0.5);
d=(1-simi)^0.5;   %%��Ӧ��Bhattacharyya���룬��Ϊ�����ӵĹ۲�
%����ÿһ����ѡ������Ŀ����������Ƴ̶ȸ���Ȩ��
w=(1/(sita*(2*pi)^0.5))*exp(-(d^2)/(2*sita^2));   %%p��z/x��=(1/(sita*(2*pi)^0.5))*exp(-(d^2)/(2*sita^2))Ϊ�����ӵĹ۲���ʣ�������sitaͨ��ȡΪ0.2
