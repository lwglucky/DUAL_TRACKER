function [New_Sample_Set,New_Sample_probability]=assemble(Sample_Set,usetimes,Sample_probability,N,pos)

%�ҵ�usetimes��Ԫ��Ϊ0��Ҫ����Ԫ�ص�λ��
b=find(usetimes==0);
%�ҵ�usetimes��Ԫ�ش���1��Ҫ���и��Ƶ����ӵ�λ��
c=find(usetimes>1);
%�ҵ����ø��Ƶ����ӵ�λ��
d=find(usetimes==1);


%�Բ���Ҫ���Ƶ�����ֱ�ӽ����滻
k=1;
length_d=length(d);
while(k<=length_d)
        New_Sample_Set(d(k)).x=Sample_Set(d(k)).x;
        New_Sample_Set(d(k)).y=Sample_Set(d(k)).y;
        k=k+1;
end

%�ֱ��b,c�����������
length_b=length(b);
length_c=length(c);%��¼Ҫ��������λ�õ����鳤��

%����Ҫ���и��Ƶ������Ƚ���һ���滻
k=1;
while(k<=length_c)
        New_Sample_Set(c(k)).x=Sample_Set(c(k)).x;
        New_Sample_Set(c(k)).y=Sample_Set(c(k)).y;
        k=k+1;
end

for i=1:length_b
    New_Sample_Set(b(i)).x = pos(2);
    New_Sample_Set(b(i)).y = pos(1);
end
% i=1;
% j=1;
% while(i<=length_c)
%     while(usetimes(c(i))>1&&j<=length_b)
%         wi=Sample_probability(c(i))/(Sample_probability(c(i))+Sample_probability(b(j)));
%         wj=Sample_probability(b(j))/(Sample_probability(c(i))+Sample_probability(b(j)));
%         New_Sample_Set(b(j)).x=round(wi*Sample_Set(c(i)).x+wj*Sample_Set(b(j)).x);
%         New_Sample_Set(b(j)).y=round(wi*Sample_Set(c(i)).y+wj*Sample_Set(b(j)).y);
%         j=j+1;
%         usetimes(c(i))=usetimes(c(i))-1;
%     end
%     i=i+1;
% end
%��������ÿ�����ӵ�Ȩֵ���·���Ϊ1/N
New_Sample_probability = 1/N*ones(1,N);
