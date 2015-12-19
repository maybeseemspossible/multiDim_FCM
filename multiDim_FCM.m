function [ Unow, center, now_obj_fcn ] = MultiDim_FCM( data, clusterNum )
%MULTIDIM_FCM for FCM in multidimensional data.
%data should be construct in a m*n matrix as input, m for the numuber of samples. n is the size of the features

if nargin < 2
    clusterNum = 2;   % number of cluster
end

[sample_num, feat_num] = size(data);%样本个数, 每个样本所含的属性数
expoNum = 2;%参数可变  控制聚类结果模糊程度的常数
epsilon = 0.001;%迭代终止的e
mat_iter = 100;   % number of maximun iteration

% 初始化U
Upre = rand(sample_num, clusterNum);
dep_sum = sum(Upre, 2);
dep_sum = repmat(dep_sum, [1,clusterNum]);
Upre = Upre./dep_sum;

center=zeros(clusterNum,feat_num); %初始聚类中心
%计算聚类中心点
for i=1:clusterNum
    center(i,:) = sum(repmat(Upre(:,i),[1,feat_num]).*data)/sum(Upre(:,i));
end

%计算代价函数
pre_obj_fcn = 0;
for i=1:clusterNum
    pre_obj_fcn = pre_obj_fcn + sum(sum((repmat(Upre(:,i),[1,feat_num]).*data - center(i)).^2));
end
fprintf('Initial objective fcn = %f\n', pre_obj_fcn);

for l = 1:mat_iter %最大迭代次数100
    
    %隶属函数的迭代修正
    Unow = zeros(size(Upre));
    for i=1:clusterNum
        for j=1:sample_num
            tmp=0;
            disUp = sum((data(j,:)-center(i,:)).^2);%样本j到第i类聚类中心的距离
            for k = 1:clusterNum
                disDn = sum((data(j,:)-center(k,:)).^2);%样本j到所有类聚类中心的距离总和
                tmp = tmp + (disUp/disDn)^(1/(expoNum-1));
            end
            Unow(j,i)=1/tmp;
        end
    end
    
    now_obj_fcn = 0;
    for i=1:clusterNum
        now_obj_fcn = now_obj_fcn + sum(sum((repmat(Unow(:,i),[1,feat_num]).*data - center(i)).^2));
    end
    fprintf('Iter = %d, Objective = %f\n', l, now_obj_fcn);
    
    if max(max(max(abs(Unow-Upre))))<epsilon || abs(now_obj_fcn - pre_obj_fcn)<epsilon 
        break;
    else %聚类中心的迭代修正
        Upre = Unow.^expoNum;

        for i=1:clusterNum
            center(i,:) = sum(repmat(Upre(:,i),[1,feat_num]).*data)/sum(Upre(:,i));
        end
        pre_obj_fcn = now_obj_fcn;
    end
end
end
