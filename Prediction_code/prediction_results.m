clc; clear all; close all;
% imds = imageDatastore('Extracted');
% pximds = imageDatastore('1st_manual\');
% I=imread('ISIC_0000000.png');
% path_gt='D:\PhD Work\Swin Transformer\Visual Results\BCDU_Swin_Sep_conv_BN_DiceLoss\';
path='C:\Users\40377084\Downloads\LF-Net_CHASE\';
% path_gt='C:\Users\shahz\Downloads\Mehwish Results-20230625T034411Z-001\Mehwish Results\';
for i=1:8
fprintf('processing image number %d \n',i);
    gt=imread(strcat(path,num2str(i),'_GT.png'));
    %     gt=imread(strcat(path_gt, num2str(i),'.tiff'));
    gt=logical(gt(:,:,1));
%     gt=imresize(gt,[256 256]);

%     fprintf('processing image number %d \n',ff);
    I=imread(strcat(path,num2str(i),'_Pred.png'));
    Newimg=im2double(I(:,:,1));

%     Newimg=imresize(Newimg,[256 256]);
    gn=gt;
    if(sum(sum(gn))==0)
        out=gt;
    else
        labels1 = zeros(size(gn(:)));
        labels1(gn==0)=-1; labels1(gn==1)=1;
        level = graythresh(Newimg);
        scores_od=Newimg;
        m=max(max(gn));
        if (m==0)
            out = imbinarize(Newimg,level);
        else
            [rec_p{i},prec_p{i},T_pr,aucpr(i)] = perfcurve(labels1(:),scores_od(:),1,'xCrit', 'reca', 'yCrit', 'prec');

            F = (2*(prec_p{i}.*rec_p{i}))./(prec_p{i}+rec_p{i});
            thresh = T_pr(F==max(F));
            max_F(i) = max(F);                                                                                                                                                                                                                                                                                                                                                     
            thresh=thresh;
            out = imbinarize(Newimg,thresh(1));
        end
        out = logical(out);
    end
    %
    %              siz=[512 512];
    %             out=imresize(out,siz);
    %             gt=logical(imresize(gt,siz));
    %         a=a+1;
    %     end

    out = bwareaopen(out, 50);
%     stats = regionprops(out, 'Area');
%     sz=size(stats);
%     s = zeros(sz)
%     for q=1:sz(1)
%     s(q)=stats(q).Area;
% 
%     end


%     out = max(s);
    %     se = strel('disk',10);
    %     out = imdilate(out,se);
    % %     out = imerode(out,se);

    %     out = imfill(out,[2 2],8);
    %     gt = logical(gt);
    %     se = strel('disk',2);
    %     gt = imdilate(gt,se);
    %     gt = imerode(gt,se);
    %         imwrite(gt,['GT_tiff\' num2str(i) '.tiff']);
  
    %             figure,imshow([gt out],[])
    [measures(i,:),~,~,~,~] = getQualityMeasures_logical(out(:), gt(:));

    imwrite(out,[path 'Binary\' num2str(i) '.png']);
    %
    %     imwrite(gt,[path '\B\' num2str(i) '_GT.tiff']);

    %         imwrite(gt,['visualRes/gt' num2str(i) '.jpg']);
    %         imwrite(Newimg,['visualRes/SegmentedImg' num2str(i) '.jpg']);
end

ind=find(isnan(measures));
measures(ind)=1;
se = mean(measures(:,1)); sp = mean(measures(:,2));
acc = mean(measures(:,3)); dice = mean(measures(:,4));
J= mean(measures(:,5));
%%

m=[acc dice J se sp]*100
