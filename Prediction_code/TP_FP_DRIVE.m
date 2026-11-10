clc
clear all
close all
num=1;

path='C:\Users\40377084\Downloads\Seg-Net-Combined-DRIVE (1)\';

for ii=1:20
    fprintf('processing image number %d \n',ii);

    file = strcat(path,'Binary\',num2str(ii),'.png');    
    im=imread(file);
    gt=imread(strcat(path,num2str(ii),'_GT.png'));
    
    % figure,imshow(im),impixelinfo
    % figure,imshow(gt),impixelinfo
%     im=imresize(im,[256 256]);
%     gt=imresize(gt,[256 256]);
    im=logical(im(:,:,1));
    gt=logical(gt(:,:,1));
    
    ne=zeros(size(im,1), size(im,2), 3);
    for i=1:size(im,1)
        for j=1:size(im,2)
            if(im(i,j) > 0 && gt(i,j) == 1)
                ne(i,j,2)=255;
            elseif(im(i,j) > 0 && gt(i,j) == 0)
                ne(i,j,1)=255;
            elseif(im(i,j) == 0 && gt(i,j) ==1 )
                ne(i,j,3)=255;
            end
        end
    end
%     figure,imshow([ne gt]),impixelinfo
    ne;
    
    new_gt=zeros([size(gt,1) size(gt,2) 3]);
    new_gt(:,:,2)=gt;

%     figure,imshow([ne new_gt]),impixelinfo
    
    % path_save='D:\PhD Work\Swin Transformer\Visual Results\DDTI_Augmented_C_F_Transfer_Learning\1-Fold\TP_TF\';
%     nam=strcat(path_save,num2str(ii),'_',name,'.png');
%     name1=strcat(path,'TP_FP\', num2str(ii),'_GT.jpg');
        imwrite(ne,[path 'TP_FP\' num2str(ii) '_Pred.png']);
        imwrite(new_gt,[path 'TP_FP\' num2str(ii) '_GT.png']);
    
%     imwrite(new_gt,name1)

end