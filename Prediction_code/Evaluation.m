clc; clear all; close all;

path='C:\Users\40377084\Downloads\LVS-DRIVE\';

for i=1:20
    fprintf('processing image number %d \n',i);
    gt=imread(strcat(path,num2str(i),'_GT.png'));
    gt=logical(gt(:,:,1));

    I=imread(strcat(path,num2str(i),'_Pred.png'));
    Newimg=im2double(I(:,:,1));

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
            out = imbinarize(Newimg,thresh(1));
        end
        out = logical(out);
    end

    out = bwareaopen(out, 50);

    [measures(i,:),~,~,~,~] = getQualityMeasures_logical(out(:), gt(:));
    
    % Compute clDice score for this image
    cldice_scores(i) = clDice(out, gt);

    imwrite(out,[path 'Binary\' num2str(i) '.png']);
end

ind=find(isnan(measures));
measures(ind)=1;

se = mean(measures(:,1)); sp = mean(measures(:,2));
acc = mean(measures(:,3)); dice = mean(measures(:,4));
J= mean(measures(:,5));
avg_cldice = mean(cldice_scores);

m=[acc dice J se sp avg_cldice]*100;

fprintf('Accuracy: %.2f%%, Dice: %.2f%%, Jaccard: %.2f%%, Sensitivity: %.2f%%, Specificity: %.2f%%, clDice: %.2f%%\n',...
    acc*100, dice*100, J*100, se*100, sp*100, avg_cldice*100);

% --- clDice function definition ---
function score = clDice(predMask, gtMask)
    % Skeletonize masks
    predSkeleton = bwmorph(predMask, 'skel', Inf);
    gtSkeleton = bwmorph(gtMask, 'skel', Inf);

    % Compute numerator and denominator for clDice
    numerator = 2 * (sum(predSkeleton(:) & gtMask(:)) + sum(gtSkeleton(:) & predMask(:)));
    denominator = sum(predSkeleton(:)) + sum(gtMask(:)) + sum(gtSkeleton(:)) + sum(predMask(:));

    % Handle division by zero
    if denominator == 0
        score = 1; % Perfect match if both empty
    else
        score = numerator / denominator;
    end
end
