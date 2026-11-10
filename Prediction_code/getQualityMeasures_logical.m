function [qualityMeasures,tp,fn,tn,fp] = getQualityMeasures_logical(result, GoldImage)
tmpBwImg = result;
testingAns = GoldImage;
tp = sum(tmpBwImg == 1 & testingAns ==1);
fn = sum(tmpBwImg == 0 & testingAns ==1);
tn = sum(tmpBwImg == 0 & testingAns ==0);
fp = sum(tmpBwImg == 1 & testingAns ==0);
GraySensitivity = tp/(tp+fn); %true positive rate
GraySpecificity = tn/(tn+fp); %true negative rate
GrayAccuracy = (tp+tn)/(tp+fn+tn+fp);
GrayDice = (2*tp)/((2*tp)+fp+fn);
Precision=tp/(tp+fp);
Recall=tp/(tp+fn);
 J = jaccard_region( result, GoldImage);
F1 = (2*Precision*Recall)/(Precision+Recall);
qualityMeasures = [GraySensitivity GraySpecificity GrayAccuracy GrayDice J];
end