function [matched_point, res] = match_img(detect_method, descriptor_method, imgA, imgB)
%MATCH_IMG match imgA and imgB use given method
%
I1 = rgb2gray(imread(imgA));
I1 = imresize(I1,[360,200]);
I2 = rgb2gray(imread(imgB));
I2 = imresize(I2,[360,200]);
if detect_method == "FAST"
    points1 = detectFASTFeatures(I1);   
    points2 = detectFASTFeatures(I2);
elseif detect_method == "ME"
    points1 = detectMinEigenFeatures(I1);   
    points2 = detectMinEigenFeatures(I2);
elseif detect_method == "Corner"
    points1 = detectHarrisFeatures(I1);   
    points2 = detectHarrisFeatures(I2);
elseif detect_method == "SURF"
    points1 = detectSURFFeatures(I1);   
    points2 = detectSURFFeatures(I2);
elseif detect_method == "KAZE"
    points1 = detectKAZEFeatures(I1);   
    points2 = detectKAZEFeatures(I2);
elseif detect_method == "BRISK"
    points1 = detectBRISKFeatures(I1);   
    points2 = detectBRISKFeatures(I2);
elseif detect_method == "MSER"
    points1 = detectMSERFeatures(I1);   
    points2 = detectMSERFeatures(I2);
elseif detect_method == "ORB"
    points1 = detectORBFeatures(I1);   
    points2 = detectORBFeatures(I2);
else
    fprintf("not support detect method now!")
end

if descriptor_method == "HOG"   
    [f1,vpts1] = extractHOGFeatures(I1,points1);
    [f2,vpts2] = extractHOGFeatures(I2,points2);
elseif descriptor_method == "LBP"
    f1 = extractLBPFeatures(I1,'CellSize', [32,32]);
    f2 = extractLBPFeatures(I2,'CellSize', [32,32]);
elseif descriptor_method == "SURF"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','SURF');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','SURF');
elseif descriptor_method == "KAZE"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','KAZE');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','KAZE');
elseif descriptor_method == "FREAK"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','FREAK');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','FREAK');
elseif descriptor_method == "BRISK"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','BRISK');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','BRISK');
elseif descriptor_method == "ORB"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','ORB');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','ORB');
elseif descriptor_method == "Block"
    [f1,vpts1] = extractFeatures(I1,points1,'Method','Block');
    [f2,vpts2] = extractFeatures(I2,points2,'Method','Block');   
else
    fprintf("not support descriptor method now")
end
indexPairs = matchFeatures(f1,f2, 'MatchThreshold',20);
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));
%fprintf("dt: %s ds: %s matched point is: %d \n",detect_method, descriptor_method,matchedPoints1.Count)
thresh = 5;
if matchedPoints1.Count > thresh
    res = 1;
else
    res = 0;
end
matched_point = matchedPoints1.Count;
%figure;
%ax = axes;
%showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
%title(ax, 'Candidate point matches');
%legend(ax, 'Matched points 1','Matched points 2');

end

