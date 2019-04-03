%% Read the image.
  I = imread('cameraman.tif');
  
%% Harras Corner
  corners = detectHarrisFeatures(I);
  [features, valid_corners] = extractFeatures(I, corners)
  figure; imshow(I); hold on
  plot(valid_corners);
  
 %% SURF
  points = detectSURFFeatures(I);
  [features, valid_points] = extractFeatures(I, points);
  figure; imshow(I); hold on;
  plot(valid_points.selectStrongest(10),'showOrientation',true);
 
 %% MSER
  regions = detectMSERFeatures(I);
  [features, valid_points] = extractFeatures(I,regions,'Upright',true);%HOG
  figure; imshow(I); hold on;
  plot(valid_points,'showOrientation',true);
 %% BRISK/FAST/Harris/KAZEF/MinEigen/MSER/SURF
 %detectBRISKFeatures detectFASTFeatures detectHarrisFeatures
 %detectKAZEFeatures detectMinEigenFeatures detectMSERFeatures detectSURFFeatures
 
 %% Match
  %indexPairs = matchFeatures(features1,features2)
  %[indexPairs,matchmetric] = matchFeatures(features1,features2)
  %[indexPairs,matchmetric] = matchFeatures(features1,features2,Name,Value)
 