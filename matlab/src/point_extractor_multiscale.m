clear variables; clc; close all;
%dataset_name = 'vggAffineDataset';
%subsets = { 'bikes', 'trees', 'graf', 'boat', 'bark', 'wall', 'leuven', 'ubc'};
%subsets = {'bikes'};
% dataset_name = 'EFDataset';
% subsets = {'notredame','obama','paintedladies','rushmore','yosemite'};
dataset_name = 'WebcamDataset';
%subsets = {'Chamonix','Courbevoie','Frankfurt','Mexico','Panorama','StLouis'};
subsets = {'Mexico'};
dir_name = ['/Users/Xu/program/Image_Genealogy/code/Covariant_Feature_Detection/eval/' ...
        'vlbenchmakrs/vlbenchmakrs-1.0-beta/data/'];

fullPathFilter = '../filters/BestFilters_Standard/Original/MexicoMed.mat';
approxPathFilter = '../filters/BestFilters_Standard/Approx/MexicoMed24.mat';
addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
sRoot = strjoin(tmp,'/');
setup_path;
radius = 10;
det_thres = -0.1;
standard_circle = (0:1:10)';

detector = load(fullPathFilter,'res');
detector = detector.res;
param = detector.param;
delta = detector.delta; %backup old delta from previous res structure
detector = load(approxPathFilter);
detector.param = param;
detector.delta = delta;

pyramid_level = 1;
point_number = 150;
for set_index = 1:numel(subsets)
    subset = subsets{set_index};
    disp(set_index);
    image_list = load_image_list([dir_name 'datasets/' dataset_name '/'], subset);
    [s, mess, messid] = mkdir([dir_name 'tilde_p24_feature_point/' dataset_name '/' subset '/']);
    for i = 1:numel(image_list)
        
        image = imread([dir_name 'datasets/' dataset_name '/' subset '/' image_list(i).name]);
        if(size(image,3) == 1)
                image = repmat(image, [1 1 3]);
        end
        if isfield(param, 'fMultiScaleList')
            param = rmfield(param, 'fMultiScaleList');
        end
        factor = 1;
        feature = [];
        for j = 1:pyramid_level
            input_preproc_image = PreProcessTrainImage(image, param);
            [score, ~ ] = fastELLFiltering_approx(input_preproc_image, -inf, detector);

            fs = param.nBinSize;
            [score, max_img] = ApplyNonMax2Score(score, param);
            binary_res = max_img .* (score > -Inf);
            % Mutiplied fs with param.fScaling to consider scaling (25/04/2014 KMYI)
            binary_res(1:fs,:) = 0;
            binary_res(end-fs+1:end,:) = 0;
            binary_res(:,1:fs) = 0;
            binary_res(:,end-fs+1:end) = 0;

            idx = find(binary_res);
            [I,J] = ind2sub(size(binary_res),idx);
            features = [J I zeros(size(I,1),3) repmat(radius*2,size(I,1),1)]';
            features = mergeScoreImg2Keypoints(features, score);

            %sort by score
            [~,idx] = sort(-features(5,:));
            features = features(:,idx);
            %keep the 500 best
            features = features(:,1:min(size(features,2),round(point_number/factor^2)));

            feature_t = zeros(6,size(features,2));
            feature_t(1,:) = radius;
            feature_t(5,:) = radius;
            feature_t(3,:) = features(1,:);
            feature_t(6,:) = features(2,:);
            feature_t = feature_t'*factor;
            
            if isempty(feature)
                feature = feature_t;
            else
                feature = [feature;feature_t];
            end
            
            %# Create the gaussian filter with hsize = [5 5] and sigma = 2
            G = fspecial('gaussian',[5 5],sqrt(2));
            %# Filter it
            image = imfilter(image,G,'same');
            image = imresize(image, 1/sqrt(2));
            factor = factor*sqrt(2);
        end
        disp(size(feature,1));
        save([dir_name 'tilde_p24_feature_point/' dataset_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature');
    end
end

