%% getKeypoints_TILDEP.m --- 
% 
% Filename: getKeypoints_TILDEP.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:21:16 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:21:22 2015 (+0200)
%           By: Kwang
%     Update #: 1
% URL: 
% Doc URL: 
% Keywords: 
% Compatibility: 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Commentary: 
% 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Change Log:
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


function [keypts] = getKeypoints_CavariantPoint(img_info, p)
    fixed_scale = 16;
    dir_name = ['/Users/Xu/program/Image_Genealogy/code/Covariant_Feature_Detection/eval/' ...
        'vlbenchmakrs/vlbenchmakrs-1.0-beta/data/'];
    slash_pos = strfind(img_info.full_feature_prefix,'/');
    subset_name =img_info.full_feature_prefix((slash_pos(end-3)+1):(slash_pos(end-2)-1));
    image_name = img_info.full_feature_prefix((slash_pos(end)+1):end);
    feature_filename = [dir_name, 'feature_point/', 'vggAffineDataset/', subset_name, '/' , image_name, '.mat'];
    x = load(feature_filename);
    I = x.feature(:,6);
    J = x.feature(:,3);
    scale = x.feature(:,1);
    keypts = [J I zeros(size(I,1),2), ones(size(I,1),1), scale]';
    %keypts = mergeScoreImg2Keypoints(keypts, score_res);

end

function [] = parsavefilter(fname, score_res, binary_res)
    save(fname, 'score_res', 'binary_res');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_TILDEP.m ends here
