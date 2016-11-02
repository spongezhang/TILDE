clear variables; clc; close all;
dir_name = '../data/';
dataset_name = 'images';
image_list = load_image_list(dir_name, dataset_name);

for i = 1:numel(image_list)
    image = imread([dir_name dataset_name '/' image_list(i).name]);
    load([dir_name 'result/' image_list(i).name(1:end-4) '.mat']);
    outputs = x;
    clear x;

    outputs = permute(outputs,[2,3,1]);
    offset_x = (size(image,2)-size(outputs,2)*4)/2;
    offset_y = (size(image,1)-size(outputs,1)*4)/2;

    output_width = size(outputs,2);
    output_height = size(outputs,1);

    grid_x = (1:output_width)';
    grid_x = repmat(grid_x,1,output_height)*4+offset_x;
    grid_x = grid_x';

    grid_y = (1:output_height)';
    grid_y = repmat(grid_y,1,output_width)*4+offset_y;
    grid_x = grid_x;% - outputs(:,:,1);
    grid_y = grid_y;% - outputs(:,:,2);
    vote = zeros(size(grid_x));
    for j = 1:size(grid_x,1)
        for k = 1:size(grid_x,2)
            index_x = k-round(outputs(j,k,1)/4);
            index_y = j-round(outputs(j,k,2)/4);
            if index_x>=1&&index_x<=output_width&&index_y>=1&&index_y<=output_height
                vote(index_y,index_x) = vote(index_y,index_x)+1;
            end
        end
    end
    vote = reshape(vote,1,output_width*output_height);
    grid_x = reshape(grid_x,1,output_width*output_height);
    grid_y = reshape(grid_y,1,output_width*output_height);
    grid_x(vote<0) = [];
    grid_y(vote<0) = [];
    
    imagesc(image);
    axis off image
    hold on;
    plot(grid_x,grid_y,'g.');
    hold off;
    %pause;
    saveas(gcf,[dir_name 'grid/',...
        image_list(i).name(1:end-4),'.png']);
end


