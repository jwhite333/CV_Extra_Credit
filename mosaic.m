%{
close all;
clear;
image_names = ["cast-left.jpg";"cast-right.jpg"];

accuracy_target = 0.8;
max_iters = 10000;
min_error_threshold = 1;

% Load images
images = [];
for image_num = 1:2
    image_name = char(fullfile('images',image_names(image_num,:)));
    temp = im2double(imread(image_name));
    images = cat(3,images,rgb2gray(temp));
end

% Get size
dim = size(images);
x_max = dim(1);
y_max = dim(2);
image_num = dim(3);

% Calculate derivatives
dx = [-1 0 1; -1 0 1; -1 0 1];
dy = [-1 -1 -1; 0 0 0; 1 1 1];

for counter= 1:image_num
    Ix(:,:,counter) = imfilter(images(:,:,counter), dx);
    Iy(:,:,counter) = imfilter(images(:,:,counter), dy);
end


% Classification parameters
k = 0.04;           % Value for Harris corner detector
f_sigma = 2;        % Gaussian filtering variance
tolerence = 1;      % Homography pixel error tolerence

% Classification
corners = zeros(x_max,y_max,image_num);
cxx = imgaussfilt(Ix.^2, f_sigma,'FilterSize',9);
cyy = imgaussfilt(Iy.^2, f_sigma,'FilterSize',9);
cxy = imgaussfilt(Ix.*Iy, f_sigma,'FilterSize',9);
R = zeros(x_max,y_max,image_num);
for counter = 1:image_num
    for x = 1:x_max
        for y = 1:y_max
            % Create M
            M = [cxx(x,y,counter), cxy(x,y,counter);cxy(x,y,counter),cyy(x,y,counter)];
            % Make decision
            R(x,y,counter) = det(M)-k*(trace(M))^2;
            
        end
    end
end

for counter = 1:image_num
    temp = max(max(R(:,:,counter)));
    maxR(1,counter) = 0.001*temp;
end


% Do non-max suppression on corners
for counter = 1:image_num
    for x = 2:x_max-1
        for y = 2:y_max-1
            if R(x,y,counter)>maxR(1,counter) && R(x,y,counter)>R(x-1,y-1,counter) ...
                && R(x,y,counter)>R(x-1,y,counter) && R(x,y,counter)>R(x-1,y+1,counter) ...
                && R(x,y,counter)>R(x,y-1,counter) && R(x,y,counter)>R(x,y+1,counter) ...
                && R(x,y,counter)>R(x+1,y-1,counter) && R(x,y,counter)>R(x+1,y,counter) ...
                && R(x,y,counter)>R(x+1,y+1,counter)
                    corners(x,y,counter) = 1;
            end
        end
    end
end

figure(1)
[r,c] = find(corners(:,:,1));
imshow(images(:,:,1));
hold on
plot(c,r,"yd");
hold off

figure(2)
[r,c] = find(corners(:,:,2));
imshow(images(:,:,2));
hold on
plot(c,r,"yd");
hold off

%% b Correspondences between 2 images

% Image 1
[row, col] = find(corners(:,:,1) > 0);
num = size(row);
matches1=[];
counter = 1;
for x = 1:num(1)
    if(row(x,1) > 3 && row(x,1) <= x_max-3) && (col(x,1) > 3 && col(x,1) <= y_max-3)
        g = images(row(x,1)-3:row(x,1)+3, col(x,1)-3:col(x,1)+3, 1);
        f = images(:,:,2);
        
        NCC = normxcorr2(g,f);
        [ypeak, xpeak] = find(NCC==max(NCC(:)));
        matches1(counter,1) = row(x,1);
        matches1(counter,2) = col(x,1);
        matches1(counter,3) = ypeak-3;
        matches1(counter,4) = xpeak-3;
        
        counter = counter +1;
    end
end

% Image 2
[row, col] = find(corners(:,:,2));
num = size(row);
matches2=[];
counter = 1;
for x = 1:num(1)
    if(row(x,1) >3 && row(x,1) <= x_max-3) && (col(x,1) > 3 && col(x,1) <=y_max -3)
        g = images(row(x,1)-3:row(x,1)+3, col(x,1)-3:col(x,1)+3, 2);
        f = images(:,:,1);
        
        NCC = normxcorr2(g,f);
        [ypeak, xpeak] = find(NCC==max(NCC(:)));
        matches2(counter,1) = row(x,1);
        matches2(counter,2) = col(x,1);
        matches2(counter,3) = ypeak-3;
        matches2(counter,4) = xpeak-3;
        counter = counter +1;
    end
    
end

% Remove false matches
true_matches = [];
total_matches1 = size(matches1);
for iter1 = 1:total_matches1(1)
    image1_x = matches1(iter1,1);
    image1_y = matches1(iter1,2);
    image2_x = matches1(iter1,3);
    image2_y = matches1(iter1,4);
    
    total_matches2 = size(matches2);
    for iter2 = 1:total_matches2(1)
        if abs(matches2(iter2,1)-matches1(iter1,3)) < tolerence
            if abs(matches2(iter2,2)-matches1(iter1,4)) < tolerence
                if abs(matches2(iter2,3)-matches1(iter1,1)) < tolerence
                    if abs(matches2(iter2,4)-matches1(iter1,2)) < tolerence
                        true_matches = [true_matches; image1_x,image1_y,image2_x,image2_y];
                    end
                end
            end
        end
    end
end

% Display correspondence pairs on images
figure(3);
%imshowpair(images(:,:,1),images(:,:,2),'montage');
%hold on
t=size(true_matches);
for index = 1:t(1) 
    plot([true_matches(index, 2) true_matches(index,4)+512], ...
        [true_matches(index, 1) true_matches(index,3)]);
    hold on
end
hold off

%% 8-Point algorithm with RANSAC
min_error = min_error_threshold*2;
loop_counter = 0;
F = [];
while min_error > min_error_threshold
   % Choose 8 points at random
   samples = randi([1,t(1)],1,8);
   
   % Create coefficients matrix
   C = zeros(8,9);
   for iter = 1:8
       x1xp1 = true_matches(samples(iter),1)*true_matches(samples(iter),3);
       x1yp1 = true_matches(samples(iter),1)*true_matches(samples(iter),4);
       x1 = true_matches(samples(iter),1);
       y1xp1 = true_matches(samples(iter),2)*true_matches(samples(iter),3);
       y1yp1 = true_matches(samples(iter),2)*true_matches(samples(iter),4);
       y1 = true_matches(samples(iter),2);
       xp1 = true_matches(samples(iter),3);
       yp1 = true_matches(samples(iter),4);
       C(iter,:) = [x1xp1,x1yp1,x1,y1xp1,y1yp1,y1,xp1,yp1,1];
   end
   
   % Get SVD
   [U,~,~] = svd(C'*C);
   F_tmp = U(:,9);
   F_tmp = [F_tmp(1:3,:)';F_tmp(4:6,:)';F_tmp(7:9,:)'];
  
   % Find error
   error = 0;
   for iter = 1:size(true_matches,1)
       pl = [true_matches(iter,1);true_matches(iter,2);1];
       pr = [true_matches(iter,3);true_matches(iter,4);1];
       error = error + pr'*F_tmp*pl;
   end
   if abs(error) < min_error
       min_error = abs(error);
       F = F_tmp;
   end
   
   % Stop loop after too long
   loop_counter = loop_counter+1;
end
%}

% Get correspondences / disparity
disparity_map = zeros(x_max,y_max,2);
for xl = 1:x_max
    for yl = 1:y_max
        
        % Set up equation for line
        xr_coeff = xl*F(1,1)+yl*F(1,2)+F(1,3);
        yr_coeff = xl*F(2,1)+yl*F(2,2)+F(2,3);
        const = xl*F(3,1)+yl*F(3,2)+F(3,3);
        
        % Get x min/max
        x_at_y1 = round((const-yr_coeff)/xr_coeff);
        x_at_ymax = round((const-y_max*yr_coeff)/xr_coeff);
        
        % NCC max
        ncc_max = 0;
        ncc_pos = [1,1];
        
        % Traverse line
        xr_points = linspace(x_at_y1,x_at_ymax,y_max-1)';
        yr_points = (const-(xr_points(:)*xr_coeff))/yr_coeff;
        for iter = 1:size(xr_points,1)
            
            xr = round(xr_points(iter));
            yr = round(yr_points(iter));
            
            % Get template and target
            if xr > 0 && yr > 0
                template = zeros(3,3);
                for tx = -1:1
                    for ty = -1:1
                        if (xl+tx >= 1) && (yl+ty >= 1) && ....
                                (xl+tx <= x_max) && (yl+ty <= y_max)
                            template(tx+2,ty+2) = images(xl+tx,yl+ty,1);
                        end
                    end
                end
                target = zeros(3,3);
                for tx = -1:1
                    for ty = -1:1
                        if (xr+tx >= 1) && (yr+ty >= 1) && ...
                                (xr+tx <= x_max) && (yr+ty <= y_max)
                            target(tx+2,ty+2) = images(xr+tx,yr+ty,2);
                        end
                    end
                end
                c = normxcorr2(template,target);
                ncc = c(3,3);
                if ncc > ncc_max
                    ncc_max = ncc;
                    ncc_pos = [xr,yr];
                end
            end
        end
        
        % Get disparity
        disparity_map(xl,yl,:) = images(xl,yl,:)-images(ncc_pos(1),ncc_pos(2),:);
    end
end
            %{
    find line eq
    search noisy line in image 2
    find point diff
    plot x,y disparity
%}
    