% points_triangle1,2,3, sotres the sift points and desp_1,2,3 stores the descriptors for matching
peak=2;
points_world=[points_triangle1;points_triangle2;points_triangle3];
desp_world=[desp_1,desp_2,desp_3];
camera_pose=[];
% followed by task1
for i=1:24
   %i=1;
    I_detected=imread(strcat('DSC_97',num2str(50+i),'.JPG'));
    BW = roipoly(I_detected);
    I_single=single(rgb2gray(I_detected));
    % computing sift points
    [f,d] = vl_sift(I_single,'PeakThresh',peak) ;
    for idx = 1:size(f,2)
        if  BW(int16(f(2,idx)),int16(f(1,idx)))==false
               f(1,idx)=[0];
        end
    end    
    d( :, f(1,:)==0 ) = [];
    f( :, f(1,:)==0 ) = [];  %columns
    % matching
    [matches, scores] = vl_ubcmatch(d,desp_world);
    p_image=f(1:2,matches(1,:));
    p_world=points_world(matches(2,:),:);
% load to be tracked, select area and   match sift points in triangle
% use ransac for calculation
    confidence=0.9;
    outlier_prob=0.4;
    N=int16(log(1-confidence)/log(1-(1-outlier_prob)^4));
    outlier_crit=5;
    best_inlier_number=0;
    for i=1:N
        point_size=size(p_image,2);
        chosen_sample=randperm(point_size,4);
        world_sample=p_world(chosen_sample,:);
        image_sample=p_image(:,chosen_sample);
        try
            [worldOrientation2,worldLocation2,inlierIdx] = estimateWorldCameraPose(image_sample',world_sample,cameraParams,'MaxReprojectionError',100000000000000000);
            %[worldOrientation2,worldLocation2,inlierIdx] = estimateWorldCameraPose(imagePoints2,xcoor,cameraParams,'MaxReprojectionError',1000000000000000000);
            [rotationMatrix2,translationVector2] = cameraPoseToExtrinsics(worldOrientation2,worldLocation2);
            % calculate the reprojection of the 3d points
            p_image_calc = worldToImage(cameraParams,rotationMatrix2,translationVector2,p_world);
            p_image_calc = p_image_calc';
            distance=p_image-p_image_calc;
            distance=distance.^2;
            distance=sqrt(sum(distance,1));
            inlier_number=sum(distance<outlier_crit);
            if inlier_number>best_inlier_number
                best_inlier_number=inlier_number;
                best_inlierIdx=distance<outlier_crit;
                best_worldOrientation=worldOrientation2;
                best_worldLocation=worldLocation2; 
                best_rotationMatrix=rotationMatrix2;
                best_translationVector=translationVector2;
            end
        catch
            continue
        end
    end
    camera_pose=[camera_pose;best_worldOrientation,best_worldLocation'];
end    
for num=1:size(camera_pose,1)/3
     num1=3*num-2;
     num2=3*num;
     plotCamera('Location', camera_pose(3*num-2:3*num,4), 'Orientation', camera_pose(3*num-2:3*num,1:3), 'Size', 0.02, 'Color','red');
     hold on
     xlabel('x');
     xlim([-0.7 0.7])
     ylabel('y');
     zlabel('z');
     ylim([-0.6 0.6])
     zlim([-0.5 0.5])
     grid on
end


% store all the positions and visualize