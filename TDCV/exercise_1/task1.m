% world coordinates
a = 0.1650;
b = 0.0630;
h = 0.0930;
wo_A = [0.5*b,-0.5*a,0.5*h];wo_B = [0.5*b,0.5*a,0.5*h];wo_C = [-0.5*b,0.5*a,0.5*h];
wo_D = [-0.5*b,-0.5*a,0.5*h];wo_E = [0.5*b,-0.5*a,-0.5*h];wo_F = [0.5*b,0.5*a,-0.5*h];
wo_G=[-0.5*b,0.5*a,-0.5*h];wo_H=[-0.5*b,-0.5*a,-0.5*h];

peak=2;      % peak value for sift points
front=1;      % calculate the front 6 triangles
back=1;       % calculate the back 4 triangles
visualization=1;  % visualization of the points

%%%%%%%%%%%%%%%%%    front side %%%%%%%%%%%%%%%%%%%%%％
if(front)
    % image 44 is chosen for front side with 6 triangles
     I = imread('DSC_9744.JPG') ;
     worldPoints = [wo_A;wo_B;wo_C;wo_F];
     % choose the image points for A,B,C,F
    imshow(I);
    coor = ginput(4)

    % camera intrinsic parameters
    % fx=fy=2960.37845, cx=1841.68855, cy=1235.23369
    IntrinsicMatrix = [2960.37845,0,0;0,2960.37845,0;1841.68855,1235.23369,1];
    % generate the camera parameters
    cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);
    % estimate the initial camera pose
    [worldOrientation,worldLocation] = estimateWorldCameraPose(coor,worldPoints,cameraParams,'MaxReprojectionError',1000000000000000);
    %calculate extrinstic matrix
    [rotationMatrix,translationVector] = cameraPoseToExtrinsics(worldOrientation,worldLocation);

    % selection of SIFT points in a specfic area
    BW = roipoly(I);
    I1=single(rgb2gray(I));
    % computing sift points
    [f1,d1] = vl_sift(I1,'PeakThresh',peak) ;
    %f1=int16(f1);
    for idx = 1:size(f1,2)
        if  BW(int16(f1(2,idx)),int16(f1(1,idx)))==false
               f1(1,idx)=[0];
        end
    end    
    d1( :, f1(1,:)==0 ) = [];  %columns
    f1( :, f1(1,:)==0 ) = [];  %columns

    % projection of the points to triangles 1 to 6
    worldPoints = pointsToWorld(cameraParams,rotationMatrix,translationVector,f1(1:2,:)');
    [row,col] = size(worldPoints);
    worldLocationBig = repmat(worldLocation,row,1);
    worldPoints(row,3)=0;
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_B,wo_F,'lineType','line');
    desp_1=d1(:,INTERSECT);
    points_triangle1=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_B,wo_C,wo_F,'lineType','line');
    desp_2=d1(:,INTERSECT);
    points_triangle2=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_B,wo_C,'lineType','line');
    desp_3=d1(:,INTERSECT);
    points_triangle3=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_E,wo_F,'lineType','line');
    points_triangle4=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_C,wo_G,wo_F,'lineType','line');
    points_triangle5=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_C,wo_D,'lineType','line');
    points_triangle6=xcoor(INTERSECT,:);

    % visualization
    xcoor1=[points_triangle1;points_triangle2;points_triangle3;points_triangle4;points_triangle5;points_triangle6;];
%     figure
%     scatter3(xcoor1(:,1),xcoor1(:,2),xcoor1(:,3));
%     xlabel('x');
%     ylabel('y');
%     zlabel('z');
end
%%%%%%%%%%%%%%%%%%%        back  side      %%%%%%%%%%%%%%%%%%%%% 
if(back)
    % image 48 is chosen for front side with 6 triangles
     I = imread('DSC_9748.JPG') ;
     worldPoints = [wo_C;wo_D;wo_A;wo_H];
     % choose the image points for A,B,C,F
    imshow(I);
    coor = ginput(4)

    % camera intrinsic parameters
    % fx=fy=2960.37845, cx=1841.68855, cy=1235.23369
    IntrinsicMatrix = [2960.37845,0,0;0,2960.37845,0;1841.68855,1235.23369,1];
    % generate the camera parameters
    cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);
    % estimate the initial camera pose
    [worldOrientation,worldLocation] = estimateWorldCameraPose(coor,worldPoints,cameraParams,'MaxReprojectionError',1000000000000000);
    %calculate extrinstic matrix
    [rotationMatrix,translationVector] = cameraPoseToExtrinsics(worldOrientation,worldLocation);

    % selection of SIFT points in a specfic area
    BW = roipoly(I);
    I1=single(rgb2gray(I));
    % computing sift points
    [f2,d2] = vl_sift(I1,'PeakThresh',peak) ;
    %f1=int16(f1);
    for idx = 1:size(f2,2)
        if  BW(int16(f2(2,idx)),int16(f2(1,idx)))==false
               f2(1,idx)=[0];
        end
    end    
    f2( :, f2(1,:)==0 ) = [];  %columns

    % projection of the points to triangles 7 to 10
    worldPoints = pointsToWorld(cameraParams,rotationMatrix,translationVector,f2(1:2,:)');
    [row,col] = size(worldPoints);
    worldLocationBig = repmat(worldLocation,row,1);
    worldPoints(row,3)=0;
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_E,wo_H,'lineType','line');
    points_triangle7=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_A,wo_D,wo_H,'lineType','line');
    points_triangle8=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_C,wo_D,wo_H,'lineType','line');
    points_triangle9=xcoor(INTERSECT,:);
    [INTERSECT, t, u, v, xcoor] = TriangleRayIntersection(worldLocation, worldLocationBig-worldPoints,wo_C,wo_G,wo_H,'lineType','line');
    points_triangle10=xcoor(INTERSECT,:);

    % visualization
     xcoor2=[points_triangle7;points_triangle8;points_triangle9;points_triangle10];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%xcoor_all=[xcoor1;xcoor2];
if (visualization)
    xcoor_all=[];
    if front
      xcoor_all=[xcoor_all;xcoor1];
    end
    if back
      xcoor_all=[xcoor_all;xcoor2];  
    end
    figure
    scatter3(xcoor_all(:,1),xcoor_all(:,2),xcoor_all(:,3));
    xlabel('x');
    ylabel('y');
    zlabel('z');
end


