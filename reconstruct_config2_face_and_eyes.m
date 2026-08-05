%% Face Reconstruction + Haar Comparison
clear; clc; close all;

%% =========================
% USER SETTINGS

imgDir    = fullfile(fileparts(mfilename("fullpath")), "faceViews"); % folder of face images
outSize   = [240 240];   % [H W] for normalized face crop
useMedian = false;       % true = median stacking, false = average stacking
showDebug = true;        % show alignment montage during processing

%% =========================
% LOAD IMAGES

imds = imageDatastore(imgDir);
if numel(imds.Files) < 2
    error("Need at least 2 images in folder: %s", imgDir);
end

%% =========================
% DETECTORS (HAAR CASCADES)

faceDet = vision.CascadeObjectDetector("FrontalFaceCART");
eyeDet  = vision.CascadeObjectDetector("EyePairBig"); % detects both eyes together

% faceDet.MergeThreshold = 4;

%% =========================
% PICK A REFERENCE FRAME (FIRST USABLE FACE+EYES)

refFace = [];
refEyes = [];
refFile = "";

for k = 1:numel(imds.Files)
    I = readimage(imds, k);
    [faceCrop, eyePts] = detectFaceAndEyes(I, faceDet, eyeDet, outSize);

    if ~isempty(faceCrop)
        refFace = faceCrop;
        refEyes = eyePts;           % [leftEye; rightEye] (in crop coords)
        refFile = imds.Files{k};
        break;
    end
end

if isempty(refFace)
    error("No usable face+eyes found in any image. Try different images or tweak detectors.");
end

%% =========================
% ACCUMULATE ALIGNED FACES

acc   = zeros([outSize 3], "double");
count = 0;

if useMedian
    alignedFaces = zeros([outSize 3 0], "single"); %#ok<NASGU>
end

for k = 1:numel(imds.Files)
    I = readimage(imds, k);
    [faceCrop, eyePts] = detectFaceAndEyes(I, faceDet, eyeDet, outSize);

    if isempty(faceCrop)
        if showDebug
            fprintf("Skip (no face/eyes): %s\n", imds.Files{k});
        end
        continue;
    end

    % Align current crop to the reference crop using the 2 eye points
    faceAligned = alignByTwoEyes(faceCrop, eyePts, refEyes, outSize);

    if isempty(faceAligned)
        if showDebug
            fprintf("Skip (bad alignment): %s\n", imds.Files{k});
        end
        continue;
    end

    if showDebug
        figure(100); clf;
        montage({refFace, faceCrop, faceAligned}, "Size", [1 3]);
        title("Reference | Current Crop | Aligned to Reference");
        drawnow;
    end

    if useMedian
        alignedFaces(:,:,:,end+1) = im2single(faceAligned); %#ok<SAGROW>
    else
        acc = acc + im2double(faceAligned);
    end

    count = count + 1;
end

if count < 2
    error("Not enough aligned frames (%d). Add more images or improve detection.", count);
end

%% =========================
% RECONSTRUCT ONE "CLEAN" FACE

if useMedian
    recon = median(alignedFaces, 4);
    recon = im2uint8(recon);
else
    recon = im2uint8(acc / count);
end

%% =========================
% COMPARE HAAR: ORIGINAL VS RECONSTRUCTED

Iorig = imread(refFile);

b0 = faceDet(Iorig);  % Haar on original full image
bR = faceDet(recon);  % Haar on reconstructed face (already cropped + aligned)

figure;
subplot(1,2,1);
imshow(Iorig);
title(sprintf("ORIGINAL (faces=%d)", size(b0,1)));
hold on; drawBoxes(b0); hold off;

subplot(1,2,2);
imshow(recon);
title(sprintf("RECONSTRUCTED (faces=%d)", size(bR,1)));
hold on; drawBoxes(bR); hold off;

figure;
imshow(recon);
title(sprintf("Reconstructed Face (%d frames stacked)", count));

fprintf("\nReference frame used: %s\n", refFile);
fprintf("Frames stacked: %d\n", count);
fprintf("Haar detections on original: %d\n", size(b0,1));
fprintf("Haar detections on reconstructed: %d\n", size(bR,1));

%% ============================================================
% LOCAL FUNCTIONS


function Iout = alignByTwoEyes(Iin, eyePts, refEyes, outSize)
% Aligns Iin (already cropped+resized) to the reference crop using:
% 1) rotation to match eye angle
% 2) scale to match inter-eye distance
% 3) translation to match eye midpoint

    L  = eyePts(1,:);         % [x y]
    R  = eyePts(2,:);
    v  = R - L;
    ang = atan2(v(2), v(1));  % current eye line angle

    Lr  = refEyes(1,:);
    Rr  = refEyes(2,:);
    vr  = Rr - Lr;
    angr = atan2(vr(2), vr(1)); % reference eye line angle

    dist  = norm(v);
    distr = norm(vr);

    if dist < 1e-6
        Iout = [];
        return;
    end

    dtheta = (angr - ang) * 180/pi; % degrees
    s      = distr / dist;          % scale factor

    % 1) rotate (crop keeps same size)
    I1 = imrotate(Iin, dtheta, "bicubic", "crop");

    % 2) resize
    I2 = imresize(I1, s, "bicubic");

    % Update eye points after rotate + scale, then translate to match ref midpoint
    [h,w,~] = size(Iin);
    center  = [(w+1)/2, (h+1)/2];

    L1 = rotatePoint(L, center, dtheta);
    R1 = rotatePoint(R, center, dtheta);

    L2 = L1 * s;
    R2 = R1 * s;
    c2 = (L2 + R2) / 2;       % current midpoint after rotate+scale
    cr = (Lr + Rr) / 2;       % reference midpoint

    t = cr - c2;              % translation in [x y]

    Rout = imref2d(outSize);
    T = [1 0 0;
         0 1 0;
         t(1) t(2) 1];

    Iout = imwarp(I2, affine2d(T), "OutputView", Rout);
end

function p2 = rotatePoint(p, center, deg)
% Rotates a 2D point p about center by deg degrees.
    theta = deg*pi/180;
    R = [cos(theta) -sin(theta);
         sin(theta)  cos(theta)];
    p2 = (R * (p - center)')' + center;
end

function [faceOut, eyePtsOut] = detectFaceAndEyes(I, faceDet, eyeDet, outSize)
% Detects largest face, crops+resizes it to outSize, then detects an eye-pair
% region and approximates left/right eye centers from that bbox.

    faceOut   = [];
    eyePtsOut = [];

    if size(I,3) == 1
        I = repmat(I,1,1,3);
    end

    % Face detection (largest bbox)
    b = faceDet(I);
    if isempty(b), return; end
    [~,idx] = max(b(:,3).*b(:,4));
    bb = b(idx,:);

    faceCrop = imcrop(I, bb);
    if isempty(faceCrop), return; end
    faceCrop = imresize(faceCrop, outSize);

    % Eye-pair detection in cropped face
    eyesBB = eyeDet(faceCrop);
    if isempty(eyesBB)
        return;
    end
    [~,eidx] = max(eyesBB(:,3).*eyesBB(:,4));
    e = eyesBB(eidx,:); % [x y w h]

    ex = e(1); ey = e(2); ew = e(3); eh = e(4);

    leftEye  = [ex + 0.30*ew, ey + 0.50*eh];
    rightEye = [ex + 0.70*ew, ey + 0.50*eh];

    faceOut   = faceCrop;
    eyePtsOut = [leftEye; rightEye];
end

function drawBoxes(bboxes)
% Draws green rectangles for each bbox [x y w h].
    for i = 1:size(bboxes,1)
        rectangle("Position", bboxes(i,:), "EdgeColor","g", "LineWidth",2);
    end
end
