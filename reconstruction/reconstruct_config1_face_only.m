clear; clc; close all

%% USER SETTINGS
imgDir = fullfile(fileparts(mfilename("fullpath")), "faceViews");
outSize = [240 240];
useMedian = false;
showDebug = true;

smileXml = ""; % disabled

%% Load images
imds = imageDatastore(imgDir);
if numel(imds.Files) < 2
    error("Need at least 2 images in folder: %s", imgDir);
end

%% Detectors
faceDet = vision.CascadeObjectDetector("FrontalFaceCART");

% Separate eye detectors (gives individual boxes)
leftEyeDet  = vision.CascadeObjectDetector("LeftEye");
rightEyeDet = vision.CascadeObjectDetector("RightEye");

% Smile detector (optional)
smileDet = [];
if strlength(smileXml) > 0 && isfile(smileXml)
    smileDet = vision.CascadeObjectDetector(smileXml);
end

%% Find reference (ONLY needs a face)
refFace = [];
refFaceBB = [];
refEyes = [];     % may stay empty
refFile = "";

for k = 1:numel(imds.Files)
    I = readimage(imds, k);
    [faceCrop, faceBB_inOrig, eyePts] = detectFaceAndEyesRobust(I, faceDet, leftEyeDet, rightEyeDet, outSize);

    if ~isempty(faceCrop)
        refFace   = faceCrop;
        refFaceBB = faceBB_inOrig;
        refEyes   = eyePts;        % may be []
        refFile   = imds.Files{k};
        break;
    end
end

if isempty(refFace)
    error("Could not detect a usable FACE in any image. Use clearer images or a different face detector.");
end

%% Accumulate aligned faces
alignedFaces = [];
acc = zeros([outSize 3], "double");
count = 0;

for k = 1:numel(imds.Files)
    I = readimage(imds, k);

    [faceCrop, faceBB_inOrig, eyePts] = detectFaceAndEyesRobust(I, faceDet, leftEyeDet, rightEyeDet, outSize);
    if isempty(faceCrop)
        if showDebug, fprintf("Skip (no face): %s\n", imds.Files{k}); end
        continue
    end

    % ---- Alignment decision:
    % If BOTH current eyes and ref eyes exist => eye-based alignment
    if size(eyePts,1) == 2 && size(refEyes,1) == 2
        faceAligned = alignByTwoEyes(faceCrop, eyePts, refEyes, outSize);
        method = "eyes";
    else
        % fallback: face-only alignment using face bbox center+scale
        faceAligned = alignByFaceOnly(faceCrop, faceBB_inOrig, refFaceBB, outSize);
        method = "face-only";
    end

    if isempty(faceAligned)
        if showDebug, fprintf("Skip (alignment failed): %s\n", imds.Files{k}); end
        continue
    end

    if showDebug
        figure(100); clf
        montage({refFace, faceCrop, faceAligned}, "Size", [1 3]);
        title("Reference | Current Crop | Aligned (" + method + ")");
        drawnow
    end

    if useMedian
        alignedFaces(:,:,:,end+1) = im2single(faceAligned); %#ok<SAGROW>
    else
        acc = acc + im2double(faceAligned);
    end
    count = count + 1;
end

if count < 2
    error("Not enough aligned frames. Need >=2 face detections.");
end

%% Reconstruct
if useMedian
    recon = median(alignedFaces, 4);
    recon = im2uint8(recon);
else
    recon = acc / count;
    recon = im2uint8(recon);
end

%% Compare detections (face always; eyes/smile optional)
Iorig = imread(refFile);

figure;

% ---------- ORIGINAL ----------
subplot(1,2,1); imshow(Iorig); title("ORIGINAL");
hold on
bF0 = faceDet(Iorig);
drawBoxesColor(bF0, "g");

if ~isempty(bF0)
    [~,idx] = max(bF0(:,3).*bF0(:,4));
    faceBB = bF0(idx,:);
    faceCrop0 = imcrop(Iorig, faceBB);

    if ~isempty(faceCrop0)
        bLE = leftEyeDet(faceCrop0);  drawBoxesColor(offsetBoxes(bLE, faceBB), "r");
        bRE = rightEyeDet(faceCrop0); drawBoxesColor(offsetBoxes(bRE, faceBB), "r");

        if ~isempty(smileDet)
            % restrict smile search to lower half of the face crop
            [h,w,~] = size(faceCrop0);
            lower = imcrop(faceCrop0, [1, round(h*0.5), w, round(h*0.5)]);
            bSM = smileDet(lower);
            if ~isempty(bSM)
                bSM(:,2) = bSM(:,2) + round(h*0.5);
            end
            drawBoxesColor(offsetBoxes(bSM, faceBB), "b");
        end
    end
end
hold off

% ---------- RECONSTRUCTED ----------
subplot(1,2,2); imshow(recon); title("RECONSTRUCTED");
hold on
bFR = faceDet(recon);
drawBoxesColor(bFR, "g");

if ~isempty(bFR)
    [~,idx] = max(bFR(:,3).*bFR(:,4));
    faceBB = bFR(idx,:);
    faceCropR = imcrop(recon, faceBB);

    if ~isempty(faceCropR)
        bLE = leftEyeDet(faceCropR);  drawBoxesColor(offsetBoxes(bLE, faceBB), "r");
        bRE = rightEyeDet(faceCropR); drawBoxesColor(offsetBoxes(bRE, faceBB), "r");

        if ~isempty(smileDet)
            [h,w,~] = size(faceCropR);
            lower = imcrop(faceCropR, [1, round(h*0.5), w, round(h*0.5)]);
            bSM = smileDet(lower);
            if ~isempty(bSM)
                bSM(:,2) = bSM(:,2) + round(h*0.5);
            end
            drawBoxesColor(offsetBoxes(bSM, faceBB), "b");
        end
    end
end
hold off

figure; imshow(recon); title(sprintf("Reconstructed Face (%d frames stacked)", count));

fprintf("\nReference frame used: %s\n", refFile);
fprintf("Frames stacked: %d\n", count);


% ------------------------------------------------------------
% Helper functions
% ------------------------------------------------------------

function [faceOut, faceBB_inOrig, eyePtsOut] = detectFaceAndEyesRobust(I, faceDet, leftEyeDet, rightEyeDet, outSize)
    faceOut = [];
    faceBB_inOrig = [];
    eyePtsOut = [];

    if size(I,3) == 1, I = repmat(I,1,1,3); end

    b = faceDet(I);
    if isempty(b), return; end

    % choose largest face
    [~,idx] = max(b(:,3).*b(:,4));
    bb = b(idx,:);                      % [x y w h] in original
    faceBB_inOrig = bb;

    faceCrop = imcrop(I, bb);
    if isempty(faceCrop), return; end

    faceCrop = imresize(faceCrop, outSize);

    % Detect eyes inside the resized crop (optional)
    bLE = leftEyeDet(faceCrop);
    bRE = rightEyeDet(faceCrop);

    % pick best candidates (largest area) for each
    if ~isempty(bLE)
        [~,i] = max(bLE(:,3).*bLE(:,4));
        le = bLE(i,:);
        leftCenter = [le(1)+0.5*le(3), le(2)+0.5*le(4)];
    else
        leftCenter = [];
    end

    if ~isempty(bRE)
        [~,i] = max(bRE(:,3).*bRE(:,4));
        re = bRE(i,:);
        rightCenter = [re(1)+0.5*re(3), re(2)+0.5*re(4)];
    else
        rightCenter = [];
    end

    % Only accept if we got both eyes and left is actually left of right
    if ~isempty(leftCenter) && ~isempty(rightCenter)
        if leftCenter(1) < rightCenter(1)
            eyePtsOut = [leftCenter; rightCenter];
        else
            % swapped: fix order
            eyePtsOut = [rightCenter; leftCenter];
        end
    end

    faceOut = faceCrop;
end

function Iout = alignByFaceOnly(faceCrop, faceBB, refBB, outSize)
    % faceCrop is already resized to outSize, so "face-only" fallback is basically:
    % just return the crop (or you could add mild correction if desired).
    %
    % If you want a tiny correction based on bbox scale, you could do:
    % s = (refBB(3)+refBB(4)) / (faceBB(3)+faceBB(4));
    % but since we resize every crop to outSize anyway, scale is already normalized.
    Iout = faceCrop;
    if ~isequal(size(Iout,1), outSize(1)) || ~isequal(size(Iout,2), outSize(2))
        Iout = imresize(Iout, outSize);
    end
end

function Iout = alignByTwoEyes(Iin, eyePts, refEyes, outSize)
    L = eyePts(1,:);  R = eyePts(2,:);
    v = R - L;
    ang = atan2(v(2), v(1));

    Lr = refEyes(1,:); Rr = refEyes(2,:);
    vr = Rr - Lr;
    angr = atan2(vr(2), vr(1));

    dtheta = (angr - ang) * 180/pi;

    dist = norm(v);
    distr = norm(vr);
    if dist < 1e-6
        Iout = [];
        return
    end
    s = distr / dist;

    I1 = imrotate(Iin, dtheta, "bicubic", "crop");
    I2 = imresize(I1, s, "bicubic");

    [h,w,~] = size(Iin);
    center = [(w+1)/2, (h+1)/2];

    L1 = rotatePoint(L, center, dtheta);
    R1 = rotatePoint(R, center, dtheta);

    L2 = L1 * s;
    R2 = R1 * s;
    c2 = (L2 + R2)/2;

    cr = (Lr + Rr)/2;
    t = cr - c2;

    Rout = imref2d(outSize);
    T = [1 0 0;
         0 1 0;
         t(1) t(2) 1];

    Iout = imwarp(I2, affine2d(T), "OutputView", Rout);
end

function p2 = rotatePoint(p, center, deg)
    theta = deg*pi/180;
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    p2 = (R * (p - center)')' + center;
end

function drawBoxesColor(bboxes, colorChar)
    if isempty(bboxes), return; end
    for i = 1:size(bboxes,1)
        rectangle("Position", bboxes(i,:), "EdgeColor", colorChar, "LineWidth", 2);
    end
end

function b2 = offsetBoxes(b, faceBB)
    if isempty(b), b2 = b; return; end
    b2 = b;
    b2(:,1) = b2(:,1) + faceBB(1);
    b2(:,2) = b2(:,2) + faceBB(2);
end
