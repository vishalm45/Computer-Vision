function CMP9135_26439978()

    % ---------------- User paths / settings ----------------
    rootDir      = 'parachute';          % main folder
    imgDir       = fullfile(rootDir,'images');
    gtDir        = fullfile(rootDir,'GT');
    imgExt       = '*.png';               % change to '*.png' if needed
    gtExt        = '*.png';               % ground‑truth extension
    outMaskDir   = fullfile(rootDir,'results_masks');
    if ~exist(outMaskDir,'dir'); mkdir(outMaskDir); end

    % -------------------------------------------------------
    imgFiles = dir(fullfile(imgDir,imgExt));
    gtFiles  = dir(fullfile(gtDir,gtExt));

    if numel(imgFiles) ~= numel(gtFiles)
        error('Number of images and ground‑truth masks does not match.');
    end

    nImages  = numel(imgFiles);
    dscVals  = zeros(nImages,1);

    % ---------------- Main processing loop ----------------
    for k = 1:nImages
        % read RGB image
        I = im2double(imread(fullfile(imgDir,imgFiles(k).name)));

        % read corresponding ground truth and binarise
        GT = imread(fullfile(gtDir,gtFiles(k).name));
        if size(GT,3) > 1
            GT = rgb2gray(GT);
        end
        GT = imbinarize(GT);

        % ------- automatic parachute segmentation -------
        M = segment_parachute(I);

        % save mask (optional, for appendix)
        [~,baseName,~] = fileparts(imgFiles(k).name);
        imwrite(M, fullfile(outMaskDir, [baseName '_mask.png']));

        % ------- Dice Similarity Coefficient -------
        dscVals(k) = dice_coefficient(M,GT);

        fprintf('Image %2d: DSC = %.4f\n',k,dscVals(k));
    end

    % ----------------- statistics & plots -----------------
    meanDSC = mean(dscVals);
    stdDSC  = std(dscVals);
    fprintf('\nMean DSC: %.4f\n',meanDSC);
    fprintf('Std  DSC: %.4f\n',stdDSC);

    % bar chart of DSC values
    figure;
    bar(1:nImages, dscVals);
    xlabel('Image index');
    ylabel('Dice Similarity Coefficient');
    title('Parachute segmentation performance');
    ylim([0 1]);

    % ---- find 5 best and 5 worst for report appendices ----
    [sortedVals, idx] = sort(dscVals,'descend');
    bestIdx  = idx(1:5);
    worstIdx = idx(end-4:end);

    fprintf('\nBest 5 images (index : DSC)\n');
    disp([bestIdx, dscVals(bestIdx)]);

    fprintf('\nWorst 5 images (index : DSC)\n');
    disp([worstIdx, dscVals(worstIdx)]);

    % Optionally, display best and worst examples
    show_examples(imgDir, gtDir, imgFiles, gtFiles, bestIdx, 'Best results');
    show_examples(imgDir, gtDir, imgFiles, gtFiles, worstIdx, 'Worst results');

end


% =========================================================
%               Parachute segmentation
% =========================================================
function M = segment_parachute(I)
    % I: RGB in [0,1]
    % M: logical mask of parachute

    [h,w,~] = size(I);

    % ---------------- 1. Colour-based likelihood ----------------
    I = imgaussfilt(I,1);
    R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);

    chroma1 = max(R,G) - B;      % colourful vs grey
    chroma2 = abs(R-G);          % red/green difference
    colourScore = chroma1 + 0.6*chroma2;
    colourScore = mat2gray(colourScore);

    % ---------------- 2. Spatial prior (where parachute can be) ----------------
    % Vertical range: middle 60% (avoid top & bottom artefacts)
    topV    = round(0.20*h);
    botV    = round(0.80*h);

    % Horizontal range: middle 70%
    leftH   = round(0.15*w);
    rightH  = round(0.85*w);

    spatialMask = false(h,w);
    spatialMask(topV:botV, leftH:rightH) = true;
    colourScore(~spatialMask) = 0;

    % ---------------- 3. Initial adaptive threshold ----------------
    T  = adaptthresh(colourScore,0.40);        % a bit lower than before
    bw = imbinarize(colourScore,T);

    bw = bwareaopen(bw,40);                    % remove very small specks

    % ---------------- 4. Choose best seed component ----------------
    cc = bwconncomp(bw);
    if cc.NumObjects == 0
        M = false(h,w);  return;
    end

    stats = regionprops(cc,'Area','Eccentricity','Perimeter','Centroid');
    scores = zeros(cc.NumObjects,1);
    for i = 1:cc.NumObjects
        a   = stats(i).Area;
        ecc = stats(i).Eccentricity;
        p   = stats(i).Perimeter + eps;
        circ = 4*pi*a / (p^2);

        % area prior: parachute is small fraction of image
        areaPrior = exp(-((a/(0.006*numel(bw)) - 1).^2));  % peak near ~0.6% of image

        % compactness prior
        scores(i) = areaPrior * circ * (1 - ecc);
    end
    [~,bestIdx] = max(scores);

    seedMask = false(size(bw));
    seedMask(cc.PixelIdxList{bestIdx}) = true;

    % if seed too tiny, slightly dilate before refinement
    if nnz(seedMask) < 20
        seedMask = imdilate(seedMask, strel('disk',2));
    end

    % ---------------- 5. Region growing on colourScore ----------------
    % Use graydiffweight + imsegfmm to expand around best seed
    distMap = graydiffweight(colourScore, seedMask, 'RolloffFactor',0.7);
    % threshold on geodesic distance
    RG = imsegfmm(distMap, seedMask, 0.045);  % 0.03–0.06 is a reasonable window
    M  = RG > 0;

    % ---------------- 6. Final shape cleanup ----------------
    M = bwareaopen(M, 80);
    M = imfill(M,'holes');
    M = imopen(M, strel('disk',2));

    % keep most compact component after growth
    cc2 = bwconncomp(M);
    if cc2.NumObjects == 0
        M = false(size(M));  return;
    end
    stats2 = regionprops(cc2,'Area','Perimeter','Eccentricity');
    score2 = zeros(cc2.NumObjects,1);
    for i = 1:cc2.NumObjects
        a   = stats2(i).Area;
        p   = stats2(i).Perimeter + eps;
        circ = 4*pi*a/(p^2);
        ecc  = stats2(i).Eccentricity;
        score2(i) = circ*(1-ecc)*log(1+a);
    end
    [~,idx2] = max(score2);
    M = false(size(M));
    M(cc2.PixelIdxList{idx2}) = true;
end




% =========================================================
%               Dice coefficient
% =========================================================
function d = dice_coefficient(M,GT)
    M  = logical(M);
    GT = logical(GT);

    intersection = nnz(M & GT);
    d = 2 * intersection / (nnz(M) + nnz(GT) + eps);
end


% =========================================================
%           Helper to show example results
% =========================================================
function show_examples(imgDir, gtDir, imgFiles, gtFiles, indices, figTitle)

    figure('Name',figTitle);
    n = numel(indices);
    for i = 1:n
        k = indices(i);
        I  = im2double(imread(fullfile(imgDir,imgFiles(k).name)));
        GT = imread(fullfile(gtDir,gtFiles(k).name));
        if size(GT,3) > 1
            GT = rgb2gray(GT);
        end
        GT = imbinarize(GT);

        M  = segment_parachute(I);

        subplot(3,n,i);      imshow(I);  title(sprintf('Image %d',k));
        subplot(3,n,i+n);    imshow(GT); title('Ground truth');
        subplot(3,n,i+2*n);  imshow(M);  title('Segmentation');
    end
end
