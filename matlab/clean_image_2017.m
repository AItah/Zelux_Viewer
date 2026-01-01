function clean_image_2017
% MATLAB 2017 classic GUI:
% - Main figure: Original + Filtered
% - Separate FFT figure: Spectrum (clickable) + notch overlays
% - Automatic resizing of axes (ResizeFcn)

    % ---------- State ----------
    S = struct();
    S.hWaistEllipse = [];
    S.hWaistText    = [];
    S.Iorig = [];
    S.Iwork = [];
    S.Ifilt = [];
    S.notchPoints = zeros(0,2); % [u0 v0] centered frequency coords (one side)

    S.figMain = [];
    S.figFFT  = [];
    S.axOrig  = [];
    S.axFilt  = [];
    S.axSpec  = [];
    S.hSpecImg = [];

    % ---------- Create main figure ----------
    S.figMain = figure('Name','FFT Lowpass + Notch Removal (MATLAB 2017)', ...
        'NumberTitle','off','MenuBar','none','ToolBar','none', ...
        'Position',[80 80 1200 700], ...
        'CloseRequestFcn',@onCloseAll);


    % Axes (positions will be set by layoutMain())
    S.axOrig = axes('Parent',S.figMain,'Units','pixels');
    title(S.axOrig,'Original'); axis(S.axOrig,'image'); set(S.axOrig,'XTick',[],'YTick',[]);

    S.axFilt = axes('Parent',S.figMain,'Units','pixels');
    title(S.axFilt,'Filtered'); axis(S.axFilt,'image'); set(S.axFilt,'XTick',[],'YTick',[]);

    % ---------- Controls (parent = main figure) ----------
    S.btnLoad  = uicontrol(S.figMain,'Style','pushbutton','String','Load Image', ...
        'Units','pixels','Callback',@onLoad);
    S.btnApply = uicontrol(S.figMain,'Style','pushbutton','String','Apply', ...
        'Units','pixels','Callback',@onApply);
    S.btnSave  = uicontrol(S.figMain,'Style','pushbutton','String','Save', ...
        'Units','pixels','Callback',@onSave);
    S.btnClear = uicontrol(S.figMain,'Style','pushbutton','String','Clear Notches', ...
        'Units','pixels','Callback',@onClearNotches);

    S.btnAuto  = uicontrol(S.figMain,'Style','pushbutton','String','Auto Detect Notches', ...
        'Units','pixels','Callback',@onAutoDetect);

    S.popFilter = uicontrol(S.figMain,'Style','popupmenu', ...
        'String',{'Gaussian LP','Butterworth LP','Ideal LP'}, ...
        'Value',1,'Units','pixels');

    S.slCutoff = uicontrol(S.figMain,'Style','slider','Min',1,'Max',100,'Value',20, ...
        'Units','pixels','Callback',@onCutoffChanged);
    S.txtCutoff = uicontrol(S.figMain,'Style','text','String','20', ...
        'Units','pixels','HorizontalAlignment','left');

    S.edOrder = uicontrol(S.figMain,'Style','edit','String','2', ...
        'Units','pixels');

    S.chkNotch = uicontrol(S.figMain,'Style','checkbox','String','Enable notch', ...
        'Value',1,'Units','pixels');

    S.edNotchSigma = uicontrol(S.figMain,'Style','edit','String','8', ...
        'Units','pixels');

    S.txtNotchCount = uicontrol(S.figMain,'Style','text','String','Notches: 0', ...
        'Units','pixels','HorizontalAlignment','left');

    % Auto-detect params
    S.edPeaks   = uicontrol(S.figMain,'Style','edit','String','12','Units','pixels');
    S.edDcRad   = uicontrol(S.figMain,'Style','edit','String','35','Units','pixels');
    S.edMinDist = uicontrol(S.figMain,'Style','edit','String','12','Units','pixels');
    S.edBgSigma = uicontrol(S.figMain,'Style','edit','String','10','Units','pixels');

    % Labels
    S.lblFilter    = uicontrol(S.figMain,'Style','text','String','Filter','Units','pixels','HorizontalAlignment','left');
    S.lblCutoff    = uicontrol(S.figMain,'Style','text','String','Cutoff %','Units','pixels','HorizontalAlignment','left');
    S.lblOrder     = uicontrol(S.figMain,'Style','text','String','Btw order','Units','pixels','HorizontalAlignment','left');
    S.lblNotchSig  = uicontrol(S.figMain,'Style','text','String','Notch \sigma(px)','Units','pixels','HorizontalAlignment','left');

    S.lblPeaks     = uicontrol(S.figMain,'Style','text','String','Peaks','Units','pixels','HorizontalAlignment','left');
    S.lblDcRad     = uicontrol(S.figMain,'Style','text','String','DC rad(px)','Units','pixels','HorizontalAlignment','left');
    S.lblMinDist   = uicontrol(S.figMain,'Style','text','String','MinDist','Units','pixels','HorizontalAlignment','left');
    S.lblBgSigma   = uicontrol(S.figMain,'Style','text','String','BG \sigma','Units','pixels','HorizontalAlignment','left');

    % Status
    S.txtStatus = uicontrol(S.figMain,'Style','text','String','Load an image to begin.', ...
        'Units','pixels','HorizontalAlignment','left');

    guidata(S.figMain,S);

    % Initial layout
    layoutMain();

    % ---------------- Callbacks ----------------

    function onCloseAll(~,~)
        S = guidata(S.figMain);
        if isfield(S,'figFFT') && ~isempty(S.figFFT) && ishandle(S.figFFT)
            delete(S.figFFT);
        end
        delete(S.figMain);
    end

    function onCutoffChanged(~,~)
        S = guidata(S.figMain);
        set(S.txtCutoff,'String',sprintf('%.0f',get(S.slCutoff,'Value')));
    end

    function onResizeMain(~,~)
        if ~ishandle(S.figMain), return; end
        layoutMain();
    end

    function layoutMain()
        S = guidata(S.figMain);

        figPos = get(S.figMain,'Position');
        W = figPos(3);
        H = figPos(4);

        margin = 12;
        topBarH = 90;
        statusH = 24;

        % Axes area
        axY = margin + statusH + margin;
        axH = max(50, H - topBarH - statusH - 3*margin);
        axW = max(50, floor((W - 3*margin)/2));

        set(S.axOrig,'Position',[margin axY axW axH]);
        set(S.axFilt,'Position',[2*margin+axW axY axW axH]);

        % Status at bottom
        set(S.txtStatus,'Position',[margin margin W-2*margin statusH]);

        % Top controls (y anchored to top)
        y1 = H - 35;
        y2 = H - 65;

        % Row 1 buttons
        x = margin;
        set(S.btnLoad, 'Position',[x y1 110 25]); x = x + 120;
        set(S.btnApply,'Position',[x y1 70  25]); x = x + 80;
        set(S.btnSave, 'Position',[x y1 70  25]); x = x + 80;
        set(S.btnClear,'Position',[x y1 110 25]); x = x + 120;
        set(S.btnAuto, 'Position',[x y1 150 25]); x = x + 160;

        % Row 2 settings (spread them but keep readable)
        x = margin;

        set(S.lblFilter,'Position',[x y2 45 18]); x = x + 45;
        set(S.popFilter,'Position',[x y2-2 120 22]); x = x + 130;

        set(S.lblCutoff,'Position',[x y2 60 18]); x = x + 60;
        set(S.slCutoff,'Position',[x y2 200 18]); x = x + 210;
        set(S.txtCutoff,'Position',[x y2 30 18]); x = x + 40;

        set(S.lblOrder,'Position',[x y2 60 18]); x = x + 60;
        set(S.edOrder,'Position',[x y2-2 40 22]); x = x + 55;

        set(S.chkNotch,'Position',[x y2-2 110 22]); x = x + 120;

        set(S.lblNotchSig,'Position',[x y2 90 18]); x = x + 90;
        set(S.edNotchSigma,'Position',[x y2-2 40 22]); x = x + 55;

        set(S.txtNotchCount,'Position',[x y2 120 18]);

        % Row 3 auto-detect params (under row 2)
        y3 = H - 85;
        x = 420;

        set(S.lblPeaks,'Position',[x y3 40 18]); x=x+40;
        set(S.edPeaks,'Position',[x y3-2 40 22]); x=x+55;

        set(S.lblDcRad,'Position',[x y3 70 18]); x=x+70;
        set(S.edDcRad,'Position',[x y3-2 40 22]); x=x+55;

        set(S.lblMinDist,'Position',[x y3 50 18]); x=x+50;
        set(S.edMinDist,'Position',[x y3-2 40 22]); x=x+55;

        set(S.lblBgSigma,'Position',[x y3 45 18]); x=x+45;
        set(S.edBgSigma,'Position',[x y3-2 40 22]);

        guidata(S.figMain,S);
    end

    function ensureFFTfigure()
        S = guidata(S.figMain);
        if isempty(S.figFFT) || ~ishandle(S.figFFT)
            S.figFFT = figure('Name','FFT Magnitude (click peaks to notch)', ...
                'NumberTitle','off','MenuBar','none','ToolBar','figure', ...
                'Position',[1400 120 700 700], ...
                'ResizeFcn',@onResizeFFT);

            S.axSpec = axes('Parent',S.figFFT,'Units','pixels');
            title(S.axSpec,'FFT magnitude (log)'); axis(S.axSpec,'image');
            set(S.axSpec,'XTick',[],'YTick',[]);
            axis(S.axSpec,'ij'); % image coordinates (y down)

            % Click support (classic figure)
            set(S.axSpec,'ButtonDownFcn',@onSpectrumClick,'HitTest','on');

            guidata(S.figMain,S);
            layoutFFT();
        else
            layoutFFT();
        end
    end

    function onResizeFFT(~,~)
        layoutFFT();
    end

    function layoutFFT()
        S = guidata(S.figMain);
        if isempty(S.figFFT) || ~ishandle(S.figFFT), return; end
        figPos = get(S.figFFT,'Position');
        W = figPos(3); H = figPos(4);
        margin = 20;
        set(S.axSpec,'Position',[margin margin W-2*margin H-2*margin]);
        drawnow;
    end

    function onLoad(~,~)
        S = guidata(S.figMain);

        [fn, fp] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif;*.*','Images'}, ...
            'Select an image');
        if isequal(fn,0)
            setStatus('Load cancelled.');
            return;
        end

        I = imread(fullfile(fp,fn));
        S.Iorig = I;
        S.Iwork = to01double(I);
        S.Ifilt = [];
        S.notchPoints = zeros(0,2);
        guidata(S.figMain,S);

        axes(S.axOrig); imshow(S.Iwork); title(S.axOrig,'Original');
        axes(S.axFilt); cla(S.axFilt); title(S.axFilt,'Filtered');

        ensureFFTfigure();
        showSpectrum(S.Iwork);

        refreshNotchCount();
        setStatus('Image loaded. FFT is in a dedicated window. Click bright peaks or use Auto Detect.');
    end

    function onApply(~,~)
        S = guidata(S.figMain);
        if isempty(S.Iwork)
            setStatus('Load an image first.');
            return;
        end

        cutoffPct   = get(S.slCutoff,'Value');
        filterIdx   = get(S.popFilter,'Value');
        orderN      = max(1, round(str2double(get(S.edOrder,'String'))));
        notchEnabled= get(S.chkNotch,'Value') ~= 0;
        notchSigma  = max(1, str2double(get(S.edNotchSigma,'String')));

        setStatus('Filtering...');
        drawnow;

        I = S.Iwork;
        if ndims(I)==2
            S.Ifilt = freqFilterOne(I, cutoffPct, filterIdx, orderN, notchEnabled, notchSigma, S.notchPoints);
        else
            out = zeros(size(I));
            for c = 1:size(I,3)
                out(:,:,c) = freqFilterOne(I(:,:,c), cutoffPct, filterIdx, orderN, notchEnabled, notchSigma, S.notchPoints);
            end
            S.Ifilt = out;
        end
        
        guidata(S.figMain,S);
        
        axes(S.axFilt); cla(S.axFilt);
        imshow(S.Ifilt); title(S.axFilt,'Filtered');
        axis(S.axFilt,'image'); set(S.axFilt,'XTick',[],'YTick',[]);
        
        % --- automatic beam waist measurement on filtered image ---
        R = measureBeamWaist(S.Ifilt);
        
        % draw overlay (ellipse at 1/e^2)
        drawWaistOverlay(S.axFilt, R);
        
        % update status
        thetaDeg = R.theta * 180/pi;
        setStatus(sprintf('Beam waist (1/e^2 radii): wMajor=%.2f px, wMinor=%.2f px, theta=%.1f deg, center=(%.1f, %.1f)', ...
            R.wMajor, R.wMinor, thetaDeg, R.cx, R.cy));
        
        % FFT in dedicated figure (as before)
        showSpectrum(S.Ifilt);
        
        setStatus('Done (waist computed).');


        setStatus('Done.');
    end

    function onSave(~,~)
        S = guidata(S.figMain);
        if isempty(S.Ifilt)
            setStatus('Nothing to save yet. Click Apply first.');
            return;
        end

        [fn, fp] = uiputfile({'*.png','PNG';'*.jpg','JPG';'*.tif','TIFF';'*.bmp','BMP'}, ...
            'Save filtered image');
        if isequal(fn,0)
            setStatus('Save cancelled.');
            return;
        end

        imwrite(toUint8(S.Ifilt), fullfile(fp,fn));
        setStatus('Saved.');
    end

    function onClearNotches(~,~)
        S = guidata(S.figMain);
        S.notchPoints = zeros(0,2);
        guidata(S.figMain,S);
        refreshNotchCount();
        if ~isempty(S.Iwork)
            ensureFFTfigure();
            showSpectrum(S.Iwork);
        end
        setStatus('Notches cleared.');
    end

    function onAutoDetect(~,~)
        S = guidata(S.figMain);
        if isempty(S.Iwork)
            setStatus('Load an image first.');
            return;
        end

        K       = max(1, round(str2double(get(S.edPeaks,'String'))));
        dcR     = max(1, round(str2double(get(S.edDcRad,'String'))));
        minDist = max(1, round(str2double(get(S.edMinDist,'String'))));
        bgSigma = max(1, str2double(get(S.edBgSigma,'String')));

        setStatus('Auto-detecting FFT peaks...');
        pts = detectNotches(S.Iwork, K, dcR, bgSigma, minDist);

        if isempty(pts)
            setStatus('No peaks detected. Try: increase Peaks, reduce DC rad, reduce BG sigma.');
            return;
        end

        S.notchPoints = unique([S.notchPoints; pts],'rows');
        guidata(S.figMain,S);

        refreshNotchCount();
        ensureFFTfigure();
        showSpectrum(S.Iwork);
        setStatus(sprintf('Auto notches added: %d (click Apply).', size(pts,1)));
    end

    function onSpectrumClick(~,~)
        S = guidata(S.figMain);
        if isempty(S.Iwork) || get(S.chkNotch,'Value')==0
            return;
        end
        ensureFFTfigure();

        cp = get(S.axSpec,'CurrentPoint');
        x = cp(1,1); y = cp(1,2);

        if ndims(S.Iwork)==3
            Ig = mean(S.Iwork,3);
        else
            Ig = S.Iwork;
        end
        [M,N] = size(Ig);
        cx = (N+1)/2;
        cy = (M+1)/2;

        u0 = round(x - cx);
        v0 = round(y - cy);

        if hypot(u0,v0) < 5
            setStatus('Notch too close to DC (center). Click a bright peak away from center.');
            return;
        end

        % Keep only one half-plane; symmetry handled later in filter
        if ~( (u0 > 0) || (u0==0 && v0>0) )
            u0 = -u0; v0 = -v0;
        end

        S.notchPoints(end+1,:) = [u0 v0];
        S.notchPoints = unique(S.notchPoints,'rows');
        guidata(S.figMain,S);

        refreshNotchCount();
        showSpectrum(S.Iwork);
        setStatus('Notch added. Click Apply to filter.');
    end

    % ---------------- Helpers ----------------

    function refreshNotchCount()
        S = guidata(S.figMain);
        set(S.txtNotchCount,'String',sprintf('Notches: %d', size(S.notchPoints,1)));
    end

    function setStatus(msg)
        S = guidata(S.figMain);
        set(S.txtStatus,'String',msg);
        drawnow;
    end

    function showSpectrum(I)
        S = guidata(S.figMain);
        ensureFFTfigure();

        if ndims(I)==3
            Ig = mean(I,3);
        else
            Ig = I;
        end
        Ig = double(Ig);

        F = fftshift(fft2(Ig));
        mag = log1p(abs(F));
        mag = mag ./ max(mag(:) + eps);

        axes(S.axSpec); cla(S.axSpec);
        S.hSpecImg = imagesc(mag);
        axis image; colormap(gray(256));
        title(S.axSpec,'FFT magnitude (log)'); set(S.axSpec,'XTick',[],'YTick',[]);
        axis(S.axSpec,'ij');

        % Click also on image object
        set(S.hSpecImg,'ButtonDownFcn',@onSpectrumClick,'HitTest','on');
        set(S.axSpec,'ButtonDownFcn',@onSpectrumClick,'HitTest','on');

        hold on;
        [M,N] = size(Ig);
        cx = (N+1)/2; cy = (M+1)/2;
        for k = 1:size(S.notchPoints,1)
            u0 = S.notchPoints(k,1);
            v0 = S.notchPoints(k,2);
            plot(cx + u0, cy + v0, 'wo', 'MarkerSize', 8, 'LineWidth', 1.5);
            plot(cx - u0, cy - v0, 'wo', 'MarkerSize', 8, 'LineWidth', 1.5);
        end
        hold off;
    end

    function Iout = freqFilterOne(Iin, cutoffPct, filterIdx, orderN, notchEnabled, notchSigma, notchPoints)
        [M,N] = size(Iin);
        F = fftshift(fft2(Iin));

        u = (-floor(N/2)):(ceil(N/2)-1);
        v = (-floor(M/2)):(ceil(M/2)-1);
        [U,V] = meshgrid(u,v);
        D = sqrt(U.^2 + V.^2);

        Dmax = min(M,N)/2;
        D0 = max(1, (cutoffPct/100) * Dmax);

        switch filterIdx
            case 1 % Gaussian LP
                H = exp(-(D.^2) / (2*(D0^2)));
            case 2 % Butterworth LP
                H = 1 ./ (1 + (D./D0).^(2*orderN));
            case 3 % Ideal LP
                H = double(D <= D0);
            otherwise
                H = exp(-(D.^2) / (2*(D0^2)));
        end

        if notchEnabled && ~isempty(notchPoints)
            for k = 1:size(notchPoints,1)
                u0 = notchPoints(k,1);
                v0 = notchPoints(k,2);
                Dk1 = (U - u0).^2 + (V - v0).^2;
                Dk2 = (U + u0).^2 + (V + v0).^2;
                Nk = (1 - exp(-Dk1/(2*notchSigma^2))) .* (1 - exp(-Dk2/(2*notchSigma^2)));
                H = H .* Nk;
            end
        end

        G = F .* H;
        Irec = real(ifft2(ifftshift(G)));
        Iout = min(max(Irec,0),1);
    end

    function pts = detectNotches(I, numPeaks, dcRadius, bgSigma, minDist)
        if ndims(I)==3
            Ig = mean(I,3);
        else
            Ig = I;
        end
        Ig = double(Ig);
        if max(Ig(:)) > 1
            Ig = Ig ./ max(Ig(:));
        end
        [M,N] = size(Ig);

        F = fftshift(fft2(Ig));
        Sspec = log1p(abs(F));
        Sspec = Sspec ./ max(Sspec(:) + eps);

        Sbg = gaussBlur2(Sspec, bgSigma);
        Shp = Sspec - Sbg;
        Shp(Shp < 0) = 0;

        [Y,X] = ndgrid(1:M,1:N);
        cx = (N+1)/2; cy = (M+1)/2;
        D = sqrt((X-cx).^2 + (Y-cy).^2);
        Shp(D <= dcRadius) = 0;

        border = 6;
        Shp(1:border,:) = 0; Shp(end-border+1:end,:) = 0;
        Shp(:,1:border) = 0; Shp(:,end-border+1:end) = 0;

        ptsRC = zeros(0,2);
        map = Shp;
        maxIters = numPeaks * 10;

        for it = 1:maxIters
            [val, idx] = max(map(:));
            if val <= 0, break; end
            [r,c] = ind2sub([M,N], idx);

            if isempty(ptsRC) || all(hypot(ptsRC(:,2)-c, ptsRC(:,1)-r) >= minDist)
                ptsRC(end+1,:) = [r c]; %#ok<AGROW>
                if size(ptsRC,1) >= numPeaks, break; end
            end

            r1 = max(1,r-minDist); r2 = min(M,r+minDist);
            c1 = max(1,c-minDist); c2 = min(N,c+minDist);
            map(r1:r2, c1:c2) = 0;
        end

        if isempty(ptsRC)
            pts = zeros(0,2); return;
        end

        u0 = round(ptsRC(:,2) - cx);
        v0 = round(ptsRC(:,1) - cy);

        keep = (u0 > 0) | (u0==0 & v0>0);
        pts = [u0(keep), v0(keep)];
    end

    function out = gaussBlur2(in, sigma)
        sigma = max(0.5, sigma);
        rad = max(1, ceil(3*sigma));
        x = -rad:rad;
        g = exp(-(x.^2)/(2*sigma^2));
        g = g / sum(g);
        out = conv2(conv2(in, g, 'same'), g', 'same');
    end

    function Id = to01double(I)
        if isinteger(I)
            Id = double(I) / double(intmax(class(I)));
        else
            Id = double(I);
            mx = max(Id(:)); mn = min(Id(:));
            if mx > 1 || mn < 0
                Id = (Id - mn) / (mx - mn + eps);
            end
        end
    end

    function U = toUint8(I)
        I = min(max(double(I),0),1);
        U = uint8(round(255*I));
    end
end



function R = measureBeamWaist(I)
% measureBeamWaist: ISO 11146-like 2D second-moment (D4?) beam size
% Returns 1/e^2 radii in pixels (w = 2*sigma), and principal axes rotation.

    % Grayscale intensity
    if ndims(I)==3
        Ig = mean(double(I),3);
    else
        Ig = double(I);
    end

    % Normalize-ish (not required, but helps numerical stability)
    mx = max(Ig(:));
    if mx > 0
        Ig = Ig ./ mx;
    end
    
    [M,N] = size(Ig);
    
    % Background estimate from borders
    br = max(1, round(0.05 * min(M,N))); % 5% border
    br = min([br, floor(M/2), floor(N/2)]); % safety
    
    b1 = Ig(1:br,:);
    b2 = Ig(end-br+1:end,:);
    b3 = Ig(:,1:br);
    b4 = Ig(:,end-br+1:end);
    
    border = [b1(:); b2(:); b3(:); b4(:)];
    bg = median(border);
    
    Ig = Ig - bg;
    Ig(Ig < 0) = 0;

    % Optional ROI masking (removes weak noise floor)
    peak = max(Ig(:));
    if peak <= 0
        R = struct('cx',NaN,'cy',NaN,'wMajor',NaN,'wMinor',NaN,'theta',NaN,'bg',bg,'peak',peak);
        return;
    end
    thr = 0.05 * peak; % 5% of peak
    mask = Ig > thr;
    if nnz(mask) > 50
        Iw = Ig;
        Iw(~mask) = 0;
    else
        Iw = Ig;
    end

    % Weighted centroid and covariance
    [X,Y] = meshgrid(1:N, 1:M);
    Wsum = sum(Iw(:)) + eps;

    cx = sum(sum(Iw .* X)) / Wsum;
    cy = sum(sum(Iw .* Y)) / Wsum;

    Xc = X - cx;
    Yc = Y - cy;

    varx  = sum(sum(Iw .* (Xc.^2))) / Wsum;
    vary  = sum(sum(Iw .* (Yc.^2))) / Wsum;
    covxy = sum(sum(Iw .* (Xc.*Yc))) / Wsum;

    C = [varx covxy; covxy vary];

    % Principal axes
    [V,D] = eig(C);
    d = diag(D);
    [dSorted, idx] = sort(d, 'descend');
    V = V(:,idx);

    sigmaMajor = sqrt(max(dSorted(1),0));
    sigmaMinor = sqrt(max(dSorted(2),0));

    % 1/e^2 radii: w = 2*sigma  (since I ~ exp(-x^2/(2?^2)) = exp(-2x^2/w^2))
    wMajor = 2*sigmaMajor;
    wMinor = 2*sigmaMinor;

    theta = atan2(V(2,1), V(1,1)); % radians

    R = struct();
    R.cx = cx; R.cy = cy;
    R.wMajor = wMajor;
    R.wMinor = wMinor;
    R.theta = theta;
    R.bg = bg;
    R.peak = peak;
end

function drawWaistOverlay(ax, R)
% Draw ellipse corresponding to 1/e^2 contour based on second moments.

    if any(isnan([R.cx R.cy R.wMajor R.wMinor R.theta]))
        return;
    end

    hold(ax,'on');

    t = linspace(0, 2*pi, 200);
    % ellipse in principal axis coords
    ex = R.wMajor * cos(t);
    ey = R.wMinor * sin(t);

    ct = cos(R.theta); st = sin(R.theta);
    x = R.cx + ct*ex - st*ey;
    y = R.cy + st*ex + ct*ey;

    plot(ax, x, y, 'y-', 'LineWidth', 2);
    plot(ax, R.cx, R.cy, 'y+', 'LineWidth', 2, 'MarkerSize', 10);

    txt = sprintf('wMajor=%.2f px, wMinor=%.2f px', R.wMajor, R.wMinor);
    text(ax, R.cx+10, R.cy+10, txt, 'Color','y', 'FontSize',10, 'FontWeight','bold');

    hold(ax,'off');
    title(S.axFilt, sprintf('Filtered | wMaj=%.2f px, wMin=%.2f px', R.wMajor, R.wMinor));

end
