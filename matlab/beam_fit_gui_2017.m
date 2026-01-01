function beam_fit_gui_2017()
%BEAM_FIT_GUI_2017 Classic GUI compatible with MATLAB R2017
    ini.path = "C:\WC\SelfEmployee_wc\2025\STED with Nir\Code\BaslerTool\data\cross_sections";
    S = struct();
    S.csvFile = '';
    S.results = table();
    S.fitOut  = [];
    S.imgFile = '';
    S.imgOrig = [];
    S.imgFilt = [];
    S.hImageFig = [];
    S.hFFTFig = [];
    S.filterMode = 'lowpass';
    S.filterCutLow = 20;
    S.filterCutHigh = 50;

    % ---- Figure ----
    S.hFig = figure('Name','Beam Profile Fitter (R2017)', ...
                    'NumberTitle','off', 'MenuBar','none', 'ToolBar','none', ...
                    'Position',[100 100 1100 650]);

    % ---- Controls area ----
    uicontrol('Style','text','Parent',S.hFig,'String','CSV file:', ...
              'HorizontalAlignment','left','Position',[20 610 60 18]);

    S.hFile = uicontrol('Style','edit','Parent',S.hFig,'String','', ...
                        'Enable','inactive','HorizontalAlignment','left', ...
                        'Position',[85 607 520 24]);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','Browse...', ...
              'Position',[615 607 90 24], 'Callback',@onBrowse);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','Use cross_sections.csv', ...
              'Position',[710 607 160 24], 'Callback',@onUseShared);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','2D Filter / FFT...', ...
              'Position',[880 607 200 24], 'Callback',@onFilter2D);

    uicontrol('Style','text','Parent',S.hFig,'String','Model:', ...
              'HorizontalAlignment','left','Position',[20 575 60 18]);

    S.hModel = uicontrol('Style','popupmenu','Parent',S.hFig, ...
                         'String',{'auto','gaussian','lorentzian'}, ...
                         'Value',1, 'Position',[85 572 120 24]);

    S.hPlotDuring = uicontrol('Style','checkbox','Parent',S.hFig, ...
                              'String','Plot during run (many figures)', ...
                              'Value',0,'Position',[220 572 220 24]);

    uicontrol('Style','text','Parent',S.hFig,'String','Save prefix:', ...
              'HorizontalAlignment','left','Position',[470 575 80 18]);

    S.hPrefix = uicontrol('Style','edit','Parent',S.hFig,'String','', ...
                          'Position',[550 572 160 24]);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','Run Fit', ...
              'FontWeight','bold','Position',[730 570 140 28], ...
              'Callback',@onRun);

    uicontrol('Style','text','Parent',S.hFig,'String','Profile:', ...
              'HorizontalAlignment','left','Position',[20 535 60 18]);

    S.hProfile = uicontrol('Style','popupmenu','Parent',S.hFig, ...
                           'String',{'(none)'},'Value',1, ...
                           'Position',[85 532 250 24]);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','Plot selected', ...
              'Position',[350 532 120 24], 'Callback',@onPlotSelected);

    uicontrol('Style','pushbutton','Parent',S.hFig,'String','Export CSVs', ...
              'Position',[480 532 120 24], 'Callback',@onExport);

    S.hStatus = uicontrol('Style','text','Parent',S.hFig,'String','Idle', ...
                          'HorizontalAlignment','left','Position',[20 505 850 18]);

    % ---- Axes for plotting ----
    S.hAx = axes('Parent',S.hFig,'Units','pixels','Position',[700 70 380 420]);
    grid(S.hAx,'on');
    xlabel(S.hAx,'Pixel'); ylabel(S.hAx,'Intensity');
    title(S.hAx,'Selected profile fit');

    % ---- Results table ----
    S.hTable = uitable('Parent',S.hFig,'Units','pixels', ...
                       'Position',[20 20 650 470]);

    % Store in guidata
    guidata(S.hFig, S);

    % ========= Callbacks =========
    function onBrowse(~,~)
        S = guidata(S.hFig);
        [f,p] = uigetfile({'*.csv','CSV (*.csv)'}, 'Select CSV',char(ini.path));
        ini.path = p;
        if isequal(f,0), return; end
        S.csvFile = fullfile(p,f);
        set(S.hFile,'String',S.csvFile);
        set(S.hStatus,'String','File selected.');
        guidata(S.hFig,S);
    end

    function onUseShared(~,~)
        S = guidata(S.hFig);
        S.csvFile = 'cross_sections.csv';
        set(S.hFile,'String',S.csvFile);
        set(S.hStatus,'String','Set file to cross_sections.csv (current folder).');
        guidata(S.hFig,S);
    end

    function onRun(~,~)
        S = guidata(S.hFig);

        if isempty(S.csvFile)
            set(S.hStatus,'String','Please choose a CSV file first.');
            return;
        end

        modelItems = get(S.hModel,'String');
        modelMode = modelItems{get(S.hModel,'Value')};
        doPlot = logical(get(S.hPlotDuring,'Value'));

        set(S.hStatus,'String','Running fits...');
        drawnow;

        try
            [S.results, S.fitOut] = fit_beam_profiles_2017(S.csvFile, modelMode, doPlot, '');
            % Show table
            set(S.hTable,'Data',table2cell(S.results), 'ColumnName',S.results.Properties.VariableNames);

            % Fill profile dropdown
            names = S.results.ProfileName;
            if iscell(names)
                items = names;
            else
                items = cellstr(names);
            end
            if isempty(items), items = {'(none)'}; end
            set(S.hProfile,'String',items,'Value',1);

            set(S.hStatus,'String',sprintf('Done. Fitted %d profile(s).', height(S.results)));
        catch ME
            set(S.hStatus,'String',['Error: ' ME.message]);
        end

        guidata(S.hFig,S);
    end

    function onPlotSelected(~,~)
        S = guidata(S.hFig);
        if isempty(S.fitOut)
            set(S.hStatus,'String','Run the fit first.');
            return;
        end

        items = get(S.hProfile,'String');
        idx = get(S.hProfile,'Value');
        if isempty(items) || strcmp(items{1},'(none)')
            set(S.hStatus,'String','No profile to plot.');
            return;
        end

        fo = S.fitOut(idx);

        cla(S.hAx);
        plot(S.hAx, fo.x, fo.y, 'o'); hold(S.hAx,'on');
        plot(S.hAx, fo.x, fo.gaussian.yhat, '-', 'LineWidth',1.2);
        plot(S.hAx, fo.x, fo.lorentzian.yhat, '-', 'LineWidth',1.2);
        plot(S.hAx, fo.x, fo.yhatBest, '-', 'LineWidth',2.2);
        grid(S.hAx,'on');
        legend(S.hAx, {'Data','Gaussian','Lorentzian','Best'}, 'Location','best');
        xlabel(S.hAx,'Pixel'); ylabel(S.hAx,'Intensity');

        bm = fo.bestModel;
        title(S.hAx, sprintf('%s | best=%s | AIC(G)=%.2f AIC(L)=%.2f', ...
            fo.name, bm, fo.gaussian.AIC, fo.lorentzian.AIC));

        set(S.hStatus,'String',['Plotted: ' fo.name]);
        guidata(S.hFig,S);
    end

    function onFilter2D(~,~)
        S = guidata(S.hFig);

        if isempty(S.imgOrig)
            [S, ok] = pickImage(S);
            if ~ok
                guidata(S.hFig,S);
                return;
            end
        else
            resp = questdlg('Use the currently loaded 2D image?','2D image', ...
                            'Reuse','Pick new','Reuse');
            if ischar(resp) && strcmp(resp,'Pick new')
                [S, ok] = pickImage(S);
                if ~ok
                    guidata(S.hFig,S);
                    return;
                end
            end
        end

        defMode = S.filterMode;
        defLow  = num2str(S.filterCutLow);
        defHigh = num2str(S.filterCutHigh);

        prompt = {'Mode: lowpass / highpass / bandpass', ...
                  'Low cutoff (% of Nyquist radius)', ...
                  'High cutoff (% for bandpass)'};
        answer = inputdlg(prompt, '2D frequency filter', 1, {defMode, defLow, defHigh});
        if isempty(answer)
            set(S.hStatus,'String','2D filter cancelled.');
            guidata(S.hFig,S);
            return;
        end

        mode = lower(strtrim(answer{1}));
        if isempty(mode), mode = defMode; end
        if ~ismember(mode, {'lowpass','highpass','bandpass'})
            set(S.hStatus,'String','Mode must be lowpass, highpass, or bandpass.');
            guidata(S.hFig,S);
            return;
        end

        cutLow = str2double(answer{2});
        cutHigh = str2double(answer{3});
        if ~isfinite(cutLow),  cutLow  = S.filterCutLow;  end
        if ~isfinite(cutHigh), cutHigh = S.filterCutHigh; end

        cutLow = max(1, min(99, cutLow));
        cutHigh = max(1, min(99, cutHigh));
        if strcmp(mode,'bandpass') && cutHigh <= cutLow
            cutHigh = min(99, cutLow + 5);
        end

        set(S.hStatus,'String','Filtering 2D image...');
        drawnow;

        S.filterMode = mode;
        S.filterCutLow = cutLow;
        S.filterCutHigh = cutHigh;

        S.imgFilt = applyFreqFilter2D(S.imgOrig, mode, cutLow, cutHigh);

        S = showFilteredImage(S, mode, cutLow, cutHigh);
        S = showFFTFigure(S);

        set(S.hStatus,'String',sprintf('2D filter done (%s %.1f%% / %.1f%%).', mode, cutLow, cutHigh));
        guidata(S.hFig,S);
    end

    function [S, ok] = pickImage(S)
        ok = false;
        [f,p] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.*','Images'}, ...
                          'Select 2D image to filter');
        if isequal(f,0)
            set(S.hStatus,'String','Image selection cancelled.');
            return;
        end

        try
            S.imgFile = fullfile(p,f);
            S.imgOrig = im2double(imread(S.imgFile));
            ok = true;
            set(S.hStatus,'String',['Loaded image: ' S.imgFile]);
        catch ME
            set(S.hStatus,'String',['Could not read image: ' ME.message]);
        end
    end

    function Iout = applyFreqFilter2D(Iin, mode, cutLow, cutHigh)
        I = double(Iin);
        if max(I(:)) > 1
            I = I ./ max(I(:));
        end

        if ndims(I) == 2
            Iout = filterChannel(I);
        else
            Iout = zeros(size(I));
            for c = 1:size(I,3)
                Iout(:,:,c) = filterChannel(I(:,:,c));
            end
        end

        function O = filterChannel(ch)
            [M,N] = size(ch);
            F = fftshift(fft2(ch));

            u = (-floor(N/2)):(ceil(N/2)-1);
            v = (-floor(M/2)):(ceil(M/2)-1);
            [U,V] = meshgrid(u,v);
            D = sqrt(U.^2 + V.^2);

            Dmax = min(M,N)/2;
            D0 = max(1, (cutLow/100) * Dmax);
            D1 = max(1, (cutHigh/100) * Dmax);

            switch mode
                case 'lowpass'
                    H = exp(-(D.^2) / (2*(D0^2)));
                case 'highpass'
                    H = 1 - exp(-(D.^2) / (2*(D0^2)));
                case 'bandpass'
                    Hlow = exp(-(D.^2) / (2*(D1^2)));
                    Hhigh = 1 - exp(-(D.^2) / (2*(D0^2)));
                    H = Hlow .* Hhigh;
                otherwise
                    H = exp(-(D.^2) / (2*(D0^2)));
            end

            G = F .* H;
            O = real(ifft2(ifftshift(G)));
            O = min(max(O,0),1);
        end
    end

    function S = showFilteredImage(S, mode, cutLow, cutHigh)
        if isempty(S.imgFilt)
            return;
        end

        if isempty(S.hImageFig) || ~ishandle(S.hImageFig)
            S.hImageFig = figure('Name','2D Filtered Image', ...
                'NumberTitle','off','MenuBar','none','ToolBar','figure');
        else
            figure(S.hImageFig); clf(S.hImageFig);
        end

        subplot(1,2,1);
        imshow(normalizeTo01(S.imgOrig));
        title('Original');

        subplot(1,2,2);
        imshow(normalizeTo01(S.imgFilt));
        if strcmp(mode,'bandpass')
            ttl = sprintf('Filtered (%s %.1f-%.1f%%)', mode, cutLow, cutHigh);
        else
            ttl = sprintf('Filtered (%s %.1f%%)', mode, cutLow);
        end
        title(ttl);
    end

    function S = showFFTFigure(S)
        if isempty(S.imgFilt)
            return;
        end

        if isempty(S.hFFTFig) || ~ishandle(S.hFFTFig)
            S.hFFTFig = figure('Name','FFT magnitude (filtered image)', ...
                'NumberTitle','off','MenuBar','none','ToolBar','figure');
        else
            figure(S.hFFTFig); clf(S.hFFTFig);
        end

        if ndims(S.imgFilt)==3
            Ig = mean(S.imgFilt,3);
        else
            Ig = S.imgFilt;
        end
        F = fftshift(fft2(double(Ig)));
        mag = log1p(abs(F));
        mag = mag ./ max(mag(:) + eps);

        imagesc(mag);
        axis image; colormap(gray(256)); colorbar;
        set(gca,'XTick',[],'YTick',[]);
        title('FFT magnitude (log scale)');
    end

    function out = normalizeTo01(in)
        in = double(in);
        mn = min(in(:));
        mx = max(in(:));
        if mx > mn
            out = (in - mn) ./ (mx - mn);
        else
            out = zeros(size(in));
        end
        out = min(max(out,0),1);
    end

    function onExport(~,~)
        S = guidata(S.hFig);
        if isempty(S.results) || isempty(S.fitOut)
            set(S.hStatus,'String','Run the fit first.');
            return;
        end

        prefix = get(S.hPrefix,'String');
        if isempty(prefix), prefix = 'beamfit'; end

        try
            % save results
            writetable(S.results, [prefix '_fit_results.csv']);

            % save best fit curves on original x grid
            T = readtable(S.csvFile);
            Xref = T{:,1};
            YAll = T{:,2:end};
            Yfit = nan(size(YAll));

            for c = 1:size(YAll,2)
                if isfield(S.fitOut(c),'x') && ~isempty(S.fitOut(c).x)
                    Yfit(:,c) = interp1(S.fitOut(c).x, S.fitOut(c).yhatBest, Xref, 'linear', 'extrap');
                end
            end

            Tout = array2table([Xref, Yfit], 'VariableNames', T.Properties.VariableNames);
            writetable(Tout, [prefix '_bestfit_curves.csv']);

            set(S.hStatus,'String',['Exported: ' prefix '_fit_results.csv and ' prefix '_bestfit_curves.csv']);
        catch ME
            set(S.hStatus,'String',['Export error: ' ME.message]);
        end

        guidata(S.hFig,S);
    end
end
