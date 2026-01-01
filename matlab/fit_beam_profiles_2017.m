function [results, fitOut] = fit_beam_profiles_2017(csvFile, modelMode, doPlot, savePrefix)
%FIT_BEAM_PROFILES_2017 (MATLAB R2017 compatible)
% Fits each profile column (except first) to Gaussian and Lorentzian.
% Picks best by AIC when modelMode='auto'.
%
% CSV format:
%   Row 1: headers
%   Col 1: pixel axis (x)
%   Col 2..end: intensity profiles
%
% Inputs:
%   csvFile   : e.g. 'cross_sections.csv'
%   modelMode : 'auto' (default) | 'gaussian' | 'lorentzian'
%   doPlot    : true/false (default false) -> creates a figure per profile (optional)
%   savePrefix: '' or prefix to export CSV outputs (default '')
%
% Outputs:
%   results : table of fitted parameters and goodness
%   fitOut  : struct array, one element per profile, with detailed fit info

    if nargin < 2 || isempty(modelMode), modelMode = 'auto'; end
    if nargin < 3 || isempty(doPlot),    doPlot    = false;  end
    if nargin < 4 || isempty(savePrefix),savePrefix= '';     end

    modelMode = lower(modelMode);

    % Read CSV with header row (R2017-safe)
    T = readtable(csvFile);  % variable names may be sanitized by MATLAB
    if width(T) < 2
        error('CSV must have at least 2 columns: x + at least one profile.');
    end

    % Also try to read the ORIGINAL header line (for nicer labels)
    headerNames = {};
    fid = fopen(csvFile,'r');
    if fid ~= -1
        hdr = fgetl(fid);
        fclose(fid);
        if ischar(hdr)
            headerNames = strsplit(hdr, ',');
        end
    end

    xAll = T{:,1};
    YAll = T{:,2:end};
    nCols = size(YAll,2);

    % Profile names
    if numel(headerNames) >= 2
        % Use original header tokens if possible
        profileNames = headerNames(2:end);
    else
        profileNames = T.Properties.VariableNames(2:end);
    end

    % Prefer lsqcurvefit if available
    useLSQ = (exist('lsqcurvefit','file') == 2);

    % Preallocate
    fitOut = repmat(struct(), nCols, 1);

    ProfileName = profileNames(:);
    ColIndex    = (1:nCols)';

    BestModel   = cell(nCols,1);
    A_best      = nan(nCols,1);
    Center_best = nan(nCols,1);
    Width_best  = nan(nCols,1);  % sigma for Gaussian, gamma(FWHM) for Lorentzian
    FWHM_best   = nan(nCols,1);
    Offset_best = nan(nCols,1);

    SSE_best    = nan(nCols,1);
    RMSE_best   = nan(nCols,1);
    R2_best     = nan(nCols,1);
    AIC_G       = nan(nCols,1);
    AIC_L       = nan(nCols,1);

    for c = 1:nCols
        yAll = YAll(:,c);

        good = isfinite(xAll) & isfinite(yAll);
        x = double(xAll(good));
        y = double(yAll(good));

        fitOut(c).name = ProfileName{c};
        fitOut(c).status = 'ok';

        if numel(x) < 8
            fitOut(c).status = 'too_few_points';
            continue;
        end

        % -------- initial guesses --------
        n = numel(x);
        kEdge = max(1, round(0.1*n));
        y0 = median([y(1:kEdge); y(end-kEdge+1:end)]);

        [yMax, iMax] = max(y);
        A0 = max(eps, yMax - y0);
        x0 = x(iMax);

        w0 = max(1, 0.1*(max(x)-min(x)));
        w = max(y - y0, 0);
        if sum(w) > 0
            mu = sum(x.*w)/sum(w);
            sig = sqrt(sum(((x-mu).^2).*w)/sum(w));
            if isfinite(sig) && sig > 0
                w0 = sig;
                x0 = mu;
            end
        end

        lb = [0,      min(x), 1e-6, -Inf];
        ub = [Inf,    max(x), (max(x)-min(x))*2, Inf];

        % -------- Gaussian fit --------
        g = struct();
        try
            p0g = [A0, x0, w0, y0];
            [pg, sseG, yhatG] = doFit(@gaussModel, p0g, lb, ub, x, y, useLSQ);
            g.p = pg; g.SSE = sseG; g.yhat = yhatG;
            g.R2 = calcR2(y, yhatG);
            g.RMSE = sqrt(sseG/numel(y));
            g.AIC = calcAIC(numel(y), sseG, 4);
        catch
            g.p = [nan nan nan nan]; g.SSE = Inf; g.yhat = nan(size(y));
            g.R2 = -Inf; g.RMSE = Inf; g.AIC = Inf;
        end

        % -------- Lorentzian fit --------
        l = struct();
        try
            gamma0 = max(1e-3, 2.0*w0); % heuristic
            p0l = [A0, x0, gamma0, y0];
            [pl, sseL, yhatL] = doFit(@lorentzModel, p0l, lb, ub, x, y, useLSQ);
            l.p = pl; l.SSE = sseL; l.yhat = yhatL;
            l.R2 = calcR2(y, yhatL);
            l.RMSE = sqrt(sseL/numel(y));
            l.AIC = calcAIC(numel(y), sseL, 4);
        catch
            l.p = [nan nan nan nan]; l.SSE = Inf; l.yhat = nan(size(y));
            l.R2 = -Inf; l.RMSE = Inf; l.AIC = Inf;
        end

        AIC_G(c) = g.AIC;
        AIC_L(c) = l.AIC;

        % -------- choose best --------
        if strcmp(modelMode,'gaussian')
            best = 'gaussian';
        elseif strcmp(modelMode,'lorentzian')
            best = 'lorentzian';
        else
            if g.AIC <= l.AIC
                best = 'gaussian';
            else
                best = 'lorentzian';
            end
        end

        fitOut(c).x = x;
        fitOut(c).y = y;
        fitOut(c).gaussian = g;
        fitOut(c).lorentzian = l;
        fitOut(c).bestModel = best;

        if strcmp(best,'gaussian')
            pBest = g.p; yhatBest = g.yhat; sseBest = g.SSE; r2Best = g.R2; rmseBest = g.RMSE;
            fwhm = 2*sqrt(2*log(2))*pBest(3);  % from sigma
        else
            pBest = l.p; yhatBest = l.yhat; sseBest = l.SSE; r2Best = l.R2; rmseBest = l.RMSE;
            fwhm = pBest(3);                   % gamma is FWHM
        end

        fitOut(c).pBest = pBest;
        fitOut(c).yhatBest = yhatBest;

        BestModel{c}  = best;
        A_best(c)      = pBest(1);
        Center_best(c) = pBest(2);
        Width_best(c)  = pBest(3);
        FWHM_best(c)   = fwhm;
        Offset_best(c) = pBest(4);

        SSE_best(c)  = sseBest;
        RMSE_best(c) = rmseBest;
        R2_best(c)   = r2Best;

        if doPlot
            figure('Name',sprintf('%s (col %d)', ProfileName{c}, c));
            plot(x,y,'o'); hold on;
            plot(x,g.yhat,'-','LineWidth',1.2);
            plot(x,l.yhat,'-','LineWidth',1.2);
            plot(x,yhatBest,'-','LineWidth',2.2);
            grid on;
            legend('Data','Gaussian','Lorentzian','Best','Location','best');
            title(sprintf('%s | best=%s | R2=%.4f | AIC(G)=%.2f AIC(L)=%.2f', ...
                ProfileName{c}, best, r2Best, g.AIC, l.AIC));
            xlabel('Pixel'); ylabel('Intensity');
        end
    end

    results = table(ProfileName, ColIndex, BestModel, ...
        A_best, Center_best, Width_best, FWHM_best, Offset_best, ...
        SSE_best, RMSE_best, R2_best, AIC_G, AIC_L);

    % Optional export
    if ~isempty(savePrefix)
        writetable(results, [savePrefix '_fit_results.csv']);

        Xref = T{:,1};
        Yfit = nan(size(YAll));
        for c = 1:nCols
            if isfield(fitOut(c),'x') && ~isempty(fitOut(c).x)
                Yfit(:,c) = interp1(fitOut(c).x, fitOut(c).yhatBest, Xref, 'linear', 'extrap');
            end
        end

        % keep same column headers as readtable has
        Tout = array2table([Xref, Yfit], 'VariableNames', T.Properties.VariableNames);
        writetable(Tout, [savePrefix '_bestfit_curves.csv']);
    end
end

% --------- models ----------
function y = gaussModel(p, x)
% p = [A, center, sigma, offset]
    A = p(1); x0 = p(2); s = p(3); b = p(4);
    y = b + A .* exp(-((x - x0).^2) ./ (2*s.^2));
end

function y = lorentzModel(p, x)
% p = [A, center, gamma(FWHM), offset]
    A = p(1); x0 = p(2); g = p(3); b = p(4);
    h = 0.5*g;
    y = b + A .* (h.^2 ./ ((x - x0).^2 + h.^2));
end

% --------- fitter wrapper ----------
function [pFit, sse, yhat] = doFit(modelFun, p0, lb, ub, x, y, useLSQ)
    if useLSQ
        opts = optimoptions('lsqcurvefit', ...
            'Display','off', 'MaxFunctionEvaluations',2e4, 'MaxIterations',2e3);
        pFit = lsqcurvefit(@(p,xx) modelFun(p,xx), p0, x, y, lb, ub, opts);
        yhat = modelFun(pFit, x);
        r = yhat - y;
        sse = sum(r.^2);
    else
        % fminsearch fallback with bounds via transform
        z0 = invTransformParams(p0, lb, ub);
        opts = optimset('Display','off','MaxFunEvals',5e4,'MaxIter',5e4);
        zFit = fminsearch(@(zz) objSSE(zz, modelFun, lb, ub, x, y), z0, opts);
        pFit = transformParams(zFit, lb, ub);
        yhat = modelFun(pFit, x);
        r = yhat - y;
        sse = sum(r.^2);
    end
end

function sse = objSSE(z, modelFun, lb, ub, x, y)
    p = transformParams(z, lb, ub);
    r = modelFun(p,x) - y;
    sse = sum(r.^2);
end

% --------- parameter transforms (bounds) ----------
function p = transformParams(z, lb, ub)
    p = zeros(size(z));
    for i = 1:numel(z)
        lbi = lb(i); ubi = ub(i);
        if isfinite(lbi) && isfinite(ubi)
            p(i) = lbi + (ubi - lbi) * sigmoid(z(i));
        elseif isfinite(lbi) && ~isfinite(ubi)
            p(i) = lbi + exp(z(i));
        elseif ~isfinite(lbi) && isfinite(ubi)
            p(i) = ubi - exp(z(i));
        else
            p(i) = z(i);
        end
    end
end

function z = invTransformParams(p, lb, ub)
    z = zeros(size(p));
    for i = 1:numel(p)
        lbi = lb(i); ubi = ub(i);
        if isfinite(lbi) && isfinite(ubi)
            t = (p(i) - lbi) / (ubi - lbi);
            t = min(max(t, 1e-9), 1-1e-9);
            z(i) = log(t/(1-t));
        elseif isfinite(lbi) && ~isfinite(ubi)
            z(i) = log(max(p(i) - lbi, 1e-12));
        elseif ~isfinite(lbi) && isfinite(ubi)
            z(i) = log(max(ubi - p(i), 1e-12));
        else
            z(i) = p(i);
        end
    end
end

function s = sigmoid(t)
    s = 1 ./ (1 + exp(-t));
end

% --------- metrics ----------
function r2 = calcR2(y, yhat)
    ssRes = sum((y - yhat).^2);
    ssTot = sum((y - mean(y)).^2);
    if ssTot <= 0
        r2 = NaN;
    else
        r2 = 1 - ssRes/ssTot;
    end
end

function aic = calcAIC(n, sse, k)
    sse = max(sse, eps);
    aic = n*log(sse/n) + 2*k;
end
