clear all
close
rng('default')
%addpath(genpath('/Users/matteoseveri/Desktop/Machine Learning/FNAR/tensor_toolbox-v3.7'))
%savepath
addpath(genpath("/Users/gianl/Desktop/UniBo/2° anno/(1) Machine Learning For Economists/FNAR"))
savepath


%% IMPORT DATA
% Import tensor data
W = readmatrix("FNAR.csv");
W = W(1:end,6:end);
W = reshape(W',[41 41 45 15]);
W = permute(W,[2 1 3 4]);
Wt = tensor(W);
N = size(Wt,1);
m = size(Wt,3);
Tw = size(Wt,4);

% Import y: multivariate time series for VAR
y = xlsread("Unemp_differences_def",2,'B4:AP18');



%% Tensor PC Evaluation 
num_fact = 45;
r = [41, 41,num_fact];

[Gamma,U,M,UM]=PC_tensor_norm(Wt,r);

% Eigenvalue-ratio criterion
ev = M{3}(:);                 
rmax = min([numel(ev)-1, 20]);
ratios = ev(1:rmax) ./ ev(2:rmax+1);
[~, r_hat] = max(ratios);

fprintf('Eigenvalue-ratio selected r = %d\n', r_hat);


% Screeplot
figure;

vals = M{3}/sum(M{3});
num_factors = length(vals);

plot(vals, 'LineWidth', 1.5);
hold on;

plot(1:3, vals(1:3), 'o', 'MarkerSize', 6, ...
     'MarkerFaceColor', 'auto', 'LineWidth', 1.3);

xline(1, '--k');
xline(2, '--k');
xline(3, '--k');

xlabel('Factor');
ylabel('Variance share');

xticks([1:5, 10:5:num_factors]);
grid on;
      

% Divergent eigenvalues
n_values   = 5:1:N;
num_layers = size(W,3);                       
evals_by_n = nan(num_layers, numel(n_values));

ctr = 1;
for n = n_values
    idx   = 1:n;
    W_sub = W(idx,idx,:,:);                     
    [Gamma,U,M,UM] = PC_tensor_norm(W_sub, r);
    evals = M{3};                             
    evals_by_n(1:numel(evals), ctr) = evals;
    ctr = ctr + 1;
end

figure; hold on; grid on;
k_to_plot = min(num_fact, size(evals_by_n,1));
colors = lines(k_to_plot);

for j = 1:k_to_plot
    scatter(n_values, evals_by_n(j,:), 36, colors(j,:), 'filled');  
end

xlabel('N (cross-sectional size)');
ylabel('Eigenvalue');



%% Tensor PC ESTIMATION
num_fact = 3;
r = [41, 41, num_fact];

[Gamma,U,M,UM] = PC_tensor_norm(Wt, r);

% then compute F_hat using the oriented U
F_hat = ttm(Wt, {diag(M{3}(1:r(3)))^(-1)*U{3}'*N^2}, 3);

% Fit and idiosyncratic tensors
W_hat = ttm(F_hat, {U{3}}, 3);
E_hat = Wt - W_hat;

R2 =  norm(W_hat)^2/norm(Wt)^2;



%% Construct W(t-1)*y(t-1) for FNAR

T = size(y,1);  % number of time periods (15)
num_fact = r(3);  % number of network factors
Wyt = zeros(T, N, num_fact);  % preallocate (T x N x r)

for t = 2:T
    y_lag = y(t-1, :)'; 
    for j = 1:num_fact
        F_layer = F_hat(:, :, j, t-1);  % N x N matrix: factor j at t-1
        Wyt(t, :, j) = (1/N) * (double(F_layer) * y_lag)';  % 1 x N
    end
end



%% ESTIMATION of Factor Network Autoregression with Homogeneous Momentum and Nodal effect
drop = 1;
T = size(y,1)-drop;
N = size(y,2);

% OLS
Y_homo = reshape(y(drop+1:end,:),[T*N 1]);
Y_homo_lag1 = reshape(y(drop:end-1,:),[T*N 1]);
WY_homo_lag1 = reshape(Wyt(drop+1:end,:,:),[T*N size(Wyt,3)]);
X_homo = [WY_homo_lag1 Y_homo_lag1 ones(T*N,1)];

[FNAR.beta_homo(:,1),FNAR.res_homo(:,1),FNAR.Avar_homo_OLS_HAC,~,FNAR.r2_homo(1,1),~] = ML_ols(Y_homo,X_homo,0,1);


% Covariance of residuals low rank plus sparse
q=1;
FNAR.res_homo_cov_hat_OLS = cov(reshape(FNAR.res_homo(:,1),T,N));
[Vres,Dres]= eigs(FNAR.res_homo_cov_hat_OLS,q);
FNAR.G_homo = ((Dres)^(-1/2)*Vres'*reshape(FNAR.res_homo(:,1),T,N)')';
FNAR.Lambda_homo = Vres*(Dres)^(1/2);
FNAR.epsilon_homo = reshape(FNAR.res_homo(:,1),T,N)- FNAR.G_homo*FNAR.Lambda_homo';
FNAR.S = diag(diag(cov(FNAR.epsilon_homo)));

FNAR.res_homo_cov_hat_factor = Vres*Dres*Vres'+FNAR.S;

wood_temp = eye(q)/(eye(q)+FNAR.Lambda_homo'*(eye(N)/FNAR.S)*FNAR.Lambda_homo);
FNAR.res_homo_cov_hat_factor_inv = (eye(N)/FNAR.S)-(eye(N)/FNAR.S)*FNAR.Lambda_homo*wood_temp*FNAR.Lambda_homo'*(eye(N)/FNAR.S);

clear Vres Dres wood_temp


% GLS
Wyt_weighted = zeros(size(Wyt));
Y_homo_weighted = reshape(y(drop+1:end,:)/((FNAR.res_homo_cov_hat_factor)^(1/2)),[T*N 1]);
Y_homo_weighted_lag1 = reshape(y(drop:end-1,:)/((FNAR.res_homo_cov_hat_factor)^(1/2)),[T*N 1]);
for k=1:num_fact
    Wyt_weighted(:,:,k) = Wyt(:,:,k)/((FNAR.res_homo_cov_hat_factor)^(1/2));
end
WY_homo_weighted_lag1 = reshape(Wyt_weighted(drop:end-1,:,:),[T*N size(Wyt,3)]);
X_homo_weighted = [WY_homo_weighted_lag1 Y_homo_weighted_lag1 ones(T*N,1)];

[FNAR.beta_homo(:,2),FNAR.res_homo(:,2),FNAR.Avar_homo_GLS_HAC,~,FNAR.r2_homo(1,2),~] = ML_ols(Y_homo_weighted,X_homo_weighted,0,1);


FNAR.res_homo_cov_hat_GLS = cov(reshape(FNAR.res_homo(:,2),T,N));

% Asymptotic covariance robust standard errors (heteroskedastic but no HAC)
Sigma_XX = X_homo'*X_homo/(N*T);
Sigma_XeeX_OLS = X_homo'*kron(eye(T),FNAR.res_homo_cov_hat_OLS)*X_homo/(N^2*T);

Sigma_XVX = X_homo_weighted'*X_homo_weighted/(N*T);
Sigma_XeeX_GLS = X_homo_weighted'*kron(eye(T),FNAR.res_homo_cov_hat_GLS)*X_homo_weighted/(N*T);

FNAR.Avar_homo_OLS = (eye(size(FNAR.beta_homo(:,1),1))/Sigma_XX)*Sigma_XeeX_OLS*(eye(size(FNAR.beta_homo(:,1),1))/Sigma_XX);
FNAR.Avar_homo_GLS = (eye(size(FNAR.beta_homo(:,2),1))/Sigma_XVX)*Sigma_XeeX_GLS*(eye(size(FNAR.beta_homo(:,2),1))/Sigma_XVX);

clear Sigma_XX Sigma_XVX Sigma_XeeX*


% standard errors
FNAR.SE_homo(:,1) = sqrt(diag(FNAR.Avar_homo_OLS)/T);
FNAR.SE_homo(:,2) = sqrt(diag(FNAR.Avar_homo_GLS)/(N*T));

% t-test
FNAR.Tstat_homo = FNAR.beta_homo./FNAR.SE_homo;
disp(FNAR.Tstat_homo)

%% Plot factor loadings for different network layers

figure('Name','','Color','w');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

num_layers = size(U{3},1);
x = 1:num_layers;

colors = [ ...
    0.2 0.7 0.3;  % green for Factor 1
    0.2 0.5 1.0;  % blue for Factor 2
    0.5 0.5 0.5;  % gray for Factor 3
    1.0 0.8 0.1;  % yellow for Factor 4
    0.4 0.7 1.0;  % light blue for Factor 5
    0.5 0.2 0.8]; % purple for factor 6

ymin = min(U{3}(:)) - 0.01;
ymax = max(U{3}(:)) + 0.01;

for j = 1:num_fact
    nexttile;
    scatter(x, U{3}(:,j), 36, colors(j,:), 'filled');
    hold on;
    yline(0,'k','LineWidth',0.6);
    xline(5,'k:','LineWidth',0.5);
    xline(26,'k:','LineWidth',0.5);
    xlabel('network layer');
    ylabel('loading');
    title(['Factor ' num2str(j)]);
    ylim([ymin ymax]);
    xlim([1 num_layers]);
    grid on;
    box on;
end

%% Average network factors over time
t_start = 1;                                 
t_end   = size(F_hat,4);
Favg    = mean(double(F_hat(:,:,:,t_start:t_end)), 4);   % N x N x r
r       = size(Favg,3);

% Shape and colors
n  = 1681; 
h  = floor(n/2);

white = [1.0 1.0 1.0];
red   = [1.00 0.15 0.15];    % dark red
green = [0.10 0.70 0.20];    % dark green

neg = [linspace(red(1),  white(1), h)', ...
       linspace(red(2),  white(2), h)', ...
       linspace(red(3),  white(3), h)'];

pos = [linspace(white(1), green(1), h)', ...
       linspace(white(2), green(2), h)', ...
       linspace(white(3), green(3), h)'];

cmap_rwgr = [neg; pos];      

nrows = floor(sqrt(r));
ncols = ceil(r/nrows);

% Plot panels
figure('Units','normalized','Position',[0.05 0.08 0.9 0.8]);
tiledlayout(nrows,ncols,'TileSpacing','compact','Padding','compact');

for j = 1:r
    A = Favg(:,:,j);
    clim = max(abs(A(:)));                         % symmetric limits around zero
    nexttile;
    imagesc(A,[-clim,clim]); axis image; set(gca,'YDir','normal');
    colormap(cmap_rwgr);
    title(sprintf('%d',j),'FontWeight','normal');
    set(gca,'TickLength',[0 0],'Box','on');        % minimal axes
end

% Country labels and axes
countries = ["AUS","AUT","BEL","BGR","BRA","CAN","CHE","CHN","CYP","CZE", ...
             "DEU","DNK","ESP","EST","FIN","FRA","GBR","GRC","HRV","HUN", ...
             "IDN","IND","IRL","ITA","JPN","KOR","LTU","LUX","LVA","MEX", ...
             "MLT","NLD","NOR","POL","PRT","ROU","SVK","SVN","SWE","TUR","USA"];  


axs = findobj(gcf,'Type','axes');
axs = flipud(axs);

for k = 1:numel(axs)
    ax = axs(k);

    set(ax, 'XTick', 1:numel(countries), ...
            'YTick', 1:numel(countries), ...
            'XTickLabel', countries, ...
            'YTickLabel', countries, ...   
            'TickLabelInterpreter','none', ...
            'FontSize', 6);

    set(ax, 'XAxisLocation','top');
    set(ax, 'YAxisLocation','left');

    xtickangle(ax,90);
end




%% %% Tensor PC estimation across two time periods (2000–2007 vs 2008–2014)

years = 2000:2014;
idx_pre  = find(years <= 2007);   % 2000–2007 (first 8 years)
idx_post = find(years >= 2008);   % 2008–2014 (last 7 years)

Wt_pre  = Wt(:,:,:,idx_pre);      % tensor for 2000–2007
Wt_post = Wt(:,:,:,idx_post);     % tensor for 2008–2014

num_fact_two = 3;                 
r_two = [N, N, num_fact_two];

% PCA on pre-2008 period
[Gamma_pre, U_pre, M_pre, UM_pre] = PC_tensor_norm(Wt_pre, r_two);

% PCA on post-2008 period
[Gamma_post, U_post, M_post, UM_post] = PC_tensor_norm(Wt_post, r_two);

% Tensor sign flipping for FNAR coefficient interpretation
U_pre{3}(:,1)  = -U_pre{3}(:,1);
U_post{3}(:,1) = -U_post{3}(:,1);

U_post{3}(:,2) = -U_post{3}(:,2);
U_pre{3}(:,3) = -U_pre{3}(:,3);

F_hat_pre = ttm(Wt_pre, {diag(M_pre{3}(1:r_two(3)))^(-1)*U_pre{3}'*N^2}, 3);
F_hat_post = ttm(Wt_post, {diag(M_post{3}(1:r_two(3)))^(-1)*U_post{3}'*N^2}, 3);

% Fit and idiosyncratic tensors
W_hat_pre = ttm(F_hat_pre, {U_pre{3}}, 3);
W_hat_post = ttm(F_hat_post, {U_post{3}}, 3);

E_hat_pre = Wt_pre - W_hat_pre;
E_hat_post = Wt_post - W_hat_post;

R2_pre =  norm(W_hat_pre)^2/norm(Wt_pre)^2;
R2_post =  norm(W_hat_post)^2/norm(Wt_post)^2;


%% Construct W(t-1)*y(t-1) for FNAR  (annual data)
y_pre  = y(idx_pre, :);     % 8 × N
y_post = y(idx_post, :);    % 7 × N

T_pre  = length(idx_pre); 
T_post  = length(idx_post); 

Wyt_pre = zeros(T_pre, N, num_fact_two); 
Wyt_post = zeros(T_post, N, num_fact_two);

for t = 2:T_pre
    y_lag_pre = y_pre(t-1, :)';  % N x 1 vector: y_{t-1}
    for j = 1:num_fact_two
        F_layer_pre = F_hat_pre(:, :, j, t-1);  % N x N matrix: factor j at t-1
        Wyt_pre(t, :, j) = (1/N) * (double(F_layer_pre) * y_lag_pre)';  % 1 x N
    end
end

for t = 2:T_post
    y_lag_post = y_post(t-1, :)';  % N x 1 vector: y_{t-1}
    for j = 1:num_fact_two
        F_layer_post = F_hat_post(:, :, j, t-1);  % N x N matrix: factor j at t-1
        Wyt_post(t, :, j) = (1/N) * (double(F_layer_post) * y_lag_post)';  % 1 x N
    end
end



%% FNAR ESTIMATION: PRE PERIOD (2000–2007)
drop_pre = 1;
T_pre_eff = T_pre - drop_pre;
N_pre = size(y_pre,2);

Y_pre = reshape(y_pre(drop_pre+1:end,:), [T_pre_eff*N_pre 1]);

Ylag_pre = reshape(y_pre(drop_pre:end-1,:), [T_pre_eff*N_pre 1]);

WY_pre = reshape(Wyt_pre(drop_pre+1:end,:,:), [T_pre_eff*N_pre num_fact_two]);

X_pre = [WY_pre  Ylag_pre  ones(T_pre_eff*N_pre,1)];

% OLS estimation
[FNAR_pre.beta, FNAR_pre.res, FNAR_pre.Avar_OLS_HAC,~, FNAR_pre.R2,~] = ML_ols(Y_pre, X_pre, 0, 1);

% Residual covariance decomposition
q = 1;
ResMat_pre = reshape(FNAR_pre.res, T_pre_eff, N_pre);
FNAR_pre.cov_res = cov(ResMat_pre);

[Vpre, Dpre] = eigs(FNAR_pre.cov_res, q);
FNAR_pre.G = ((Dpre)^(-1/2)*Vpre'*ResMat_pre')';
FNAR_pre.Lambda = Vpre*(Dpre)^(1/2);
FNAR_pre.epsilon = ResMat_pre - FNAR_pre.G*FNAR_pre.Lambda';
FNAR_pre.S = diag(diag(cov(FNAR_pre.epsilon)));

FNAR_pre.cov_factor = Vpre*Dpre*Vpre' + FNAR_pre.S;

wood = eye(q)/(eye(q)+FNAR_pre.Lambda'*(eye(N_pre)/FNAR_pre.S)*FNAR_pre.Lambda);
FNAR_pre.cov_factor_inv = ...
    (eye(N_pre)/FNAR_pre.S) ...
    - (eye(N_pre)/FNAR_pre.S)*FNAR_pre.Lambda*wood ...
      *FNAR_pre.Lambda'*(eye(N_pre)/FNAR_pre.S);

clear Vpre Dpre wood

%% Pre Period T-Test with OLS SE 

k_pre = size(FNAR_pre.beta,1);      
N_pre = size(y_pre,2);
T_pre_eff = size(y_pre,1)-1;

sigma2_pre = (FNAR_pre.res' * FNAR_pre.res) / (T_pre_eff*N_pre - k_pre);

XXinv_pre = inv(X_pre' * X_pre);

Var_beta_pre = sigma2_pre * XXinv_pre;

SE_pre = sqrt(diag(Var_beta_pre));

tstat_pre = FNAR_pre.beta ./ SE_pre;

pval_pre = 2*(1 - normcdf(abs(tstat_pre)));

disp('=== PRE-period SE ===');
disp(SE_pre);
disp('=== PRE-period p-values ===');
disp(pval_pre);


%% FNAR ESTIMATION: POST PERIOD (2008–2014)
drop_post = 1;
T_post_eff = T_post - drop_post;   % = 6
N_post = size(y_post,2);

Y_post = reshape(y_post(drop_post+1:end,:), [T_post_eff*N_post 1]);

Ylag_post = reshape(y_post(drop_post:end-1,:), [T_post_eff*N_post 1]);

WY_post = reshape(Wyt_post(drop_post+1:end,:,:), [T_post_eff*N_post num_fact_two]);

X_post = [WY_post  Ylag_post  ones(T_post_eff*N_post,1)];

% OLS estimation
[FNAR_post.beta, FNAR_post.res, FNAR_post.Avar_OLS_HAC,~, FNAR_post.R2,~] = ML_ols(Y_post, X_post, 0, 1);

% Residual covariance decomposition
ResMat_post = reshape(FNAR_post.res, T_post_eff, N_post);
FNAR_post.cov_res = cov(ResMat_post);

[Vpost, Dpost] = eigs(FNAR_post.cov_res, q);
FNAR_post.G = ((Dpost)^(-1/2)*Vpost'*ResMat_post')';
FNAR_post.Lambda = Vpost*(Dpost)^(1/2);
FNAR_post.epsilon = ResMat_post - FNAR_post.G*FNAR_post.Lambda';
FNAR_post.S = diag(diag(cov(FNAR_post.epsilon)));

FNAR_post.cov_factor = Vpost*Dpost*Vpost' + FNAR_post.S;

wood2 = eye(q)/(eye(q)+FNAR_post.Lambda'*(eye(N_post)/FNAR_post.S)*FNAR_post.Lambda);
FNAR_post.cov_factor_inv = ...
    (eye(N_post)/FNAR_post.S) ...
    - (eye(N_post)/FNAR_post.S)*FNAR_post.Lambda*wood2 ...
      *FNAR_post.Lambda'*(eye(N_post)/FNAR_post.S);

clear Vpost Dpost wood2

%% Post Period T-Test with OLS SE 

k_post = size(FNAR_post.beta,1);
N_post = size(y_post,2);
T_post_eff = size(y_post,1)-1;

sigma2_post = (FNAR_post.res' * FNAR_post.res) / (T_post_eff*N_post - k_post);

XXinv_post = inv(X_post' * X_post);

Var_beta_post = sigma2_post * XXinv_post;

SE_post = sqrt(diag(Var_beta_post));

tstat_post = FNAR_post.beta ./ SE_post;
pval_post = 2*(1 - normcdf(abs(tstat_post)));

% Display
disp('=== POST-period SE ===');
disp(SE_post);
disp('=== POST-period p-values ===');
disp(pval_post);



%% Plot factor loadings across layers for the two periods

figure('Name','Layer loadings: 2000–2007 vs 2008–2014','Color','w');
tiledlayout(num_fact_two, 2, 'TileSpacing','compact','Padding','compact');

num_layers = size(U_pre{3},1);
x = 1:num_layers;

colors_two = [ ...
    0.2 0.7 0.3;   % Factor 1
    0.2 0.5 1.0;   % Factor 2
    0.5 0.5 0.5];  % Factor 3

for j = 1:num_fact_two

    % Common vertical scale for factor j across the two periods
    ymin_j = min([U_pre{3}(:,j); U_post{3}(:,j)]) - 0.01;
    ymax_j = max([U_pre{3}(:,j); U_post{3}(:,j)]) + 0.01;

    % 2001–2007
    nexttile;
    scatter(x, U_pre{3}(:,j), 36, colors_two(j,:), 'filled');
    hold on;
    yline(0,'k','LineWidth',0.6);
    xlabel('network layer');
    ylabel('loading');
    title(sprintf('Factor %d (2000–2007)', j));
    ylim([ymin_j ymax_j]);
    xlim([1 num_layers]);
    grid on;
    box on;

    % 2008–2014
    nexttile;
    scatter(x, U_post{3}(:,j), 36, colors_two(j,:), 'filled');
    hold on;
    yline(0,'k','LineWidth',0.6);
    xlabel('network layer');
    ylabel('loading');
    title(sprintf('Factor %d (2008–2014)', j));
    ylim([ymin_j ymax_j]);
    xlim([1 num_layers]);
    grid on;
    box on;

end






