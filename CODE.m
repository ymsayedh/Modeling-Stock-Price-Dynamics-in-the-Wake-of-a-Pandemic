% Modeling Stock Price Dynamics in the Wake of a Pandemic


%% 1. Import Data
% Define file paths
stocks_file = 'Stocks(S&P500).csv';                  % Enter Stock Price file
covid_file = 'Pandemic(COVID19_USA).csv';            % Enter Pandemic Spread file

% Read S&P500 data
stocks_data = readtable(stocks_file);
stocks_data.Date = datetime(stocks_data.Date, 'InputFormat', 'dd-MM-yyyy');

% Read COVID19 data
covid_data = readtable(covid_file);
covid_data.Date = datetime(covid_data.Date, 'InputFormat', 'dd-MM-yyyy');


%% 2. Synchronize Data by Dates
start_date = max(min(stocks_data.Date), min(covid_data.Date));
end_date = min(max(stocks_data.Date), max(covid_data.Date));

stocks_data = stocks_data(stocks_data.Date >= start_date & stocks_data.Date <= end_date, :);
covid_data = covid_data(covid_data.Date >= start_date & covid_data.Date <= end_date, :);

% Extract Close Prices and Infection Rates
S_real = str2double(strrep(stocks_data.Close, ',', ''));
p_real = covid_data.Total_Deaths;

% Match lengths
min_length = min(length(S_real), length(p_real));
S_real = S_real(1:min_length);
p_real = p_real(1:min_length);

% Normalize Data
S_real = (S_real - min(S_real)) / (max(S_real) - min(S_real));
p_real = (p_real - min(p_real)) / (max(p_real) - min(p_real));


%% 3. Define Model Parameters
% Spatial Domain
L = 100; Nx = 200; x = linspace(0, L, Nx);
dx = x(2) - x(1);

% Temporal Domain
T = 365; Nt = min_length; t = linspace(0, T, Nt);
dt = t(2) - t(1);

% Initialize Variables
S = zeros(Nx, Nt);
p = zeros(Nx, Nt);

% Initial Conditions
S(:, 1) = S_real(1);
p(:, 1) = p_real(1) * exp(-((x - L/2).^2) / (2*(L/10)^2));

% Model Parameters
Ds = 0.1; Dp = 0.05; alpha = 0.02; beta = 0.01; gamma = 0.03; 
noise_amp = 0.005; initial_beta = beta;
real_weight = 0.2; % Weight for real stock price correction
feedback_gain = 0.05; % Proportional feedback gain for error correction

%% 4. Simulate the Dynamics
for n = 1:Nt-1
    beta = initial_beta * exp(-n / Nt); % Exponential decay for recovery

    for i = 2:Nx-1
        % Simulate stock prices
        lagged_p = p(i, max(1, n-10));
        time_scaling_factor = 1 + 0.5 * p_real(n); % Dynamic noise scaling
        noise_dynamic = noise_amp * time_scaling_factor * randn();

        % Apply feedback adjustment
        feedback_error = real_weight * (S_real(n) - S(i, n));

        S(i, n+1) = S(i, n) + dt * ( ...
            Ds * (S(i+1, n) - 2*S(i, n) + S(i-1, n)) / dx^2 ...
            - alpha * lagged_p^2 * S(i, n) ...
            + beta * (1 - S(i, n)) ...
            + noise_dynamic ...
            + feedback_gain * feedback_error); % Error correction term

        % Pandemic dynamics
        logistic_growth = gamma * p(i, n) * (1 - p(i, n));
        interaction_effect = -0.01 * S(i, n) * p(i, n);
        diffusion_effect = Dp * (p(i+1, n) - 2*p(i, n) + p(i-1, n)) / dx^2;

        p(i, n+1) = p(i, n) + dt * (diffusion_effect + logistic_growth + interaction_effect);
    end

    % Periodic realignment with real data
    if mod(n, 10) == 0
        p(:, n+1) = p_real(n) * exp(-((x - L/2).^2) / (2*(L/10)^2));
        S(:, n+1) = S(:, n+1) + real_weight * (S_real(n) - mean(S(:, n+1)));
    end

    % Boundary conditions
    S(1, n+1) = S(2, n+1);
    S(end, n+1) = S(end-1, n+1);
    p(1, n+1) = max(0, p(2, n+1));
    p(end, n+1) = max(0, p(end-1, n+1));
end

%% 6. Visualize Results

% Pandemic Severity
figure;
surf(x, t, p', 'EdgeColor', 'none');
view(2);
colorbar;
title('Pandemic Severity Dynamics');
xlabel('Space');
ylabel('Time (days)');
zlabel('Normalized Infection Rate');
saveas(gcf, 'Pandemic_Severity_Dynamics.png');

% Real vs Predicted Stock Prices
center_index = round(Nx/2);
S_predicted = S(center_index, :);

figure;
plot(t, S_real, 'b', 'LineWidth', 2); hold on;
plot(t, S_predicted, 'r--', 'LineWidth', 2);
title('Real vs Predicted Stock Prices');
xlabel('Time (days)');
ylabel('Normalized Stock Price');
legend('Real S&P500', 'Predicted S&P500');
saveas(gcf, 'Real_vs_Predicted_StockPrices.png');

% Root Mean Squared Error
mse = sqrt(mean((S_simulated - S_real').^2));
disp(['Root Mean Squared Error: ', num2str(mse)]);
