% Define the function f(x)
f = @(x) x^3 - 5*x^2 + 2*x + 15;

% Derivative of f(x) for Newton-Raphson
df = @(x) 3*x^2 - 10*x + 2;

% Bisection Method implementation
function [root, iter, f_values] = bisection(f, a, b, tol, max_iter)
    iter = 0;
    fa = f(a);
    fb = f(b);

    if fa * fb > 0
        error('The function must have different signs at the endpoints a and b');
    end

    % Array to store f(x) values for each iteration
    f_values = [];

    while (b - a) / 2 > tol && iter < max_iter
        iter = iter + 1;
        c = (a + b) / 2;
        fc = f(c);

        % Store f(c) for plotting
        f_values(iter) = fc;

        if fc == 0
            break;
        elseif fa * fc < 0
            b = c;
            fb = fc;
        else
            a = c;
            fa = fc;
        end
    end

    root = (a + b) / 2;
end

% False Position Method implementation
function [root, iter, f_values] = false_position(f, a, b, tol, max_iter)
    iter = 0;
    fa = f(a);
    fb = f(b);

    if fa * fb > 0
        error('The function must have different signs at the endpoints a and b');
    end

    % Array to store f(x) values for each iteration
    f_values = [];

    while abs(b - a) > tol && iter < max_iter
        iter = iter + 1;
        % Compute the false position
        c = (a * fb - b * fa) / (fb - fa);
        fc = f(c);

        % Store f(c) for plotting
        f_values(iter) = fc;

        if fc == 0
            break;
        elseif fa * fc < 0
            b = c;
            fb = fc;
        else
            a = c;
            fa = fc;
        end
    end

    root = c;
end

% Secant Method implementation
function [root, iter, f_values] = secant(f, x0, x1, tol, max_iter)
    iter = 0;

    % Array to store f(x) values for each iteration
    f_values = [];

    while abs(x1 - x0) > tol && iter < max_iter
        iter = iter + 1;
        fx0 = f(x0);
        fx1 = f(x1);

        % Compute the next approximation
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0);

        % Store f(x2) for plotting
        f_values(iter) = f(x2);

        x0 = x1;
        x1 = x2;
    end

    root = x1;
end

% Newton-Raphson Method implementation
function [root, iter, f_values] = newton_raphson(f, df, x0, tol, max_iter)
    iter = 0;

    % Array to store f(x) values for each iteration
    f_values = [];

    while abs(f(x0)) > tol && iter < max_iter
        iter = iter + 1;

        % Compute the next approximation
        x1 = x0 - f(x0) / df(x0);

        % Store f(x1) for plotting
        f_values(iter) = f(x1);

        x0 = x1;
    end

    root = x0;
end

% Function to plot the function values of two intervals
function plot_two_intervals(iter1, f_values1, iter2, f_values2, method_name)
    figure;
    plot(1:iter1, f_values1, '-ro', 'LineWidth', 2); hold on;
    plot(1:iter2, f_values2, '-bo', 'LineWidth', 2);
    xlabel('Number of Iterations');
    ylabel('f(x)');
    title([method_name ' Function Value vs Iterations for Two Intervals']);
    legend('Interval 1', 'Interval 2');
    hold off;
end

% Function to plot all four methods for each interval
function plot_all_methods(iter_bis, f_values_bis, iter_fp, f_values_fp, iter_sec, f_values_sec, iter_nr, f_values_nr, interval_name)
    figure;
    plot(1:iter_bis, f_values_bis, '-ro', 'LineWidth', 2); hold on;
    plot(1:iter_fp, f_values_fp, '-go', 'LineWidth', 2);
    plot(1:iter_sec, f_values_sec, '-bo', 'LineWidth', 2);
    plot(1:iter_nr, f_values_nr, '-ko', 'LineWidth', 2);
    xlabel('Number of Iterations');
    ylabel('f(x)');
    title(['All Methods: Function Value vs Iterations (' interval_name ')']);
    legend('Bisection', 'False Position', 'Secant', 'Newton-Raphson');
    hold off;
end

% Function to plot all four methods for two different thresholds
function plot_all_methods_threshold(iter_bis, f_values_bis, iter_fp, f_values_fp, iter_sec, f_values_sec, iter_nr, f_values_nr, threshold)
    figure;
    plot(1:iter_bis, f_values_bis, '-ro', 'LineWidth', 2); hold on;
    plot(1:iter_fp, f_values_fp, '-go', 'LineWidth', 2);
    plot(1:iter_sec, f_values_sec, '-bo', 'LineWidth', 2);
    plot(1:iter_nr, f_values_nr, '-ko', 'LineWidth', 2);
    xlabel('Number of Iterations');
    ylabel('f(x)');
    title(['All Methods: Function Value vs Iterations (Threshold ', num2str(threshold), ')']);
    legend('Bisection', 'False Position', 'Secant', 'Newton-Raphson');
    hold off;
end

% Function to plot method performance with different thresholds
function plot_method_with_thresholds(iter_tol1, f_values_tol1, iter_tol2, f_values_tol2, method_name)
    figure;
    plot(1:iter_tol1, f_values_tol1, '-o', 'Color', [1, 0, 0], 'LineWidth', 2, 'DisplayName', 'Threshold 0.01'); hold on;
    plot(1:iter_tol2, f_values_tol2, '-x', 'Color', [0, 0, 1], 'LineWidth', 2, 'DisplayName', 'Threshold 0.0001');
    xlabel('Number of Iterations');
    ylabel('f(x)');
    title([method_name ' Function Value vs Iterations (Thresholds 0.01 and 0.0001)']);
    legend('show');
    hold off;
end

% Main program to call methods for two intervals and thresholds
function root_approximation(f, df, a1, b1, x01, x11, a2, b2, x02, x12, tol1, tol2, max_iter)
    % Interval 1
    [root1_bis, iter1_bis, f_values1_bis] = bisection(f, a1, b1, tol1, max_iter);
    [root1_fp, iter1_fp, f_values1_fp] = false_position(f, a1, b1, tol1, max_iter);
    [root1_sec, iter1_sec, f_values1_sec] = secant(f, x01, x11, tol1, max_iter);
    [root1_nr, iter1_nr, f_values1_nr] = newton_raphson(f, df, x01, tol1, max_iter);

    % Interval 2
    [root2_bis, iter2_bis, f_values2_bis] = bisection(f, a2, b2, tol1, max_iter);
    [root2_fp, iter2_fp, f_values2_fp] = false_position(f, a2, b2, tol1, max_iter);
    [root2_sec, iter2_sec, f_values2_sec] = secant(f, x02, x12, tol1, max_iter);
    [root2_nr, iter2_nr, f_values2_nr] = newton_raphson(f, df, x02, tol1, max_iter);

    % Print approximated root values and number of iterations for interval 1
    fprintf('Interval 1:\n');
    fprintf('Bisection Method Root: %.5f, Iterations: %d\n', root1_bis, iter1_bis);
    fprintf('False Position Method Root: %.5f, Iterations: %d\n', root1_fp, iter1_fp);
    fprintf('Secant Method Root: %.5f, Iterations: %d\n', root1_sec, iter1_sec);
    fprintf('Newton-Raphson Method Root: %.5f, Iterations: %d\n', root1_nr, iter1_nr);

    % Print approximated root values and number of iterations for interval 2
    fprintf('Interval 2:\n');
    fprintf('Bisection Method Root: %.5f, Iterations: %d\n', root2_bis, iter2_bis);
    fprintf('False Position Method Root: %.5f, Iterations: %d\n', root2_fp, iter2_fp);
    fprintf('Secant Method Root: %.5f, Iterations: %d\n', root2_sec, iter2_sec);
    fprintf('Newton-Raphson Method Root: %.5f, Iterations: %d\n', root2_nr, iter2_nr);

    % Plot function values vs iterations for two intervals for each method
    plot_two_intervals(iter1_bis, f_values1_bis, iter2_bis, f_values2_bis, 'Bisection');
    plot_two_intervals(iter1_fp, f_values1_fp, iter2_fp, f_values2_fp, 'False Position');
    plot_two_intervals(iter1_sec, f_values1_sec, iter2_sec, f_values2_sec, 'Secant');
    plot_two_intervals(iter1_nr, f_values1_nr, iter2_nr, f_values2_nr, 'Newton-Raphson');

    % Bisection Method
    [root_bis_tol1, iter_bis_tol1, f_values_bis_tol1] = bisection(f, a1, b1, tol1, max_iter);
    [root_bis_tol2, iter_bis_tol2, f_values_bis_tol2] = bisection(f, a1, b1, tol2, max_iter);
    plot_method_with_thresholds(iter_bis_tol1, f_values_bis_tol1, iter_bis_tol2, f_values_bis_tol2, 'Bisection');

    % False Position Method
    [root_fp_tol1, iter_fp_tol1, f_values_fp_tol1] = false_position(f, a1, b1, tol1, max_iter);
    [root_fp_tol2, iter_fp_tol2, f_values_fp_tol2] = false_position(f, a1, b1, tol2, max_iter);
    plot_method_with_thresholds(iter_fp_tol1, f_values_fp_tol1, iter_fp_tol2, f_values_fp_tol2, 'False Position');

    % Secant Method
    [root_sec_tol1, iter_sec_tol1, f_values_sec_tol1] = secant(f, x01, x11, tol1, max_iter);
    [root_sec_tol2, iter_sec_tol2, f_values_sec_tol2] = secant(f, x01, x11, tol2, max_iter);
    plot_method_with_thresholds(iter_sec_tol1, f_values_sec_tol1, iter_sec_tol2, f_values_sec_tol2, 'Secant');

    % Newton-Raphson Method
    [root_nr_tol1, iter_nr_tol1, f_values_nr_tol1] = newton_raphson(f, df, x01, tol1, max_iter);
    [root_nr_tol2, iter_nr_tol2, f_values_nr_tol2] = newton_raphson(f, df, x01, tol2, max_iter);
    plot_method_with_thresholds(iter_nr_tol1, f_values_nr_tol1, iter_nr_tol2, f_values_nr_tol2, 'Newton-Raphson');
end

% Initial interval and guesses for each method
a1 = -10; b1 = 0;   % First interval for Bisection/False Position
x01 = -10; x11 = -9; % Initial guesses for Secant/Newton-Raphson
a2 = -2; b2 = 10;   % Second interval for Bisection/False Position
x02 = -5; x12 = 12;  % Initial guesses for Secant/Newton-Raphson

tol1 = 1e-2;        % First threshold (10^-2)
tol2 = 1e-4;        % Second threshold (10^-4)
max_iter = 100;

% Call the root approximation
root_approximation(f, df, a1, b1, x01, x11, a2, b2, x02, x12, tol1, tol2, max_iter);

% Define the function f2(x) and its derivative
f2 = @(x) x^3 - 2*x + 2;
df2 = @(x) 3*x^2 - 2;

% Newton-Raphson Method implementation
function [root, iter, f_values] = newton_raphson(f, df, x0, tol, max_iter)
    iter = 0;
    f_values = [];
    while abs(f(x0)) > tol && iter < max_iter
        iter = iter + 1;
        x1 = x0 - f(x0) / df(x0);
        f_values(iter) = f(x1);
        x0 = x1;
    end
    root = x0;
end

% Main program to call methods for two intervals and thresholds
function root_approximation(f, df, a1, b1, x01, x11, a2, b2, x02, x12, tol1, tol2, max_iter)
    % Interval 1
    [root1_bis, iter1_bis, f_values1_bis] = bisection(f, a1, b1, tol1, max_iter);
    [root1_fp, iter1_fp, f_values1_fp] = false_position(f, a1, b1, tol1, max_iter);
    [root1_sec, iter1_sec, f_values1_sec] = secant(f, x01, x11, tol1, max_iter);
    [root1_nr, iter1_nr, f_values1_nr] = newton_raphson(f, df, x01, tol1, max_iter);

    % Interval 2
    [root2_bis, iter2_bis, f_values2_bis] = bisection(f, a2, b2, tol1, max_iter);
    [root2_fp, iter2_fp, f_values2_fp] = false_position(f, a2, b2, tol1, max_iter);
    [root2_sec, iter2_sec, f_values2_sec] = secant(f, x02, x12, tol1, max_iter);
    [root2_nr, iter2_nr, f_values2_nr] = newton_raphson(f, df, x02, tol1, max_iter);

    % Print approximated root values and number of iterations for interval 1
    fprintf('Interval 1 (Tol=%.2e):\n', tol1);
    fprintf('Bisection Method Root: %.5f, Iterations: %d\n', root1_bis, iter1_bis);
    fprintf('False Position Method Root: %.5f, Iterations: %d\n', root1_fp, iter1_fp);
    fprintf('Secant Method Root: %.5f, Iterations: %d\n', root1_sec, iter1_sec);
    fprintf('Newton-Raphson Method Root: %.5f, Iterations: %d\n', root1_nr, iter1_nr);

    % Print approximated root values and number of iterations for interval 2
    fprintf('Interval 2 (Tol=%.2e):\n', tol1);
    fprintf('Bisection Method Root: %.5f, Iterations: %d\n', root2_bis, iter2_bis);
    fprintf('False Position Method Root: %.5f, Iterations: %d\n', root2_fp, iter2_fp);
    fprintf('Secant Method Root: %.5f, Iterations: %d\n', root2_sec, iter2_sec);
    fprintf('Newton-Raphson Method Root: %.5f, Iterations: %d\n', root2_nr, iter2_nr);

    % Bisection Method for both thresholds
    [root_bis_tol1, iter_bis_tol1, f_values_bis_tol1] = bisection(f, a1, b1, tol1, max_iter);
    [root_bis_tol2, iter_bis_tol2, f_values_bis_tol2] = bisection(f, a1, b1, tol2, max_iter);
    fprintf('\nBisection Method:\n');
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol1, root_bis_tol1, iter_bis_tol1);
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol2, root_bis_tol2, iter_bis_tol2);

    % False Position Method for both thresholds
    [root_fp_tol1, iter_fp_tol1, f_values_fp_tol1] = false_position(f, a1, b1, tol1, max_iter);
    [root_fp_tol2, iter_fp_tol2, f_values_fp_tol2] = false_position(f, a1, b1, tol2, max_iter);
    fprintf('\nFalse Position Method:\n');
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol1, root_fp_tol1, iter_fp_tol1);
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol2, root_fp_tol2, iter_fp_tol2);

    % Secant Method for both thresholds
    [root_sec_tol1, iter_sec_tol1, f_values_sec_tol1] = secant(f, x01, x11, tol1, max_iter);
    [root_sec_tol2, iter_sec_tol2, f_values_sec_tol2] = secant(f, x01, x11, tol2, max_iter);
    fprintf('\nSecant Method:\n');
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol1, root_sec_tol1, iter_sec_tol1);
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol2, root_sec_tol2, iter_sec_tol2);

    % Newton-Raphson Method for both thresholds
    [root_nr_tol1, iter_nr_tol1, f_values_nr_tol1] = newton_raphson(f, df, x01, tol1, max_iter);
    [root_nr_tol2, iter_nr_tol2, f_values_nr_tol2] = newton_raphson(f, df, x01, tol2, max_iter);
    fprintf('\nNewton-Raphson Method:\n');
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol1, root_nr_tol1, iter_nr_tol1);
    fprintf('Root (Tol=%.2e): %.5f, Iterations: %d\n', tol2, root_nr_tol2, iter_nr_tol2);
end

% Initial interval and guesses for each method
a1 = -10; b1 = 0;   % First interval for Bisection/False Position
x01 = -10; x11 = -9; % Initial guesses for Secant/Newton-Raphson
a2 = -2; b2 = 10;   % Second interval for Bisection/False Position
x02 = -5; x12 = 12;  % Initial guesses for Secant/Newton-Raphson

tol1 = 1e-2;        % First threshold (10^-2)
tol2 = 1e-4;        % Second threshold (10^-4)
max_iter = 100;

% Call the root approximation
root_approximation(f, df, a1, b1, x01, x11, a2, b2, x02, x12, tol1, tol2, max_iter);


