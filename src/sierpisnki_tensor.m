% Parameters
grid_size = 150; % Size of each slice (grid_size x grid_size)
num_slices = 20; % Number of zoom levels (depth of tensor)

% Initialize the tensor
sierpinski_tensor = zeros(grid_size, grid_size, num_slices);

% Loop through each zoom level
for slice_idx = 1:num_slices
    zoom = 2^(slice_idx - 1); % Exponential zoom-out factor

    % Create a fine grid and calculate modulo conditions
    [X, Y] = meshgrid(linspace(0, 1, grid_size * zoom), linspace(0, 1, grid_size * zoom));
    condition = mod(floor(X * zoom) + floor(Y * zoom), 2) == 0;

    % Downsample to the desired grid size
    sierpinski_slice = condition(1:zoom:end, 1:zoom:end);

    % Store in the tensor
    sierpinski_tensor(:, :, slice_idx) = sierpinski_slice;
end

% Visualize slices
for slice_idx = 1:num_slices
    figure;
    imagesc(sierpinski_tensor(:, :, slice_idx));
    colormap gray;
    axis image off;
    title(['Sierpinski Triangle - Slice ' num2str(slice_idx)]);
end