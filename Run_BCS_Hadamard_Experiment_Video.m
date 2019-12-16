clear;
close all;
clc;
profile on

%			This file is simulation script for 
%		A Measurment Coding System for Block-based 
%	Compressive Sensing Images by Using Pixel-Domain Features
%   Originally written by Jirayu Peetakul, Hosei University, Japan
%				For further libraries files. 
%	Please contact jirayu.peetakul.6u@stu.hosei.ac.jp
%

path(path, './l1magic/Optimization');
path(path, './l1magic/Measurements');
path(path, './l1magic/Data');
path(path, './images/Original_images');
path(path, './images/Demo');
path(path, './images/Reconstruct_images');
path(path, './dualcodes');

imageOriginalPath = './images/Demo/';
imageFiles = [dir(fullfile(imageOriginalPath,'*png'));
              dir(fullfile(imageOriginalPath,'*tiff'));
              dir(fullfile(imageOriginalPath,'*bmp'));
              dir(fullfile(imageOriginalPath,'*mat'))];
numFiles = length(imageFiles);

%___SIMULATION SETUPS___
calculation_time = 0;
sub_pixels       = 4;
n                = sub_pixels*sub_pixels; % NOTE: small error still present after increasing m;
bits_shift       = [4];
bpp_buffer       = 0;
huffman_symbols  = [0:1:256]; % Distinct data symbols appearing in sig
total            = sum(huffman_symbols);

measurement_matrix_lists        = [4];
measurement_matrix_construction = 'binary_hadamard';
image_reconstruction_algorithm  = 'l1_eq_pd';
image_transformation_algorithm  = 'ifwht';
localpath = strcat('./images/Original_images', ...
                   measurement_matrix_construction, '/', ...
                   image_reconstruction_algorithm, '/', ...
                   image_transformation_algorithm, '/');

image_psnr = zeros(1,length(bits_shift));
image_ssim = zeros(1,length(bits_shift));

for image_loop = 1:numFiles  %6 8 9 10
    %___LOAD IMAGE___
    image_loop
    load_image = imread(imageFiles(image_loop).name);
    if size(load_image,3)
        original_image_red   = load_image(:,:,1); % Red channel
        original_image_green = load_image(:,:,2); % Green channel
        original_image_blue  = load_image(:,:,3); % Blue channel
    else
        original_image = double(load_image);
    end
    
    for matrix_loop = 1:length(measurement_matrix_lists)
        switch measurement_matrix_lists(matrix_loop)
            case {12, 48, 192}
                m = measurement_matrix_lists(matrix_loop);
                sampling_rate = 0.75;

            case {8, 32, 128}
                m = measurement_matrix_lists(matrix_loop);
                sampling_rate = 0.50;

            case {4, 16, 64}
                m = measurement_matrix_lists(matrix_loop);
                sampling_rate = 0.25;
                
            case {2}
                m = measurement_matrix_lists(matrix_loop);
                sampling_rate = 0.125;
                
            case {1}
                m = measurement_matrix_lists(matrix_loop);
                sampling_rate = 0.0625;
        end

        for bits_shift_loop = 1:length(bits_shift)
            switch measurement_matrix_construction
                case 'binary_random'
                    phi                      = randi([0, 1], [m, n]); % This will give you a friendly measurement matrix (M must smaller than N)
                case 'binary_hadamard'
                    hadamard_matrix          = hadamard(n);
                    HadIdx                   = 0:n-1;                          % Hadamard index
                    M                        = log2(n)+1;                           % Number of bits to represent the index
                    binHadIdx                = fliplr(dec2bin(HadIdx,M))-'0'; % Bit reversing of the binary index
                    binSeqIdx                = zeros(n,M-1);                  % Pre-allocate memory
                    for k = M:-1:2
                        % Binary sequency index 
                        binSeqIdx(:,k) = xor(binHadIdx(:,k),binHadIdx(:,k-1));
                    end
                    SeqIdx                   = binSeqIdx*pow2((M-1:-1:0)');    % Binary to integer sequency index
                    walshMatrix              = hadamard_matrix(SeqIdx+1,:); % 1-based indexing
                    phi                      = max(walshMatrix(1:m,1:n), 0);
                case 'guassian'
                    phi                      = randn(m,n);
            end
            
            %___THETA___
            %___NOTE: Avoid calculating Psi (nxn) directly to avoid memory issues___
            theta = zeros(m,n);
            for theta_loop = 1:n
                ek = zeros(1,n);
                ek(theta_loop) = 1;
                switch image_transformation_algorithm
                    case 'idct'
                        psi = idct2(ek)';
                    case 'ifwht'
                        psi = ifwht(ek)';
                    case ''
                end
                theta(:,theta_loop)      = phi*psi;
            end

            %___RESET STATE___

            N_1 = zeros(1, size(load_image,1)/sub_pixels) + sub_pixels;
            N_2 = zeros(1, size(load_image,2)/sub_pixels) + sub_pixels;

            C_red = mat2cell(double(original_image_red), N_1, N_2);
            C_green = mat2cell(double(original_image_green), N_1, N_2);
            C_blue = mat2cell(double(original_image_blue), N_1, N_2);
            
            y_buffer_up    = zeros((m), size(load_image,1)/sub_pixels);
            y_buffer_left  = zeros(m, 1);
            y_buffer_dc    = zeros(m, 1);
            y_buffer_cp    = (zeros(m, 1)) + (0);

            previous_image_block = cell(1);
            previous_image_block{1,1} = zeros(sub_pixels, sub_pixels);
            tic
            
            %%___RED___%%
            %___THE RANDOM PROJECTION (COMPRESSION)___
            for rows = 1:size(load_image,1)/sub_pixels
                for columns = 1:size(load_image,2)/sub_pixels
                    one_block_image_red = reshape(C_red{rows,columns}.',1,[])';
                    y = BCS_encoder(one_block_image_red, phi);

                    [SAD y_prediction, y_buffer_left, y_buffer_up] = intra_prediction(y, ...
                                                                                      phi, ...
                                                                                      rows, ...
                                                                                      columns, ...
                                                                                      sub_pixels, ...
                                                                                      m, ...
                                                                                      n, ...
                                                                                      y_buffer_left, ...
                                                                                      y_buffer_up, ...
                                                                                      y_buffer_cp, ...
                                                                                      y_buffer_dc);
                    y_residual = (y-y_prediction);
                    y_quantized = floor(bitsra(y_residual, bits_shift(bits_shift_loop)));

                    %___DE-QUANTIZED___
                    y_dequantized = floor(bitsll(y_quantized, bits_shift(bits_shift_loop)));
                    y_deresidual = (y_dequantized+y_prediction);
                    [reconstructed_image{rows, columns}, MIN_ENERGY{rows, columns}, L1_NORM{rows, columns}] = BCS_reconstruction(y, ...
                                                                                                                                 y_deresidual, ...
                                                                                                                                 phi, ...
                                                                                                                                 rows, ...
                                                                                                                                 columns, ...
                                                                                                                                 theta, ...
                                                                                                                                 image_transformation_algorithm, ...
                                                                                                                                 image_reconstruction_algorithm, ...
                                                                                                                                 sub_pixels);
                    
                end
            end
            %___FINAL PROCESS ZONE___%
            final_image_reconstruct_red = medfilt2(cell2mat(reconstructed_image), [3 2], 'symmetric');
            %%___RED___%%
            
            %%___GREEN___%%
            %___THE RANDOM PROJECTION (COMPRESSION)___
            for rows = 1:size(load_image,1)/sub_pixels
                for columns = 1:size(load_image,2)/sub_pixels
                    one_block_image_green = reshape(C_green{rows,columns}.',1,[])';
                    y = BCS_encoder(one_block_image_green, phi);

                    [SAD y_prediction, y_buffer_left, y_buffer_up] = intra_prediction(y, ...
                                                                                      phi, ...
                                                                                      rows, ...
                                                                                      columns, ...
                                                                                      sub_pixels, ...
                                                                                      m, ...
                                                                                      n, ...
                                                                                      y_buffer_left, ...
                                                                                      y_buffer_up, ...
                                                                                      y_buffer_cp, ...
                                                                                      y_buffer_dc);
                    y_residual = (y-y_prediction);
                    y_quantized = floor(bitsra(y_residual, bits_shift(bits_shift_loop)));

                    %___DE-QUANTIZED___
                    y_dequantized = floor(bitsll(y_quantized, bits_shift(bits_shift_loop)));
                    y_deresidual = (y_dequantized+y_prediction);
                    [reconstructed_image{rows, columns}, MIN_ENERGY{rows, columns}, L1_NORM{rows, columns}] = BCS_reconstruction(y, ...
                                                                                                                                 y_deresidual, ...
                                                                                                                                 phi, ...
                                                                                                                                 rows, ...
                                                                                                                                 columns, ...
                                                                                                                                 theta, ...
                                                                                                                                 image_transformation_algorithm, ...
                                                                                                                                 image_reconstruction_algorithm, ...
                                                                                                                                 sub_pixels);
                    
                end
            end
            %___FINAL PROCESS ZONE___%
            final_image_reconstruct_green = medfilt2(cell2mat(reconstructed_image), [3 2], 'symmetric');
            %%___GREEN___%%
            
            %%___BLUE___%%
            %___THE RANDOM PROJECTION (COMPRESSION)___
            for rows = 1:size(load_image,1)/sub_pixels
                for columns = 1:size(load_image,2)/sub_pixels
                    one_block_image_blue = reshape(C_blue{rows,columns}.',1,[])';
                    y = BCS_encoder(one_block_image_blue, phi);

                    [SAD y_prediction, y_buffer_left, y_buffer_up] = intra_prediction(y, ...
                                                                                      phi, ...
                                                                                      rows, ...
                                                                                      columns, ...
                                                                                      sub_pixels, ...
                                                                                      m, ...
                                                                                      n, ...
                                                                                      y_buffer_left, ...
                                                                                      y_buffer_up, ...
                                                                                      y_buffer_cp, ...
                                                                                      y_buffer_dc);
                    y_residual = (y-y_prediction);
                    y_quantized = floor(bitsra(y_residual, bits_shift(bits_shift_loop)));

                    %___DE-QUANTIZED___
                    y_dequantized = floor(bitsll(y_quantized, bits_shift(bits_shift_loop)));
                    y_deresidual = (y_dequantized+y_prediction);
                    [reconstructed_image{rows, columns}, MIN_ENERGY{rows, columns}, L1_NORM{rows, columns}] = BCS_reconstruction(y, ...
                                                                                                                                 y_deresidual, ...
                                                                                                                                 phi, ...
                                                                                                                                 rows, ...
                                                                                                                                 columns, ...
                                                                                                                                 theta, ...
                                                                                                                                 image_transformation_algorithm, ...
                                                                                                                                 image_reconstruction_algorithm, ...
                                                                                                                                 sub_pixels);
                    
                end
            end
            %___FINAL PROCESS ZONE___%
            final_image_reconstruct_blue = medfilt2(cell2mat(reconstructed_image), [3 2], 'symmetric');
            %%___BLUE___%%
            
            video_buffer(:,:,:,image_loop) = cat(3, final_image_reconstruct_red, final_image_reconstruct_green, final_image_reconstruct_blue);
        end
    end
end
calculation_time/numFiles

video_out = VideoWriter('demo_RGB_video_hadamard_4_Q4_medfilter33.avi', 'Uncompressed AVI'); %create the video object
open(video_out); %open the file for writing
for loop = 1:numFiles
    writeVideo(video_out, uint8(video_buffer(:,:,:,loop)));
end
close(video_out);
profile report
profile off