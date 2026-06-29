A=medicalVolume('70');
A=squeeze(A);
alpha = [0 0 0.72 1.0];
color = [0 0 0; 186 65 77; 231 208 141; 255 255 255] ./ 255;
intensity = [-3024 100 400 1499];
queryPoints = linspace(min(intensity),max(intensity),256);
alphamap = interp1(intensity,alpha,queryPoints)';
colormap = interp1(intensity,color,queryPoints);
vol = volshow(A,Colormap=colormap,Alphamap=alphamap,RenderingStyle="CinematicRendering");