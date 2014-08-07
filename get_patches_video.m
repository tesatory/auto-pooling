function [patchesA, patchesB] = get_patches_video(rfSize, scale, data_sz,step)
n = 0;

n = n + 1; video_paths{n} = 'video\vimeo-cc\airplane\airplanes_&_helicopters_16_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\airplane\airplanes_&_helicopters_18_640x372.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\airplane\amelia_earhart_lockheed_l12_at_ageda_evening_640x368.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\airplane\formation_acro_with_team_aeroshell_at_oshkosh_airventure_2011_640x480.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\bird\bird_watching_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\bird\birds_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\bird\birds_and_clouds_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\bird\birds_in_the_backyard_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\car\early_spring_shows_quick_edit_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\car\jay_stellatos_subaru_outback_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\car\mmpower_intro_other_cars_2012_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\car\oscar_raus_mk4_jetta_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\cat\block_f_cats_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\cat\cats_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\cat\mint_floor_cleaner_vs_eero_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\cat\my_cat_sanga_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\deer\deer_at_work_480x272.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\deer\deer_on_a_snowy_day_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\deer\the_bowing_deer_of_nara_osaka_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\deer\wild_deer_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\dog\circle_dog_640x368.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\dog\dogs1_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\dog\dogs_34_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\dog\dogs_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\frog\eastern_dwarf_tree_frogs_640x512.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\frog\orange-eyed_tree_frogs_640x512.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\frog\pine_barrens-_tree_frogs_and_water_640x368.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\frog\tree_frog_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\horse\clydesdale_horse_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\horse\horse_compilation_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\horse\horses_03_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\horse\horses_reel_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\nature\be_a_pioneer_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\nature\drifters_of_the_deep_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\nature\finca_bellavista_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\nature\frozen_silence_640x360.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\ship\and_the_ship_sails_on_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\ship\grandes_veleiros___tall_ships_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\ship\port_of_san_diego_festival_of_sail_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\ship\uss_oriskany_sinking_640x480.mp4';

n = n + 1; video_paths{n} = 'video\vimeo-cc\truck\altec_lrv56-_2000_gmc_c8500_forestry_truck_640x372.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\truck\kenway_dump_truck_next_door_640x360.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\truck\serco_8500-_1999_international_8100_6x4_grapple_truck_640x372.mp4';
n = n + 1; video_paths{n} = 'video\vimeo-cc\truck\vehicle_08_640x360.mp4';

for n = 1:length(video_paths)
    fprintf('%s\n', video_paths{n});
    vr_list{n} = VideoReader(video_paths{n});
end

rfSize = rfSize * scale;
patchesA = zeros(rfSize,rfSize,3,data_sz);
patchesB = zeros(rfSize,rfSize,3,data_sz);

n = 1;
while n <= data_sz    
    fprintf('processing %d\n',n);
    
    % select random video
    vr = vr_list{randi(length(video_paths))};
    frame_w = vr.Width;
    frame_h = vr.Height;
    batchsz = 100; % cant process at once because of low memory
    
    % avoid texts in start and end
    frame_start = round(vr.NumberOfFrames * 0.2);
    frame_end = vr.NumberOfFrames - round(vr.NumberOfFrames * 0.1);

    frames = read(vr,[1 batchsz] + randi([frame_start frame_end-batchsz]));
    frames = double(frames) / 256;
    
    for k = 1:batchsz*10
        % patches at random times
        fn = randi(batchsz - step);
        
        % check continuety
        if mean(mean(mean(abs(frames(:,:,:,fn) ...
                - frames(:,:,:,fn+step))))) > 0.05
            continue
        end
        
        % patches at random positions
        offx = randi([0 frame_w-rfSize]);
        offy = randi([0 frame_h-rfSize]);
        
        A = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn);
        B = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn + step);
        
        % check if it is all same color
        if min(std(A(:)), std(B(:))) < 0.04 
            continue
        end
        
        % check if A = B
        if std((A(:)-B(:))) < 0.04
            continue
        end
        
        patchesA(:,:,:,n) = A;
        patchesB(:,:,:,n) = B;
        n = n + 1;
        if n > data_sz
            break
        end
    end
end

rfSize = rfSize / scale;
patchesA2 = zeros(rfSize,rfSize,3,data_sz);
patchesB2 = zeros(rfSize,rfSize,3,data_sz);
for i = 1:data_sz
    patchesA2(:,:,:,i) = imresize(patchesA(:,:,:,i),1/scale);
    patchesB2(:,:,:,i) = imresize(patchesB(:,:,:,i),1/scale);
end
patchesA = reshape(patchesA2, rfSize^2*3, data_sz)';
patchesB = reshape(patchesB2, rfSize^2*3, data_sz)';
