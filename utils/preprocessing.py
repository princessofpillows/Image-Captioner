
import cv2, glob, h5py, random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize


def check_data(data_dir):
    # Ensure searching a valid directory
    if not Path(data_dir).is_dir():
        print('Error: path to', data_dir, 'does not exist, change data_dir in config.py to a valid directory')
        return

    videos = list(data_dir.glob('*.mp4'))
    if not videos:
        print('Error: data directory ' + data_dir + ' does not contain data in required format in all folders')
        return
    
    # Create array of random values, where length and range of s = length of datasets
    s = np.arange(np.asarray(videos).shape[0])
    np.random.shuffle(s)
    # Shuffle data randomly
    videos = np.asarray(videos)[s]

    return videos


def package_data(image_size, data_dir):

    data_dir = Path(data_dir)
    videos = check_data(data_dir)

    # Open file for r/w ('a' specifies not to overwrite)
    h5f = h5py.File('data/videoData.h5', 'a')

    videodata = []
    infodata = []

    # Loops through all videos
    for i in tqdm(range(len(videos))):

        video = cv2.VideoCapture(str(videos[i]))

        # Ensure video opens successfully
        if not video.isOpened():
            print('bad video')
            video.release()

        # Get framerate
        fps = int(np.rint(video.get(cv2.CAP_PROP_FPS)))

        count = 0
        # Set refresh rate to 3hz
        hz = fps / 3
        # Set rate for random distortions
        bug = random.randint(1, fps)

        # Process video frame by frame
        while video.isOpened():
            # Get frame of video
            ret, frame = cv2.VideoCapture.read(video)
                
            # Check if reached end of video
            if ret != True:
                break

            # Record frame at 3hz with downsampled resolution
            if int(count % hz) == 0:
                frame = resize(frame, image_size, preserve_range=True).astype('uint8')
                info = 0
                # Apply random transformations
                if int(count % bug) == 0:
                    bug = random.randint(1, fps)
                    distort = np.random.randint(25, size=frame.shape, dtype='uint8')
                    choice = random.uniform(0, 1)

                    if choice > 0.75:
                        frame -= distort
                    elif choice > 0.5:
                        frame += distort
                    elif choice > 0.25:
                        frame *= (0.01*distort).astype('uint8')
                    elif choice > 0:
                        frame *= (0.01 / (distort + 0.01)).astype('uint8')
                    #cv2.imshow('newImage', frame)
                    #cv2.waitKey(0)
                    info = 1

                videodata.append(frame)
                infodata.append(info)

            # Count frames to ensure 3hz
            count += 1

        # Close video object
        video.release()

    # Get data ready to write
    video_data = np.asarray(videodata)
    info_data = np.asarray(infodata)

    # Write data
    try:
        h5f.create_dataset('videos', data=video_data, dtype='uint8')
        h5f.create_dataset('info', data=info_data)
    # If datasets already exist, delete and recreate them
    except RuntimeError:
        print('Warning: datasets already defined, resetting these datasets')
        del h5f['videos']
        del h5f['info']
        h5f.create_dataset('videos', data=video_data, dtype='uint8')
        h5f.create_dataset('info', data=info_data)

    h5f.close()
