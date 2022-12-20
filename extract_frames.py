from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
import os
from datetime import timedelta
import logging
import argparse

# Code is based on: https://www.thepythoncode.com/article/extract-frames-from-videos-in-python


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def main_logic(video_file, fps, output_path):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    filename, _ = os.path.splitext(video_file)
    filename += "-moviepy"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # if the fps value is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, int(fps))
    # if fps value is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = (
        1 / video_clip.fps
        if saving_frames_per_second == 0
        else 1 / saving_frames_per_second
    )
    # iterate over each possible frame
    df = pd.DataFrame(
        columns=[
            "video_name",
            "video_id",
            "frame_name",
            "frame_no",
            "time_start",
            "time_end",
            "label",
        ]
    )
    counter = 1
    logging.info("Extracting frames from video: %s", video_file)
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration))
        frame_filename = os.path.join(
            output_path,
            f"{video_file.split('/')[-1]}_{counter}_{frame_duration_formatted}.jpg",
        )
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)

        new_row = {
            "video_name": video_file,
            "frame_no": counter,
            "frame_name": frame_filename,
            "time_start": timedelta(seconds=current_duration),
            "time_end": timedelta(seconds=current_duration + step),
        }
        df_temp = pd.DataFrame(new_row, index=[0])
        df = pd.concat([df, df_temp], ignore_index=True)
        counter = counter + 1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fps",
        action="store",
        help="number of frames to be extracted per second",
        default=30,
    )
    parser.add_argument(
        "--input_path",
        help="give the path to the folder that contains the videos",
        action="store",
        default=".",
    )
    parser.add_argument("--output_path", action="store", default="Extracted_frames")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # assign directory
    directory = args.input_path
    # fps to extract
    fps = args.fps
    output_path = args.output_path

    df_merged = pd.DataFrame()
    # iterate over files in chosen directory
    logging.info("Extracting frames from videos in directory: %s", directory)
    counter = 1
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # Check if it's a video file
        if os.path.isfile(f) and (
            f.endswith(".mp4") or f.endswith(".MOV") or f.endswith(".mkv")
        ):
            video_file = f
            df = main_logic(video_file, fps, output_path)
            df["time_start"] = df["time_start"].astype(str).str.replace("0 days ", "")
            df["time_end"] = df["time_end"].astype(str).str.replace("0 days ", "")
            df["video_id"] = counter
            df_merged = pd.concat([df_merged, df], ignore_index=True)
            counter = counter + 1
    df_merged = df_merged.reset_index()
    df_output_path = os.path.join(output_path, "labels.csv")
    df_merged.to_csv(df_output_path, index=False)
    logging.info(
        f"Done! Manual labeling can be added to the csv file: {df_output_path}"
    )


if __name__ == "__main__":
    main()
