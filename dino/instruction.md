sudo ffmpeg -i 0116_1.mov -vf "fps=1" frame_%04d.jpg
python run_video.py --encoder vits --video-path assets/examples_video/davis_dolphins.mp4 --outdir video_depth_vis
