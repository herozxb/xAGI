python convert.py -s data/
python train.py -s data/
cd /home/deep/gaussian_splatting/SIBR_viewers/install/bin
./SIBR_gaussianViewer_app -m ../../../output/2aa78b4d-d/


sudo ffmpeg -i 0116_1.mov -vf "fps=2" frame_%04d.jpg

