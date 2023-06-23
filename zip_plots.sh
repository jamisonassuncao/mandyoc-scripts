DIRNAME=$(basename "$(dirname "$PWD")")

echo "Zipping $DIRNAME directory..."

zip $DIRNAME'_imgs.zip' *.png
zip $DIRNAME'_videos.zip' *.mp4
zip $DIRNAME'_gifs.zip' *.gif

rm *.png

echo "Zipping completed!"