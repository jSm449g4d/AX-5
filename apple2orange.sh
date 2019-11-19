
echo "download from \"Index of /~taesung_park/CycleGAN/datasets\""
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip
echo $URL
wget -N $URL
unzip apple2orange.zip
rm apple2orange.zip
