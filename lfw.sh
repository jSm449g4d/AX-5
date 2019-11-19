
echo "download from \"Labeled Faces in the Wild\""
URL=http://vis-www.cs.umass.edu/lfw/lfw.tgz
echo $URL
wget -N $URL
tar zxf lfw.tgz
rm lfw.tgz
