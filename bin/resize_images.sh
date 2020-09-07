

for x in /eye2gene/data/heyex/*/*png
do
	y=`echo ${x} | sed 's/heyex/heyex_small/'`
	y2=`dirname $y`
	mkdir -p $y2
	convert $x -resize 256x256 $y
done

