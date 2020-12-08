rm Analyzed_Data/StrobeData.txt
for file in $(ls -d 1min*)
do 
#echo "Analyzing $file"
python Packet_analyzer_TestStrobe.py -f $file -p -A DB -d 1 100
done
