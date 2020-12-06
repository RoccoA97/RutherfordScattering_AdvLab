for file in $(ls ./1min*)
do 
#echo "Analyzing $file"
python Packet_analyzer.py -f $file -p -A DB -d 1 100
done
