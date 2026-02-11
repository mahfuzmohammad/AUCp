# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=1
#data='rsna'
gpu=0

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python panda.py -d rsna -g $gpu --epochs 100 --ewc;
python panda.py -d vin -g $gpu --epochs 100 --ewc;
done