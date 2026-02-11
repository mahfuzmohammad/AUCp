# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
#data='rsna'
gpu=5

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python panda.py -d isic -g $gpu --epochs 100 --ewc;
python panda.py -d brats -g $gpu --epochs 100 --ewc;
done