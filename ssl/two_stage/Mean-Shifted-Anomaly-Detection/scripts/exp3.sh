# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
#data='rsna'
gpu=7

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python main.py -d isic -g $gpu --backbone 18 --epochs 100;
python main.py -d brats -g $gpu --backbone 18 --epochs 100;
done