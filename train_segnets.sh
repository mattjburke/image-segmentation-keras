echo "beginning all segnets"

echo "beginning training segnet"
python3 train_segnet.py
echo "finished training and evaluating segnet"

echo "beginning training vgg_segnet"
python3 train_vgg_segnet.py
echo "finished training and evaluating vgg_segnet"

echo "beginning training resnet50segnet"
python3 train_resnet50_segnet.py
echo "finished training and evaluating resnet50_segnet"

echo "beginning training mobilenet_segnet"
python3 train_mobilenet_segnet.py
echo "finished training and evaluating mobilenet_segnet"

echo "finished all segnets"