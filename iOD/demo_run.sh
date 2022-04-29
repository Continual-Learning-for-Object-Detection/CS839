
#!/bin/bash
filename='datasets/VOC2007/ImageSets/Main/test.txt'
n=1
while read line; do
# reading each line
echo "datasets/VOC2007/JPEGImages/$line.jpg"
python demo/demo.py --config-file configs/PascalVOC-Detection/iOD/ft_2_p_2.yaml \
  --input "datasets/VOC2007/JPEGImages/$line.jpg" \
  --output results_demo \
  --opts MODEL.WEIGHTS output/2_p_2_ft_DN/model_final.pth

done < $filename