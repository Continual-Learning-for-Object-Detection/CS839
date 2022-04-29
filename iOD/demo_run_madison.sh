
#!/bin/bash
filename='/zdata/users/chwu/file_list.txt'
n=1
while read line; do
# reading each line
echo "$line"
python demo/demo.py --config-file configs/PascalVOC-Detection/iOD/ft_2_p_2.yaml \
  --input "$line" \
  --output madison_results_demo\
  --opts MODEL.WEIGHTS output/2_p_2_ft_DN/model_final.pth

done < $filename