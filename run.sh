#python deep_sort_app.py --sequence_dir=./MOT16/test/MOT16-06 --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True

#python generate_detections.py --model=resources/networks/mars-small128.ckpt --mot_dir=./MOT16/test --output_dir=./resources/detections/MOT16_test
python generate_detections.py --mot_dir=./MOT16/test --output_dir=./resources/detections/MOT16_test
#python deep_sort_app.py --sequence_dir=./MOT16/test/MOT16-06 --detection_file=./resources/detections/MOT16_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True


# produce sense time's detection for MOT16 test
if false; then
for i in `ls resources/detections/MOT16_POI_test/ | grep npy`
do
j=`basename $i .npy`
mkdir -p ./MOT16/test/$j/st;
python npy2det.py resources/detections/MOT16_POI_test/$i ./MOT16/test/$j/st/st.txt
echo $i
done
fi

# produce sense time's detection for MOT16 train
if false; then
for i in `ls resources/detections/MOT16_POI_train/ | grep npy`
do
j=`basename $i .npy`
mkdir -p ./MOT16/train/$j/st;
python npy2det.py resources/detections/MOT16_POI_train/$i ./MOT16/train/$j/st/st.txt
echo $i
done
fi


# produce features of MOT16 with sense time's detections
if false; then
mkdir -p ./resources/detections/MOT16_SORT_test
python generate_detections.py --model=resources/networks/mars-small128.ckpt --mot_dir=./MOT16/test --output_dir=./resources/detections/MOT16_SORT_test
mkdir -p ./resources/detections/MOT16_SORT_train
python generate_detections.py --model=resources/networks/mars-small128.ckpt --mot_dir=./MOT16/train --output_dir=./resources/detections/MOT16_SORT_train
fi


# track with POI's feature
if false; then
mkdir -p outputs/MOT16_POI_train
for i in `ls ./MOT16/train/`
do
  echo $i
  python deep_sort_app.py --sequence_dir=./MOT16/train/$i --detection_file=./resources/detections/MOT16_POI_train/$i.npy --min_confidence=0.3 --nn_budget=100 --display=False --output_file=outputs/MOT16_POI_train/$i.txt
done
fi

# track with SORT's feature
if false; then
mkdir -p outputs/MOT16_SORT_train
for i in `ls ./MOT16/train/`
do
  echo $i
  python deep_sort_app.py --sequence_dir=./MOT16/train/$i --detection_file=./resources/detections/MOT16_SORT_train/$i.npy --min_confidence=0.3 --nn_budget=100 --display=False --output_file=outputs/MOT16_SORT_train/$i.txt
done
fi
