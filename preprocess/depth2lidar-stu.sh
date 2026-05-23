
set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  /root/autodl-tmp/stu_dataset/$num_seq \
    --depth_dir /root/autodl-tmp/stu_dataset/sequences_sql_depth/$num_seq \
    --save_dir /root/autodl-tmp/stu_dataset/sequences_sql_lidar/$num_seq
}
# exeFunc 125
exeFunc 144
# for i in {00..06}
# do
#     exeFunc $i
# done
