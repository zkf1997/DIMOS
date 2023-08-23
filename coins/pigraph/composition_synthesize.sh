interactions=('sit on-chair+touch-table' 'sit on-sofa+touch-table'
                              'stand on-floor+touch-board_panel' 'stand on-floor+touch-table'
                      'stand on-floor+touch-tv_monitor' 'stand on-floor+touch-shelving' 'stand on-floor+touch-wall')
for ((i = 0; i < ${#interactions[@]}; i++))
  do
    echo ${interactions[$i]}
#    python synthesize.py --use_penetration 1 --composition 1 --visualize 0 --gender neutral --num_results 32 --num_skeletons 8 --num_translations 256 --num_rotations 12 --interaction "${interactions[$i]}"
    source ../cluster/launch_cluster.sh python synthesize.py --use_penetration 0 --composition 1 --visualize 0 --gender neutral --num_results 32 --num_skeletons 8 --num_translations 256 --num_rotations 12 --interaction \'${interactions[$i]}\' --save_dir pigraph_nopene_composition
  done