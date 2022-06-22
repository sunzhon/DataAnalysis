#! /bin/zsh

if [ $# -gt 0 ]; then
    dir_path=$1
else
    dir_path=${HOME}
fi

result_folder="$dir_path/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing"
dataset_folder="$dir_path/Drop_landing_workspace/suntao/"

if [[ ! -d $result_folder ]]; then
    mkdir -p $result_folder
fi

if [[ ! -d $dataset_folder ]]; then
    mkdir -p $dataset_folder
fi


remote_rawdataset_folder="/mnt/sun/My Passport/Drop_landing_workspace/suntao/D drop landing/"
local_1_rawdataset_folder="/media/sun/My Passport/Drop_landing_workspace/suntao/D drop landing/"

if [[ -d $remote_rawdataset_folder ]]; then
    cp -r "/mnt/sun/My Passport/Drop_landing_workspace/suntao/D drop landing/" $dataset_folder
elif [[ -d $local_1_rawdataset_folder ]]; then
    cp -r "/media/sun/My Passport/Drop_landing_workspace/suntao/D drop landing/" $dataset_folder
else
    cp -r "/media/sun/DATA/Drop_landing_workspace/suntao/D drop landing/" $dataset_folder
fi

export MEDIA_NAME="$dir_path"
