#! /bin/zsh

if [ $# -gt 0 ]; then
    testing_folders=$1
else
    testing_folders='.'
fi

list_hyper_files=($(find $testing_folders -name hyperparams.yaml))

data_file="$testing_folders/testing_result_folders.txt"
echo "" > $data_file
echo "START TO COLLECT TEST DATA"

all_sensor_configs=( 'T' 'S' 'F' 'W' 'C'  'FS' 'FT' 'FW' 'FC' 'ST' 'SW' 'SC' 'TW' 'TC' 'WC' 'FST' 'FSW' 'FSC' 'FTW' 'FTC' 'FWC' 'STW' 'STC' 'SWC' 'TWC' 'FSTW' 'FSTC' 'FSWC' 'FTWC' 'STWC' 'FSTWC')

for hyper_file in ${list_hyper_files}; do 
    str=${hyper_file%/*}
    str1=${str##*/}
    echo $str1
    if [[ $str1 =~ "test_[0-9]+" ]];then
        echo ${hyper_file}
        folder_path=$(cd $(dirname $hyper_file); pwd)
        echo ${folder_path}
        lstm=$(awk -F"[ :-]+" '$1~/lstm_units/{print $2}' $hyper_file | grep -o -E "\w+")
        sensors_fields=$(awk -F"[ :-]+" '$2~/Accel_X/{array[$2]++}END{for(i in array)print i}' $hyper_file)
        echo ${lstm} 
        echo ${sensors_fields}
        sensor_configs=$(echo $sensors_fields | grep -o -E "[A-Z]{2,10}")
        echo $sensor_configs
        combined_sensor_config_name=$(echo $sensor_configs | grep -o -E "^[A-Z]|(\s[A-Z])+" | grep -o -E "[A-Z]+" | tr -d '\n')
        echo $combined_sensor_config_name
        for a_sensor_config in ${all_sensor_configs[@]}; do
            n_a_sensor_config=$(echo $a_sensor_config | wc -m)
            n_a_sensor_config=$[$n_a_sensor_config-1]
            temp=$(echo $combined_sensor_config_name | grep -E -o "([$a_sensor_config]){$n_a_sensor_config}") 
            if [ "$temp" = "$combined_sensor_config_name" ]; then
                real_sensor_config_name=$a_sensor_config
                echo "number of sensor name characters" $n_a_sensor_config
                echo "temp combined sensor name: $temp"
                echo "combined sensor config name: $combined_sensor_config_name"
                echo "real sensor config:" $real_sensor_config_name
                echo "${real_sensor_config_name}\t${lstm}\t${folder_path}" >> $data_file
            fi
        done
    fi
done

echo "END TO COLLECT TEST DATA"

