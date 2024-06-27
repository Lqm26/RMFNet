# You should first set the path to your dataset and the output path in the config, and then run the following command based on your needs.
eventNFS/nfs-syn/RGB-syn
python train.py -c config/train_EventNFS.yml
python train.py -c config/train_nfs_syn.yml
python train.py -c config/train_RGB_syn.yml.yml