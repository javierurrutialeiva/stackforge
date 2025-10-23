N=$1
echo "Downloading snap $N1"
tar -cvf snap_$N.tar /home/tnguser//sims.TNG/TNG300-3/output/snapdir_$N/*
rsync -avh snap_$N.tar javierul@login.sherlock.stanford.edu:/scratch/users/javierul
echo "Downloading group $N1"
tar -cvf group_$N.tar /home/tnguser/sims.TNG/TNG300-3/output/groups_$N/*
rsync -avh group_$N.tar javierul@login.sherlock.stanford.edu:/scratch/users/javierul
