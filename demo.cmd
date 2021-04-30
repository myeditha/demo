executable = ./run_demo.sh
log = ./log
notification = complete
output = ./condor.out
error = ./condor.error
Requirements = ( Machine != "patas-n3.ling.washington.edu" )
arguments = "/home2/myeditha/anaconda3/envs/conda_env"
queue