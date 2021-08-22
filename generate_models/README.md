To generate models we need to run the file `gen_data.py`, which takes two arguments 

`arg1=istart` starting point, for example, if we have 1000 models already generated `istart=10001` 


`arg2=num_model` number of models to generate 

# Run in Ibex

  1. **Log in to a GPU login node**
     
  `$ ssh -X <USERNAME>@glogin.ibex.kaust.edu.sa`
    
  2. **Clone the repository**
  
  ```
  $ git clone https://github.com/alaliaa/Intellegent_unflooding_for_salt_unet.git
  ```

  3. **Navigate to  `generate_models` folder and run**
  
  `$ sbatch run.sh` 


# slurm job 
Important notes 

`#SBATCH --array=start-end ` this define the number of jobs that will be implimented. Each job will be given an ID stored in `$SLURM_JOB_ID` and a task id stored in `$SLURM_ARRAY_TASK_ID`

`$SLURM_ARRAY_TASK_ID`  is used to define the starting point of `gen_data.py`

