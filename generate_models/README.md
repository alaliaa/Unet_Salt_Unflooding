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


# run.sh (Slurm job)
Important notes 

`#SBATCH --array=start-end ` this define the number of jobs that will be implimented. Each job will be given an ID stored in `$SLURM_JOB_ID` and a task id stored in `$SLURM_ARRAY_TASK_ID`

`$SLURM_ARRAY_TASK_ID`  is used to define the starting point of `gen_data.py`

for now, I am loading the deepwave and madagascar from ibex. it might be better if we create an environment instead

# fwi.py 
Important notes, 
  1. In the function `get_coordinate` for the purpose of the implimintation, the origin of the shot and the receiver were set manually and need to be changed 
  2. set `alphaTV=0` in case TV-regulization is not wanted. 
  3. smoothing is set manually 

# model_generator.py
This is the main engine to generate the random models. In the current implimintation I am usin the mean valuse of BP model to generate the random models.

