To generate models we need to run the file ''gen_data.py"

To run this in ibex please do the following step: 
  1. **Log in to a GPU login node**
     
  `$ ssh -X <USERNAME>@glogin.ibex.kaust.edu.sa`
    
  2. **Clone the repository**
  
  ```
  $ git clone https://github.com/alaliaa/Intellegent_unflooding_for_salt_unet.git
  ```

  3. **Navigate to '' generate_models folder and run**
  
  `$ sbatch run.sh` 
